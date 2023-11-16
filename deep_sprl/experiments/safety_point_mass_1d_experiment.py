import os
import torch
import gymnasium
import numpy as np
from omnisafe.envs.core import make
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import ConstrainedSelfPacedTeacherV2, SelfPacedTeacherV2, \
    ConstrainedSelfPacedWrapper, SelfPacedWrapper#, CurrOT
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.dummy_wrapper import DummyWrapper
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.teachers.util import Subsampler
from deep_sprl.util.utils import update_params
from scipy.stats import multivariate_normal

from deep_sprl.environments.safety_point_mass import ContextualSafetyPointMass1D

class SafetyPointMass1DExperiment(AbstractExperiment):
    PENALTY_COEFFICIENT = {Learner.SAC: 0.0, 
                           Learner.PPO: 1.0, # 0.1, 
                           Learner.PPOLag: 0.0}

    TARGET_TYPE = "narrow"
    TARGET_MEAN = np.array([-ContextualSafetyPointMass1D.ROOM_WIDTH*0.3])
    TARGET_VARIANCES = {
        "narrow": np.square(np.diag([.1])),
        "wide": np.square(np.diag([1.])),
    }

    LOWER_CONTEXT_BOUNDS = np.array([-ContextualSafetyPointMass1D.ROOM_WIDTH*0.3])
    UPPER_CONTEXT_BOUNDS = np.array([ContextualSafetyPointMass1D.ROOM_WIDTH/2])

    def target_log_likelihood(self, cs):
        return multivariate_normal.logpdf(cs, self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE])

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = np.random

        return rng.multivariate_normal(self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE], size=n)

    INIT_VAR = 0.1
    INITIAL_MEAN = np.array([ContextualSafetyPointMass1D.ROOM_WIDTH/2])
    # INITIAL_VARIANCE = np.diag(np.square([0.1, 0.1]))

    DIST_TYPE = "gaussian"  # "cauchy"

    STD_LOWER_BOUND = np.array([0.1])
    KL_THRESHOLD = 8000.
    KL_EPS = 1.0 # 0.5
    DELTA = 30.0 # 20.0 # 10.0
    DELTA_C = 0.0
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 10 # 20 # 100 # 200
    
    NUM_ITER = 150 # 250 # 1000 # 500
    STEPS_PER_ITER = 2000 # 4000
    DISCOUNT_FACTOR = 0.99
    LAM = 0.95 # 0.99 

    # ACL Parameters [found after search over [0.05, 0.1, 0.2] x [0.01, 0.025, 0.05]]
    ACL_EPS = 0.2
    ACL_ETA = 0.025

    PLR_REPLAY_RATE = 0.85
    PLR_BUFFER_SIZE = 100
    PLR_BETA = 0.45
    PLR_RHO = 0.15

    VDS_NQ = 5
    VDS_LR = 1e-3
    VDS_EPOCHS = 3
    VDS_BATCHES = 20

    AG_P_RAND = {Learner.PPO: 0.1, Learner.SAC: None}
    AG_FIT_RATE = {Learner.PPO: 100, Learner.SAC: None}
    AG_MAX_SIZE = {Learner.PPO: 500, Learner.SAC: None}

    GG_NOISE_LEVEL = {Learner.PPO: 0.1, Learner.SAC: None}
    GG_FIT_RATE = {Learner.PPO: 200, Learner.SAC: None}
    GG_P_OLD = {Learner.PPO: 0.2, Learner.SAC: None}

    def __init__(self, base_log_dir, curriculum_name, learner_name, parameters, seed, device):
        super().__init__(base_log_dir, curriculum_name, learner_name, parameters, seed, device)
        env_id, teacher_id, wrapper_kwargs = self.create_environment(evaluation=True)
        self.eval_env_id = f"{teacher_id}-{env_id}"
        self.eval_env_wrapper_kwargs = wrapper_kwargs
        self.eval_env = make(self.eval_env_id)
        self.eval_env.initialize_wrapper(**wrapper_kwargs)

    def create_environment(self, evaluation=False):
        env_id = "ContextualSafetyPointMass1D-v0"
        special_kwargs = {}
        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            teacher_id = "DummyTeacher"
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            teacher_id = "ALPGMM"
        elif self.curriculum.goal_gan():
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, 2))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            teacher_id = "GoalGAN"
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            teacher_id = "SelfPaced"
            special_kwargs['episodes_per_update'] = self.EP_PER_UPDATE
        elif self.curriculum.constrained_self_paced():
            teacher = self.create_self_paced_teacher(with_callback=True)
            teacher_id = "ConstrainedSelfPaced"
            special_kwargs['episodes_per_update'] = self.EP_PER_UPDATE
        elif self.curriculum.acl():
            bins = 50
            teacher = ACL(bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            teacher_id = "ACL"
            special_kwargs['context_post_processing'] = Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                     self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                     [bins, bins])
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            teacher_id = "PLR"
            special_kwargs['context_post_processing'] = teacher.post_process
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            teacher_id = "VDS"
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            teacher_id = "DummyTeacher"
        else:
            raise RuntimeError("Invalid learning type")

        wrapper_kwargs = {'log_dir': self.get_log_dir(),
                          'teacher': teacher,
                          'discount_factor': self.DISCOUNT_FACTOR,
                          'step_divider': self.STEPS_PER_ITER,
                          'eval_mode': evaluation,
                          'penalty_coeff': self.PENALTY_COEFFICIENT[self.learner]}
        
        wrapper_kwargs = update_params(wrapper_kwargs, special_kwargs)
        return env_id, teacher_id, wrapper_kwargs

    def create_learner_params(self):
        algo_specific_cfgs = {
            Learner.SAC: {
                'train_cfgs': {
                    'eval_episodes': 0,
                },
                'algo_cfgs': {
                    'steps_per_epoch': self.STEPS_PER_ITER, # to eval, log, actor scheduler step
                    'update_cycles': 5, # train_freq                      
                    'update_iters': 1, # gradient steps
                    'size': 100000,
                    'batch_size': 64,
                    'reward_normalize': False,
                    'cost_normalize': False,
                    'obs_normalize': False,
                    'max_grad_norm': 40,
                    'use_critic_norm': False,
                    'critic_norm_coeff': 0.5, # 0.001,
                    'polyak': 0.005,
                    'gamma': self.DISCOUNT_FACTOR,
                    'start_learning_steps': 10000,
                    'policy_delay': 1, # 2,
                    'use_exploration_noise': False,
                    'exploration_noise': 0.1, # = 0 if use_exploration_noise is False
                    'policy_noise': 0.2,
                    'policy_noise_clip': 0.5,
                    'alpha': 0.2,
                    'auto_alpha': True, # ent_coef
                    'use_cost': False,
                },
                'logger_cfgs': {
                    'window_lens': 10,
                },
                'model_cfgs': {
                    'actor_type': "gaussian_sac",
                    'linear_lr_decay': False,
                },
            },
            Learner.PPO:  {
                'algo_cfgs': {
                    'steps_per_epoch': self.STEPS_PER_ITER, # to eval, log, actor scheduler step
                    'update_iters': 4, #  10, # gradient steps
                    'batch_size': 64, # 128, 
                    'target_kl': 0.02,
                    'entropy_coef': 0.0,
                    'reward_normalize': False,
                    'cost_normalize': False,
                    'obs_normalize': True,
                    'kl_early_stop': True,
                    'use_max_grad_norm': True,
                    'max_grad_norm': 40.0, # 0.5, 
                    'use_critic_norm': True,
                    'critic_norm_coef': 0.001, # 0.5,
                    'gamma': self.DISCOUNT_FACTOR,
                    'cost_gamma': self.DISCOUNT_FACTOR,
                    'lam': self.LAM,
                    'lam_c': self.LAM,
                    'clip': 0.2,
                    'adv_estimation_method': 'gae',
                    'standardized_rew_adv': True,
                    'standardized_cost_adv': True,
                    'penalty_coef': self.PENALTY_COEFFICIENT[self.learner],
                    'use_cost': False,
                },
                'logger_cfgs': {
                    'window_lens': 100,
                },
                'model_cfgs': {
                    'actor_type': "gaussian_learning",
                    'linear_lr_decay': True,
                    'exploration_noise_anneal': False,
                    'std_range': [0.5, 0.1],
                },
            },
            Learner.PPOLag:  {
                'algo_cfgs': {
                    'steps_per_epoch': self.STEPS_PER_ITER, # to eval, log, actor scheduler step
                    'update_iters': 4, # 10, # gradient steps
                    'batch_size': 64, # 128,
                    'target_kl': 0.02,
                    'entropy_coef': 0.0,
                    'reward_normalize': False,
                    'cost_normalize': False,
                    'obs_normalize': True,
                    'kl_early_stop': True,
                    'use_max_grad_norm': True,
                    'max_grad_norm': 40.0, # 0.5,
                    'use_critic_norm': True,
                    'critic_norm_coef': 0.001, # 0.5,
                    'gamma': self.DISCOUNT_FACTOR,
                    'cost_gamma': self.DISCOUNT_FACTOR,
                    'lam': self.LAM,
                    'lam_c': self.LAM,
                    'clip': 0.2,
                    'adv_estimation_method': 'gae',
                    'standardized_rew_adv': True,
                    'standardized_cost_adv': True,
                    'penalty_coef': 0.0,
                    'use_cost': True,
                },
                'logger_cfgs': {
                    'window_lens': 100,
                },
                'model_cfgs': {
                    'actor_type': "gaussian_learning",
                    'linear_lr_decay': True,
                    'exploration_noise_anneal': False,
                    'std_range': [0.5, 0.1],
                },
                'lagrange_cfgs': {
                    # Tolerance of constraint violation
                    'cost_limit': self.DELTA_C,
                    # Initial value of lagrangian multiplier
                    'lagrangian_multiplier_init': 0.001,
                    # Learning rate of lagrangian multiplier
                    'lambda_lr': 0.035,
                    # Type of lagrangian optimizer
                    'lambda_optimizer': "Adam",
                },
            }
        }

        # Omnisafe parameters
        custom_cfgs = {
            'seed': self.seed,
            'train_cfgs': {
                'device': self.device,
                'torch_threads': 16,
                'vector_env_nums': 1,
                'parallel': 1,
                'total_steps': self.NUM_ITER * self.STEPS_PER_ITER,
            },
            'logger_cfgs': {
                'use_wandb': False,
                'wandb_project': 'omnisafe',
                'use_tensorboard': True,
                'save_model_freq': 5, # save model every 5 epochs
                'log_dir': self.get_log_dir(),
            },
            'model_cfgs':  {
                'weight_initialization_mode': "kaiming_uniform",
                'actor': {
                    'hidden_sizes': [64, 64], # [128, 128, 128],
                    'activation': "tanh",
                    'lr': 3e-4,
                },
                'critic': {
                    'hidden_sizes': [64, 64], # [128, 128, 128],
                    'activation': "tanh",
                    'lr': 3e-4,
                },
            },
        }

        return update_params(custom_cfgs, algo_specific_cfgs[self.learner])

    def create_experiment(self):
        env_id, teacher_id, wrapper_kwargs = self.create_environment(evaluation=False)
        custom_cfgs = self.create_learner_params()
        model, interface = self.learner.create_learner(env_id=f"{teacher_id}-{env_id}", 
                                                       custom_cfgs=custom_cfgs,
                                                       wrapper_kwargs=wrapper_kwargs)
        omnisafe_log_dir = model.agent.logger.log_dir

        if str(model.agent._env._env.teacher)=="plr":
            model.agent._env._env.learner = interface

        if str(model.agent._env._env.teacher)=="vds":
            obs_shape = model.agent._env._env._observation_space.shape
            action_dim = model.agent._env._env._action_space.shape[0]
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.array([ContextualSafetyPointMass1D.ROOM_WIDTH/2-0.5, 0., -3.5, 0.0])[None, :], 
                           contexts.shape[0], axis=0),
                 contexts], axis=-1)
            model.agent._env._env.teacher.initialize_teacher(obs_shape, action_dim, interface, state_provider)
        return model, omnisafe_log_dir 

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      np.diag(np.square([self.INIT_VAR, self.INIT_VAR])), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      dist_type=self.DIST_TYPE)
        elif self.curriculum.constrained_self_paced():
            return ConstrainedSelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                                    np.diag(np.square([self.INIT_VAR, self.INIT_VAR])), bounds, self.DELTA,
                                                    cost_ub=self.DELTA_C, max_kl=self.KL_EPS, std_lower_bound=self.STD_LOWER_BOUND.copy(),
                                                    kl_threshold=self.KL_THRESHOLD, dist_type=self.DIST_TYPE)
        else:
            raise NotImplementedError("Invalid self-paced teacher type: ", str(self.curriculum()))
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(200, 2))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)

    def get_env_name(self):
        return f"safety_point_mass_1d_{self.TARGET_TYPE}"

    def evaluate_learner(self, model_path, eval_type=""):
        num_context = None
        num_run = 1

        model = self.learner.load_for_evaluation(model_path=model_path, 
                                                 obs_space=self.eval_env._observation_space,
                                                 act_space=self.eval_env._action_space, 
                                                 custom_cfgs=self.create_learner_params(), 
                                                 device=self.device)
        eval_path = f"{os.getcwd()}/eval_contexts/{self.get_env_name()}_eval{eval_type}_contexts.npy"
        if os.path.exists(eval_path):
            eval_contexts = np.load(eval_path)
            num_context = eval_contexts.shape[0]
        else:
            raise ValueError(f"Invalid evaluation context path: {eval_path}")

        num_succ_eps_per_c = np.zeros(num_context)
        all_costs = np.zeros(num_context)
        all_returns = np.zeros(num_context)
        for i in range(num_context):
            context = eval_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_context(context)
                # print(f"Context: {context} || Eval env context: {self.eval_env.get_context()}")
                obs, info = self.eval_env.reset()
                # print(f"Initial obs: {obs}")
                t = 0
                terminated = False
                truncated = False
                success = []
                costs = []
                returns = []
                while not (terminated or truncated):
                    t +=1 
                    with torch.no_grad():
                        action = model(obs.to(self.device))
                    obs, reward, cost, terminated, truncated, info = self.eval_env.step(action)
                    # print(f"Step: {t} || Action: {action} || Obs: {obs} || Reward: {reward} || Cost: {cost} || Terminated: {terminated} || Truncated: {truncated} || Info: {info}")
                    success.append(info["success"]*1)
                    costs.append(cost)
                    returns.append(reward)
                if any(success):
                    num_succ_eps_per_c[i] += 1. / num_run
                all_costs[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(costs) / num_run
                all_returns[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(returns) / num_run
            # input("END OF EPISODE")
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")
        print(f"Average Cost: {np.mean(all_costs)}")
        # input()
        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c, all_costs
    
    def evaluate_training(self, model_path, training_contexts):
        num_run = 1 

        model = self.learner.load_for_evaluation(model_path=model_path, 
                                                            obs_space=self.eval_env._observation_space,
                                                            act_space=self.eval_env._action_space, 
                                                            custom_cfgs=self.create_learner_params(), 
                                                            device=self.device)
        num_contexts = training_contexts.shape[0]
        num_succ_eps_per_c = np.zeros(num_contexts)
        all_costs = np.zeros(num_contexts)
        all_returns = np.zeros(num_contexts)
        for i in range(num_contexts):
            context = training_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs, info = self.eval_env.reset()
                t = 0
                terminated = False
                truncated = False
                success = []
                costs = []
                returns = []
                while not (terminated or truncated):
                    t +=1 
                    with torch.no_grad():
                        action = model(obs.to(self.device))
                    obs, reward, cost, terminated, truncated, info = self.eval_env.step(action)
                    success.append(info["success"]*1)
                    costs.append(cost)
                    returns.append(reward)
                if any(success):
                    num_succ_eps_per_c[i] += 1. / num_run
                all_costs[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(costs) / num_run
                all_returns[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(returns) / num_run
            # input("END OF EPISODE")
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")
        print(f"Average Cost: {np.mean(all_costs)}")
        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_contexts):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, num_succ_eps_per_c, all_costs