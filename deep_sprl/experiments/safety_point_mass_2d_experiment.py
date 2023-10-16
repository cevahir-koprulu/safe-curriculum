import os
import gym
import torch.nn
import numpy as np
from deep_sprl.experiments.abstract_experiment import AbstractExperiment, Learner
from deep_sprl.teachers.alp_gmm import ALPGMM, ALPGMMWrapper
from deep_sprl.teachers.goal_gan import GoalGAN, GoalGANWrapper
from deep_sprl.teachers.spl import SelfPacedTeacherV2, SelfPacedWrapper, CurrOT
from deep_sprl.teachers.dummy_teachers import UniformSampler, DistributionSampler
from deep_sprl.teachers.dummy_wrapper import DummyWrapper
from deep_sprl.teachers.abstract_teacher import BaseWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from deep_sprl.teachers.acl import ACL, ACLWrapper
from deep_sprl.teachers.plr import PLR, PLRWrapper
from deep_sprl.teachers.vds import VDS, VDSWrapper
from deep_sprl.teachers.util import Subsampler
from scipy.stats import multivariate_normal

from deep_sprl.environments.safety_point_mass import ContextualSafetyPointMass

class SafetyPointMass2DExperiment(AbstractExperiment):
    TARGET_TYPE = "narrow"
    TARGET_MEAN = np.array([ContextualSafetyPointMass.ROOM_WIDTH*0.3, 
                            -ContextualSafetyPointMass.ROOM_WIDTH*0.3])
    TARGET_VARIANCES = {
        "narrow": np.square(np.diag([.1, .1])),
        "wide": np.square(np.diag([1., 1.])),
    }

    LOWER_CONTEXT_BOUNDS = np.array([-ContextualSafetyPointMass.ROOM_WIDTH/2, 
                                     -ContextualSafetyPointMass.ROOM_WIDTH*0.3])
    UPPER_CONTEXT_BOUNDS = np.array([ContextualSafetyPointMass.ROOM_WIDTH*0.3,
                                     ContextualSafetyPointMass.ROOM_WIDTH/2])
    EXT_CONTEXT_BOUNDS = np.array([5., 5.])

    def target_log_likelihood(self, cs):
        return multivariate_normal.logpdf(cs, self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE])

    def target_sampler(self, n, rng=None):
        if rng is None:
            rng = np.random

        return rng.multivariate_normal(self.TARGET_MEAN, self.TARGET_VARIANCES[self.TARGET_TYPE], size=n)

    INIT_VAR = 0.1
    INITIAL_MEAN = np.array([-ContextualSafetyPointMass.ROOM_WIDTH/2, 
                             ContextualSafetyPointMass.ROOM_WIDTH/2])
    # INITIAL_VARIANCE = np.diag(np.square([0.1, 0.1]))

    DIST_TYPE = "gaussian"  # "cauchy"

    STD_LOWER_BOUND = np.array([0.1, 0.1])
    KL_THRESHOLD = 8000.
    KL_EPS = 0.5
    DELTA = 40.0 # 0.0
    METRIC_EPS = 0.5
    EP_PER_UPDATE = 100 # 10

    NUM_ITER = 1000 # 500
    STEPS_PER_ITER = 4000
    DISCOUNT_FACTOR = 0.99
    LAM = 0.99

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
        self.eval_env, self.vec_eval_env = self.create_environment(evaluation=True)

    def create_environment(self, evaluation=False):
        env = gym.make("ContextualSafetyPointMass2D-v1")
        if evaluation or self.curriculum.default():
            teacher = DistributionSampler(self.target_sampler, self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS)
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.alp_gmm():
            teacher = ALPGMM(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(), seed=self.seed,
                             fit_rate=self.AG_FIT_RATE[self.learner], random_task_ratio=self.AG_P_RAND[self.learner],
                             max_size=self.AG_MAX_SIZE[self.learner])
            env = ALPGMMWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.goal_gan():
            samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(1000, 2))
            teacher = GoalGAN(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy(),
                              state_noise_level=self.GG_NOISE_LEVEL[self.learner], success_distance_threshold=0.01,
                              update_size=self.GG_FIT_RATE[self.learner], n_rollouts=2, goid_lb=0.25, goid_ub=0.75,
                              p_old=self.GG_P_OLD[self.learner], pretrain_samples=samples)
            env = GoalGANWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.self_paced() or self.curriculum.wasserstein():
            teacher = self.create_self_paced_teacher(with_callback=False)
            env = SelfPacedWrapper(env, teacher, self.DISCOUNT_FACTOR, episodes_per_update=self.EP_PER_UPDATE,
                                   context_visible=True)
        elif self.curriculum.acl():
            bins = 50
            teacher = ACL(bins * bins, self.ACL_ETA, eps=self.ACL_EPS, norm_hist_len=2000)
            env = ACLWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True,
                             context_post_processing=Subsampler(self.LOWER_CONTEXT_BOUNDS.copy(),
                                                                self.UPPER_CONTEXT_BOUNDS.copy(),
                                                                [bins, bins]))
        elif self.curriculum.plr():
            teacher = PLR(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.PLR_REPLAY_RATE,
                          self.PLR_BUFFER_SIZE, self.PLR_BETA, self.PLR_RHO)
            env = PLRWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.vds():
            teacher = VDS(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, self.DISCOUNT_FACTOR, self.VDS_NQ,
                          q_train_config={"replay_size": 5 * self.STEPS_PER_ITER, "lr": self.VDS_LR,
                                          "n_epochs": self.VDS_EPOCHS, "batches_per_epoch": self.VDS_BATCHES,
                                          "steps_per_update": self.STEPS_PER_ITER})
            env = VDSWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        elif self.curriculum.random():
            teacher = UniformSampler(self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
            env = BaseWrapper(env, teacher, self.DISCOUNT_FACTOR, context_visible=True)
        else:
            raise RuntimeError("Invalid learning type")

        return env, DummyVecEnv([lambda: env])

    def create_learner_params(self):
        return dict(common=dict(gamma=self.DISCOUNT_FACTOR, seed=self.seed, verbose=0, device=self.device,
                                policy_kwargs=dict(net_arch=[128, 128, 128], activation_fn=torch.nn.Tanh)),
                    ppo=dict(n_steps=self.STEPS_PER_ITER, gae_lambda=self.LAM, batch_size=128),
                    sac=dict(learning_rate=3e-4, buffer_size=10000, learning_starts=500, batch_size=64,
                             train_freq=5, target_entropy="auto"))

    def create_cem_teacher(self):
        pass

    def create_experiment(self):
        timesteps = self.NUM_ITER * self.STEPS_PER_ITER

        env, vec_env = self.create_environment(evaluation=False)
        model, interface = self.learner.create_learner(vec_env, self.create_learner_params())

        if isinstance(env, PLRWrapper):
            env.learner = interface

        if isinstance(env, VDSWrapper):
            state_provider = lambda contexts: np.concatenate(
                [np.repeat(np.array([0., 0., -3., 0.])[None, :], contexts.shape[0], axis=0),
                 contexts], axis=-1)
            env.teacher.initialize_teacher(env, interface, state_provider)

        callback_params = {"learner": interface, "env_wrapper": env, "save_interval": 5,
                           "step_divider": self.STEPS_PER_ITER}
        return model, timesteps, callback_params

    def create_self_paced_teacher(self, with_callback=False):
        bounds = (self.LOWER_CONTEXT_BOUNDS.copy(), self.UPPER_CONTEXT_BOUNDS.copy())
        if self.curriculum.self_paced() or self.curriculum.self_paced_with_cem():
            return SelfPacedTeacherV2(self.target_log_likelihood, self.target_sampler, self.INITIAL_MEAN.copy(),
                                      np.diag(np.square([self.INIT_VAR, self.INIT_VAR])), bounds, self.DELTA, max_kl=self.KL_EPS,
                                      std_lower_bound=self.STD_LOWER_BOUND.copy(), kl_threshold=self.KL_THRESHOLD,
                                      dist_type=self.DIST_TYPE)
        else:
            init_samples = np.random.uniform(self.LOWER_CONTEXT_BOUNDS, self.UPPER_CONTEXT_BOUNDS, size=(200, 2))
            return CurrOT(bounds, init_samples, self.target_sampler, self.DELTA, self.METRIC_EPS, self.EP_PER_UPDATE,
                          wb_max_reuse=1)

    def get_env_name(self):
        return f"safety_point_mass_2d_{self.TARGET_TYPE}"

    def evaluate_learner(self, path, eval_type=""):
        num_context = None
        num_run = 1

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)
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
                obs = self.vec_eval_env.reset()
                t = 0
                # print(f"Context: {context}")
                # print(f"t: {t} || obs: {obs}")
                done = False
                success = []
                costs = []
                returns = []
                while not done:
                    t +=1 
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                    # print(f"\nt: {t} || obs: {obs} || action: {action}")
                    # print(f"\t  rewards: {rewards} || infos: {infos}")
                    success.append(infos[0]["success"]*1)
                    costs.append(infos[0]["cost"])
                    returns.append(rewards[0])
                if any(success):
                    num_succ_eps_per_c[i] += 1. / num_run
                # all_costs[i] += np.sum(costs) / num_run
                all_costs[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(costs) / num_run
                all_returns[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(returns) / num_run
                # print(f"num_succ_eps_per_c[{i}]: {num_succ_eps_per_c[i]}")
                # print(f"all_costs[{i}]: {all_costs[i]}")
                # print(f"all_returns[{i}]: {all_returns[i]}")
                # input()
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")
        print(f"Average Cost: {np.mean(all_costs)}")
        # input()
        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_context):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, eval_contexts[:num_context, :], \
               np.exp(self.target_log_likelihood(eval_contexts[:num_context, :])), num_succ_eps_per_c, all_costs
    
    def evaluate_training(self, path, training_contexts):
        num_run = 1 

        model_load_path = os.path.join(path, "model.zip")
        model = self.learner.load_for_evaluation(model_load_path, self.vec_eval_env, self.device)

        num_contexts = training_contexts.shape[0]
        num_succ_eps_per_c = np.zeros(num_contexts)
        all_costs = np.zeros(num_contexts)
        all_returns = np.zeros(num_contexts)
        for i in range(num_contexts):
            context = training_contexts[i, :]
            for j in range(num_run):
                self.eval_env.set_context(context)
                obs = self.vec_eval_env.reset()
                t = 0
                done = False
                success = []
                costs = []
                returns = []
                while not done:
                    t +=1 
                    action = model.step(obs, state=None, deterministic=False)
                    obs, rewards, done, infos = self.vec_eval_env.step(action)
                    success.append(infos[0]["success"]*1)
                    costs.append(infos[0]["cost"])
                    returns.append(rewards[0])
                if any(success):
                    num_succ_eps_per_c[i] += 1. / num_run
                all_costs[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(costs) / num_run
                all_returns[i] += np.cumprod((np.ones(200)*self.DISCOUNT_FACTOR))/self.DISCOUNT_FACTOR@np.array(returns) / num_run
        print(f"Successful Eps: {100 * np.mean(num_succ_eps_per_c)}%")
        print(f"Average Cost: {np.mean(all_costs)}")
        disc_rewards = self.eval_env.get_reward_buffer()
        ave_disc_rewards = []
        for j in range(num_contexts):
            ave_disc_rewards.append(np.average(disc_rewards[j * num_run:(j + 1) * num_run]))
        return ave_disc_rewards, num_succ_eps_per_c, all_costs