import omnisafe
import deep_sprl.environments
import simple_env
import numpy as np
import torch
from deep_sprl.util.utils import update_params
from omnisafe.envs.core import support_envs

# from deep_sprl.teachers import abstract_teacher
# from deep_sprl.teachers.acl import acl 
# from deep_sprl.teachers.alp_gmm import alp_gmm_wrapper
# from deep_sprl.teachers.goal_gan import goal_gan_wrapper
# from deep_sprl.teachers.plr import prioritized_level_replay
# from deep_sprl.teachers.spl import self_paced_wrapper
# from deep_sprl.teachers.vds import vds

# env_id = 'BaseTeacherWrapper-v0'
env_id = 'ContextualSafetyPointMass2D-v0'
# env_id = 'SafetyPointGoal0-v0'
# env_id = 'Simple-v0'
# custom_cfgs = {
#     'train_cfgs': {
#         'total_steps': 2048*2048,
#         'torch_threads': 1,
#         'vector_env_nums': 1,
#         'parallel': 1,
#     },
#     'algo_cfgs': {
#         'steps_per_epoch': 1024,
#         'update_iters': 1,
#     },
#     'logger_cfgs': {
#         'use_wandb': False,
#     },
# }

STEPS_PER_ITER = 4000
DISCOUNT_FACTOR = 0.99
NUM_ITER = 1000
LAM = 0.99

algo_specific_cfgs = {
    "SAC": {
        'train_cfgs': {
            'eval_episodes': 0,
        },
        'algo_cfgs': {
            'steps_per_epoch': STEPS_PER_ITER, # to eval, log, actor scheduler step
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
            'gamma': DISCOUNT_FACTOR,
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
    "PPO":  {
        'algo_cfgs': {
            'steps_per_epoch': STEPS_PER_ITER, # to eval, log, actor scheduler step
            'update_iters': 10, # gradient steps
            'batch_size': 128,
            'target_kl': 0.02,
            'entropy_coef': 0.0,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': True,
            'kl_early_stop': True,
            'use_max_grad_norm': True,
            'max_grad_norm': 0.5,
            'use_critic_norm': True,
            'critic_norm_coef': 0.5, # 0.001,
            'gamma': DISCOUNT_FACTOR,
            'cost_gamma': DISCOUNT_FACTOR,
            'lam': LAM,
            'lam_c': LAM,
            'clip': 0.2,
            'adv_estimation_method': 'gae',
            'standardized_rew_adv': True,
            'standardized_cost_adv': True,
            'penalty_coef': 0.0,
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
    "PPOLag":  {
        'algo_cfgs': {
            'steps_per_epoch': STEPS_PER_ITER, # to eval, log, actor scheduler step
            'update_iters': 10, # gradient steps
            'batch_size': 128,
            'target_kl': 0.02,
            'entropy_coef': 0.0,
            'reward_normalize': False,
            'cost_normalize': False,
            'obs_normalize': True,
            'kl_early_stop': True,
            'use_max_grad_norm': True,
            'max_grad_norm': 0.5,
            'use_critic_norm': True,
            'critic_norm_coef': 0.5, # 0.001,
            'gamma': DISCOUNT_FACTOR,
            'cost_gamma': DISCOUNT_FACTOR,
            'lam': LAM,
            'lam_c': LAM,
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
    }
}

# Omnisafe parameters
custom_cfgs = {
    'seed': 0,
    'train_cfgs': {
        'device': "cuda:0",
        'torch_threads': 4,
        'vector_env_nums': 1,
        'parallel': 1,
        'total_steps': NUM_ITER * STEPS_PER_ITER,
    },
    'logger_cfgs': {
        'use_wandb': False,
        'wandb_project': 'omnisafe',
        'use_tensorboard': True,
        'save_model_freq': 5, # save model every 5 epochs
        'log_dir': "./test_runs",
    },
    'model_cfgs':  {
        'weight_initialization_mode': "kaiming_uniform",
        'linear_lr_decay': True,
        'actor': {
            'hidden_sizes': [128, 128, 128],
            'activation': "tanh",
            'lr': 3e-4,
        },
        'critic': {
            'hidden_sizes': [128, 128, 128],
            'activation': "tanh",
            'lr': 3e-4,
        },
    },
}


algo = "PPOLag"
custom_cfgs = update_params(custom_cfgs, algo_specific_cfgs[algo])
print(custom_cfgs)
custom_cfgs['logger_cfgs']['log_dir'] = f"./test_runs/penalty_coef={custom_cfgs['algo_cfgs']['penalty_coef']}"
agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
print(agent.agent._env._env.context)
agent.agent._env._env.context = torch.as_tensor([0.0, 0.0])
print(agent.agent._env._env.context)
# agent.learn()

print(support_envs())

from deep_sprl.teachers.dummy_teachers import DistributionSampler
from deep_sprl.teachers.dummy_wrapper import DummyWrapper
teacher = DistributionSampler(lambda n, rng=None: np.zeros((n,2)), np.zeros(2), np.ones(2))
agent = omnisafe.Agent(algo, f'DummyTeacher-{env_id}', custom_cfgs=custom_cfgs)
print(agent.agent._env._env.context)
agent.agent._env._env.context = torch.as_tensor([0.0, 0.0])
print(agent.agent._env._env.context)

print(isinstance(agent.agent, DummyWrapper))

print(agent.agent._actor_critic)

print(agent.agent.logger.log_dir)

model_params = torch.load('/home/ck28372/safe-curriculum/test_runs/penalty_coef=0.0/PPOLag-{DummyTeacher-ContextualSafetyPointMass2D-v0}/seed-000-2023-11-07-16-37-59/torch_save/epoch-0.pt',
                          map_location='cpu')
# print(model_params.keys())
# print(model_params['pi'])
# agent.agent._actor_critic.actor.load_state_dict(model_params['pi'])

from omnisafe.models.actor import ActorBuilder
actor_builder = ActorBuilder(
    obs_space=agent.agent._env.observation_space,
    act_space=agent.agent._env.action_space,
    hidden_sizes=custom_cfgs['model_cfgs']['actor']['hidden_sizes'],
    activation=custom_cfgs['model_cfgs']['actor']['activation'],
    weight_initialization_mode=custom_cfgs['model_cfgs']['weight_initialization_mode'],
)
actor = actor_builder.build_actor(custom_cfgs['model_cfgs']['actor_type'])
actor.load_state_dict(model_params['pi'])

from omnisafe.envs.core import make
env = make(env_id)
print(env._observation_space)