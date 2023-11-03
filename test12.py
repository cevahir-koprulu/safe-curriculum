import omnisafe
import deep_sprl.environments
import simple_env

env_id = 'ContextualSafetyPointMass2D-v0'
# env_id = 'SafetyPointGoal0-v0'
# env_id = 'Simple-v0'
custom_cfgs = {
    'train_cfgs': {
        'total_steps': 2048*2048,
        'torch_threads': 1,
        'vector_env_nums': 1,
        'parallel': 1,
    },
    'algo_cfgs': {
        'steps_per_epoch': 1024,
        'update_iters': 1,
    },
    'logger_cfgs': {
        'use_wandb': False,
    },
}

agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
agent.learn()