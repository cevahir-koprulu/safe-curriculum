from typing import ClassVar, List

import torch
import numpy as np

from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper

from gymnasium import spaces
from omnisafe.envs.core import make, env_register, support_envs
from omnisafe.typing import DEVICE_CPU

@env_register
class SelfPaced4CostWrapper(BaseWrapper):
    _support_envs: ClassVar[List[str]] = [f'SelfPaced4Cost-{env_id}'
                                          for env_id in support_envs() 
                                          if "Contextual" in env_id
                                          ]
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self,
                 env_id: str,
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs):
        super().__init__(env_id, num_envs, device, **kwargs)
        self._env = make(env_id[len('SelfPaced4Cost-'):])
        self.context_dim = self.context.shape[0]
        low_ext = np.concatenate((self._env._observation_space.low, -np.inf * np.ones(self.context_dim)))
        high_ext = np.concatenate((self._env._observation_space.high, np.inf * np.ones(self.context_dim)))
        self._observation_space = spaces.Box(low=low_ext, high=high_ext)
        self._action_space = self._env.action_space
        self._metadata = self._env.metadata

    def initialize_wrapper(self, 
                           log_dir,
                           teacher,
                           discount_factor,
                           context_post_processing=None,
                           episodes_per_update=50,
                           save_interval=5,
                           step_divider=1,
                           value_fn=None,
                           lam=None,
                           use_undiscounted_reward=False,
                           eval_mode=False,
                           penalty_coeff=0.,
                           ):
        super().initialize_wrapper(log_dir, teacher, discount_factor, context_post_processing, 
                                   episodes_per_update, save_interval, step_divider, value_fn, lam,
                                   use_undiscounted_reward, eval_mode, penalty_coeff)
        self.context_buffer = Buffer(3, episodes_per_update + 1, True)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        cost = undiscounted_cost if self.use_undiscounted_reward else discounted_cost
        self.context_buffer.update_buffer((cur_initial_state, cur_context, cost))
        if hasattr(self.teacher, "on_rollout_end"):
            self.teacher.on_rollout_end(cur_context, cost)
        
        if len(self.context_buffer) >= self.episodes_per_update and (
                self.algorithm_iteration % self.step_divider <= (
                    self.algorithm_iteration-self.step_length) % self.step_divider):
            __, contexts, costs = self.context_buffer.read_buffer()
            self.teacher.update_distribution(np.array(contexts), np.array(costs))

    def get_context_buffer(self):
        ins, cons, disc_costs = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_costs)