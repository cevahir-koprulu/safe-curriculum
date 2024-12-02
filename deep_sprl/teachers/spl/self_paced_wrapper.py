from typing import ClassVar, List

import torch
import numpy as np

from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper

from gymnasium import spaces
from omnisafe.envs.core import make, env_register, support_envs
from omnisafe.typing import DEVICE_CPU

@env_register
class SelfPacedWrapper(BaseWrapper):
    _support_envs: ClassVar[List[str]] = [f'SelfPaced-{env_id}'
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
        self._env = make(env_id[len('SelfPaced-'):])
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
                           reward_from_info=False,
                           cost_from_info=False,
                           eval_mode=False,
                           penalty_coeff_s=0.,
                           penalty_coeff_t=0.,
                           wait_until_policy_update=False,
                           ):
        super().initialize_wrapper(log_dir, teacher, discount_factor, context_post_processing, 
                                   episodes_per_update, save_interval, step_divider, value_fn, lam,
                                   use_undiscounted_reward, reward_from_info, cost_from_info,
                                   eval_mode, penalty_coeff_s, penalty_coeff_t, wait_until_policy_update)
        self.context_buffer = Buffer(3, episodes_per_update + 1, True)
        self.num_curriculum_updates = 0

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        ret = undiscounted_reward - self.penalty_coeff_t * undiscounted_cost \
            if self.use_undiscounted_reward \
                else discounted_reward - self.penalty_coeff_t * discounted_cost
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
        if hasattr(self.teacher, "on_rollout_end"):
            self.teacher.on_rollout_end(cur_context, ret)
        if self.wait_until_policy_update:
            if len(self.context_buffer) >= self.episodes_per_update and (
                    self.algorithm_iteration % self.step_divider <= (
                        self.algorithm_iteration-self.step_length) % self.step_divider):
                __, contexts, returns = self.context_buffer.read_buffer()
                self.teacher.update_distribution(np.array(contexts), np.array(returns))
                self.num_curriculum_updates += 1
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"Curriculum update {self.num_curriculum_updates} at iteration {self.algorithm_iteration}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            if len(self.context_buffer) >= self.episodes_per_update:
                __, contexts, returns = self.context_buffer.read_buffer()
                self.teacher.update_distribution(np.array(contexts), np.array(returns))
                self.num_curriculum_updates += 1
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"Curriculum update {self.num_curriculum_updates} at iteration {self.algorithm_iteration}")
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                
    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)