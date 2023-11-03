from typing import Any

import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper

from omnisafe.envs.core import env_register

@env_register
class DummyWrapper(BaseWrapper):
    _support_envs: ClassVar[list[str]] = ['DummyWrapper-v0']
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs = 1

    def __init__(self,
                 env_id: str,
                 env_id_actual: str,
                 teacher, 
                 discount_factor, 
                 context_visible=True, 
                 reward_from_info=False,
                 context_post_processing=None, 
                 episodes_per_update=50,
                 **kwargs):
        super().__init__(self, env_id, env_id_actual, teacher, discount_factor, context_visible,
                             reward_from_info, context_post_processing, episodes_per_update, **kwargs)
        self.use_undiscounted_reward = use_undiscounted_reward
        self.context_buffer = Buffer(3, episodes_per_update + 1, True)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        ret = undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
        if len(self.context_buffer) >= self.episodes_per_update:
            __, contexts, returns = self.context_buffer.read_buffer()

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)