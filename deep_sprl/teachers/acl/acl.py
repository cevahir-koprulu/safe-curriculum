from typing import ClassVar

import numpy as np
from deep_sprl.teachers.acl.exp3s import Exp3S
from deep_sprl.teachers.abstract_teacher import AbstractTeacher, BaseWrapper

from omnisafe.envs.core import env_register

class ACL(AbstractTeacher):

    def __init__(self, n_contexts, eta, eps=0.2, norm_hist_len=1000):
        self.bandit = Exp3S(n_contexts, eta, eps=eps, norm_hist_len=norm_hist_len)
        self.last_rewards = [None] * n_contexts

    def update(self, i, r):
        if self.last_rewards[i] is None:
            self.last_rewards[i] = r
            self.bandit.update(i, 0.)
        else:
            progress = np.abs(r - self.last_rewards[i])
            self.last_rewards[i] = r
            self.bandit.update(i, progress)

    def sample(self):
        return self.bandit.sample()

    def save(self, path):
        pass

    def load(self, path):
        pass

@env_register
class ACLWrapper(BaseWrapper):
    _support_envs: ClassVar[list[str]] = ['ACLWrapper-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
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

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        self.teacher.update(cur_context, discounted_reward)
