from typing import ClassVar, List

import torch
import numpy as np
from gymnasium import spaces
from deep_sprl.teachers.acl.exp3s import Exp3S
from deep_sprl.teachers.abstract_teacher import AbstractTeacher, BaseWrapper

from omnisafe.envs.core import make, env_register, support_envs
from omnisafe.typing import DEVICE_CPU

class ACL(AbstractTeacher):

    def __init__(self, n_contexts, eta, eps=0.2, norm_hist_len=1000):
        self.bandit = Exp3S(n_contexts, eta, eps=eps, norm_hist_len=norm_hist_len)
        self.last_rewards = [None] * n_contexts

    def __str__(self) -> str:
        return "acl"

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
    _support_envs: ClassVar[List[str]] = [f'ACL-{env_id}'
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
        self._env = make(env_id[len('ACL-'):])
        self.context_dim = self.context.shape[0]
        low_ext = np.concatenate((self._env._observation_space.low, -np.inf * np.ones(self.context_dim)))
        high_ext = np.concatenate((self._env._observation_space.high, np.inf * np.ones(self.context_dim)))
        self._observation_space = spaces.Box(low=low_ext, high=high_ext)
        self._action_space = self._env.action_space
        self._metadata = self._env.metadata

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        self.teacher.update(cur_context, discounted_reward)
