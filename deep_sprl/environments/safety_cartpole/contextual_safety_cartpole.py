
from __future__ import annotations

import random
from typing import Any, ClassVar

import numpy as np
import torch
import gymnasium
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

from deep_sprl.util.viewer import Viewer

@env_register
class ContextualSafetyCartpole2D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyCartpole2D-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 50}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    X_THRESHOLD = 2.4 # From CartPoleEnv in Gymnasium

    def __init__(self, 
                 env_id: str,
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs,
                 ):
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = device
        self._env = gymnasium.make('CartPole-v0', render_mode="rgb_array")
        self._observation_space = self._env.observation_space
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,)) # self._env.action_space

        self.context = np.array([-self.X_THRESHOLD*0.5, self.X_THRESHOLD*0.5])
        self._danger_entry_cost = 10.0

    def step(self, action):
        action = (action.detach().cpu().numpy()>=0).astype(int)[0]
        obs, reward, terminated, truncated, info = self._env.step(action)
        cost = self._danger_entry_cost * (not (self.context[0] <= obs[0] <= self.context[1]))
        if terminated or truncated:
            terminated = truncated = True
            info["final_observation"] = obs
        info["success"] = False # No success signal in cartpole
        return torch.as_tensor(obs), torch.as_tensor(reward), torch.as_tensor(cost), \
            torch.as_tensor(terminated), torch.as_tensor(truncated), {key: torch.as_tensor(value) for key, value in info.items()}
    
    def reset(self, seed: int = None, options: dict[str, Any] = None):
        if seed is not None:
            self.set_seed(seed)
        obs, info = self._env.reset()
        return torch.as_tensor(obs), {key: torch.as_tensor(value) for key, value in info.items()}

    def render(self):
        return self._env.render()

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def sample_action(self):
        return torch.as_tensor(self._action_space.sample())

    def close(self):
        self._env.close()

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, context):
        self._context = context