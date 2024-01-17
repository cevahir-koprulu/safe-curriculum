from __future__ import annotations

import random
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

from deep_sprl.util.viewer import Viewer

@env_register
class ContextualSafetyMaze3D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyMaze3D-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 100}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    MAX_TIME_STEPS = 200

    """
    The maze has the following shape:

    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],

    """

    def __init__(self, 
                 env_id: str,
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs,
                 ):
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = device
        self._action_space = spaces.Box(low=-1., high=1., shape=(2,))
        self._observation_space = spaces.Box(low=-9., high=9., shape=(2,))
        self._state = None
        self.max_step = 0.5 # 0.3
        self._timestep = 0
        self._lava_passes = 0
        self._context = np.array([6., 6., 0.05]) # Top right with tolerance of 0.05
        self._viewer = Viewer(20, 20, background=(255, 255, 255))

        self._single_pass_cost = 0.5 # 0.1

    @staticmethod
    def sample_initial_state(n=None):
        if n is None:
            return torch.as_tensor(torch.rand(2)*2.0-7.0)
        else:
            return torch.as_tensor(torch.rand(n, 2)*2.0-7.0)

    def reset(
            self, 
            seed: int = None,
            options: dict[str, Any] = None):
        if seed is not None:
            self.set_seed(seed)
        self._timestep = 0
        self._lava_passes = 0
        self._state = self.sample_initial_state()
        return torch.clone(self._state), {}

    @staticmethod
    def _is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        if torch.any(context < -7.) or torch.any(context > 7.):
            return False
        # Check that the context is not within the inner rectangle (i.e. in [-5, 5] x [-5, 5])
        elif torch.all(torch.logical_and(-5. < context, context < 5.)):
            return False
        else:
            return True

    @staticmethod
    def _project_back(old_state, new_state):
        # Project back from the bounds
        new_state = torch.clamp(new_state, -7., 7.)

        # Project back from the inner circle
        if -5 < new_state[0] < 5 and -5 < new_state[1] < 5:
            new_state = torch.where(torch.logical_and(old_state <= -5, new_state > -5), -5, new_state)
            new_state = torch.where(torch.logical_and(old_state >= 5, new_state < 5), 5, new_state)

        return new_state

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")
        self._timestep += 1
        action = torch.clip(action.detach().cpu(), 
                            torch.as_tensor(self.action_space.low), 
                            torch.as_tensor(self.action_space.high))
        action = self.max_step * (action / max(1., torch.norm(action)))
        new_state = self._project_back(self._state, self._state + action)

        self._state = new_state
        # reward = torch.as_tensor(1.0 * (np.linalg.norm(self.context[:2] - new_state.numpy()) < self.context[2]))
        success = torch.as_tensor(np.linalg.norm(self.context[:2] - new_state.numpy()) < self.context[2])
        reward = torch.as_tensor(-1.0 * (not success))
        cost = torch.as_tensor(self._single_pass_cost * ((-7 <= new_state[0] <= -5) and (-5 <= new_state[1] <= 5)))
        self._lava_passes += 1 if cost > 0 else 0
        info = {"success": success,
                "reward": torch.as_tensor(1.0) if success else torch.as_tensor(0.), "cost": cost}
        terminated = truncated = torch.as_tensor((self._timestep >= self.MAX_TIME_STEPS) or info["success"])
        if terminated:
            info["final_observation"] = new_state
        return new_state, reward, cost, terminated, truncated, info

    def render(self):
        offset = 10
        
        # Infeasible areas
        outer_border_poly = [np.array([-9, 1]), np.array([9, 1]), np.array([9, -1]), np.array([-9, -1])]
        self._viewer.polygon(np.array([0, -8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([0, 8]) + offset, 0, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([-8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.polygon(np.array([8, 0]) + offset, 0.5 * np.pi, outer_border_poly, color=(0, 0, 0))
        self._viewer.square(np.zeros(2) + offset, 0., 10, color=(0, 0, 0))

        # Cost area
        self._viewer.polygon(np.array([-6, 0]) + offset, 0.5 * np.pi, 
                             [np.array([-5, 1]), np.array([5, 1]), np.array([5, -1]), np.array([-5, -1])], 
                             color=(122, 0, 0))

        # Goal
        self._viewer.circle(self.context[:2] + offset, 0.25, color=(255, 0, 0))
        
        # Agent
        self._viewer.circle(self._state + offset, 0.25, color=(0, 0, 0))
        self._viewer.display(0.01)

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def sample_action(self):
        return torch.as_tensor(self._action_space.sample())

    def close(self):
        self._viewer.close()

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, context):
        self._context = context