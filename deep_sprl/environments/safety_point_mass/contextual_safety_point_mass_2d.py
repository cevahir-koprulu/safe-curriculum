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
class ContextualSafetyPointMass2D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyPointMass2D-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 100}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    ROOM_WIDTH = 8.
    MAX_TIME_STEPS = 200

    def __init__(self, 
                 env_id: str,
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs,
                 ):
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = device
        self._action_space = spaces.Box(low=-10., high=10., shape=(2,))
        self._observation_space = spaces.Box(low=np.array([-self.ROOM_WIDTH/2, -np.inf, -4., -np.inf]),
                                             high=np.array([self.ROOM_WIDTH/2, np.inf, 4., np.inf]))
        self._state = None
        self._timestep = 0
        self._lava_passes = []
        self._context = np.array([self.ROOM_WIDTH*0.3, -self.ROOM_WIDTH*0.3])
        self._viewer = Viewer(self.ROOM_WIDTH, 8, background=(255, 255, 255))

        self._dt = 0.01
        self._friction_param = 0.
        self._single_lava_pass_cost = 0.1

        self.goal_state = torch.as_tensor([self.ROOM_WIDTH/2-0.5, 0., -3.5, 0.0])

    def reset(
            self, 
            seed: int = None,
            options: dict[str, Any] = None):
        self._timestep = 0
        self._lava_passes = []
        if seed is not None:
            self.set_seed(seed)
        self._state = torch.as_tensor([-self.ROOM_WIDTH/2+0.5, 0., 3.5, 0.])
        return self._state, {}

    def _step_internal(self, state, action):
        action = torch.clip(action.detach().cpu(), 
                            torch.as_tensor(self.action_space.low), 
                            torch.as_tensor(self.action_space.high))

        state_der = torch.zeros(4)
        state_der[0::2] = state[1::2]
        state_der[1::2] = 1.5 * action - self._friction_param * state[1::2] + torch.normal(0, 0.05, (2,))
        new_state = torch.clip(state + self._dt * state_der, 
                               torch.as_tensor(self.observation_space.low),
                               torch.as_tensor(self.observation_space.high))

        ###### EXAMPLE ########
        # R 0 0 0 0 0 0 0 0 0 # 
        # L L L 0 0 0 0 0 0 0 #
        # L L L 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 L L L #
        # 0 0 0 0 0 0 0 L L L #
        # 0 0 0 0 0 0 0 0 0 G #     
        ###### EXAMPLE ########
        on_lava = False
        # Lava 1 (left)
        if (new_state[2] <= 3.0 and new_state[2] >= 1.0) and (
            new_state[0] >= -self.ROOM_WIDTH/2 and new_state[0] <= self._context[0]):
            on_lava = True
        # Lava 2 (right)
        if (new_state[2] >= -3.0 and new_state[2] <= -1.0) and (
            new_state[0] <= self.ROOM_WIDTH/2 and new_state[0] >= self._context[1]):
            on_lava = True
        return new_state, on_lava

    def step(self, action):
        self._timestep += 1
        new_state = self._state
        num_lava_passes = 0
        for i in range(0, 10):
            new_state, on_lava = self._step_internal(new_state, action)
            if on_lava:
                num_lava_passes += 1
        if num_lava_passes > 0:
            self._lava_passes.append([self._timestep, num_lava_passes])

        self._state = new_state
        cost = torch.as_tensor(num_lava_passes * self._single_lava_pass_cost)
        r_coeff = 0.6
        reward = torch.as_tensor(torch.exp(-r_coeff * 
                                        torch.norm(self.goal_state[0::2] - 
                                                       new_state[0::2], p=2)))
        info = {"success": torch.as_tensor(torch.norm(self.goal_state[0::2] - 
                                                          new_state[0::2], p=2) < 0.25),
                "cost": cost}
        terminated = truncated = torch.as_tensor(self._timestep >= self.MAX_TIME_STEPS)
        if truncated:
            info["final_observation"] = new_state
        return new_state, reward, cost, terminated, truncated, info

    def render(self):
        lava_1_center = np.array([self._context[0]+self.ROOM_WIDTH/2, 6.0])
        lava_1_width = self._context[0]+self.ROOM_WIDTH/2
        lava_2_center = np.array([self._context[1]+self.ROOM_WIDTH/2, 2.0])
        lava_2_width = self.ROOM_WIDTH/2-self._context[1]
        # Lava 1 (left)
        self._viewer.polygon(center=lava_1_center,
                             angle=0.0,
                             points=[np.array([lava_1_width/2, 1.0]),
                                     np.array([-lava_1_width/2, 1.0]),
                                     np.array([lava_1_width/2, -1.0]),
                                     np.array([-lava_1_width/2, -1.0])],
                             color=(255, 0, 0),
                             width=0.1)
        # Lava 2 (right)
        self._viewer.polygon(center=lava_2_center,
                             angle=0.0,
                             points=[np.array([lava_2_width/2, 1.0]),
                                     np.array([-lava_2_width/2, 1.0]),
                                     np.array([lava_2_width/2, -1.0]),
                                     np.array([-lava_2_width/2, -1.0])],
                            color=(255, 0, 0),
                            width=0.1)
        # Goal
        self._viewer.line(np.array([self.ROOM_WIDTH-0.5-0.1, 0.9]), np.array([self.ROOM_WIDTH-0.5+0.1, 1.1]),
                          color=(0, 255, 0), width=0.1)
        self._viewer.line(np.array([self.ROOM_WIDTH-0.5+0.1, 0.9]), np.array([self.ROOM_WIDTH-0.5-0.1, 1.1]),
                          color=(0, 255, 0), width=0.1)
        # Point mass
        self._viewer.circle(self._state[0::2] + np.array([self.ROOM_WIDTH/2, 4.]), 0.1, color=(0, 0, 0))
        
        self._viewer.display(self._dt)

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