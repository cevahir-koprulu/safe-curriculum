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
class ContextualSafetyDoor2D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyDoor2D-v0']
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
        # self._observation_space = spaces.Box(low=np.array([-self.ROOM_WIDTH/2, -np.inf, -4., -np.inf]),
        #                                      high=np.array([self.ROOM_WIDTH/2, np.inf, 4., np.inf]))
        self._observation_space = spaces.Box(low=np.array([-self.ROOM_WIDTH/2, -5.0, -4., -5.0]),
                                             high=np.array([self.ROOM_WIDTH/2, 5.0, 4., 5.0]))
        self._state = None
        self._timestep = 0
        self._lava_passes = []
        self._context = np.array([self.ROOM_WIDTH*0.3, -self.ROOM_WIDTH*0.3])
        self._viewer = Viewer(self.ROOM_WIDTH, 8, background=(255, 255, 255))

        self._dt = 0.01
        self._friction_param = 0.
        self._r_coeff = 0.8 # 0.6 # 0.4 # 0.3 # 0.2
        self._single_lava_pass_cost = 0.2 # 0.1 # 0.25 # 0.5 # 1.0 
        self._single_border_pass_cost = 0.01 # 0.1
        self._no_border_crossing = True
        self._max_distance = torch.sqrt(torch.as_tensor(self.ROOM_WIDTH**2+8**2))
        self._goal_state = torch.as_tensor([0., 0., -3., 0.0])

    def reset(
            self, 
            seed: int = None,
            options: dict[str, Any] = None):
        self._timestep = 0
        self._lava_passes = []
        if seed is not None:
            self.set_seed(seed)
        self._state = torch.as_tensor([0., 0., 3., 0.])
        # print(f"context: {self._context}")
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
        on_border = True
        if (self._observation_space.low[0] < new_state[0] < self.observation_space.high[0]) and \
            (self._observation_space.low[2] < new_state[2] < self.observation_space.high[2]):
            on_border = False
        ######## EXAMPLE ########
        # 0 0 0 0 0 0 0 0 0 0 0 # 
        # 0 0 0 0 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 0 0 0 0 #
        # L L L 0 0 0 0 0 L L L #
        # L L L 0 0 0 0 0 L L L #
        # 0 0 0 0 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 0 0 0 0 #
        # 0 0 0 0 0 0 0 0 0 0 G #     
        ######## EXAMPLE ########
        on_lava = False
        lava_left_end = self._context[0]-self._context[1]/2
        lava_right_end = self._context[0]+self._context[1]/2
        if ((-1.0 <= new_state[2] <= 1.0) and not (lava_left_end < new_state[0] < lava_right_end)) or \
            ((-1.0 <= state[2] <= 1.0) and not (lava_left_end < state[0] < lava_right_end)):
            on_lava = True

        crash = False
        if state[2] >= -1 > new_state[2] or state[2] <= -1 < new_state[2]:
            alpha = (-1. - state[2]) / (new_state[2] - state[2])
            x_crit = alpha * new_state[0] + (1 - alpha) * state[0]

            if np.abs(x_crit - self.context[0]) > 0.5 * self.context[1]:
                new_state = torch.as_tensor([x_crit, 0., -1., 0.])
                crash = True

        return new_state, on_lava, on_border, crash

    def step(self, action):
        # input(action)
        self._timestep += 1
        new_state = self._state
        num_lava_passes = 0
        num_border_passes = 0
        for i in range(0, 10):
            new_state, on_lava, on_border, crash = self._step_internal(new_state, action)
            if on_lava:
                num_lava_passes += 1
            if on_border:
                num_border_passes += 1
            if crash:
                break
        if num_lava_passes > 0:
            self._lava_passes.append([self._timestep, num_lava_passes])

        self._state = new_state
        cost = torch.as_tensor(num_lava_passes * self._single_lava_pass_cost +
                               num_border_passes * self._single_border_pass_cost * self._no_border_crossing)
        reward = torch.as_tensor(torch.exp(-self._r_coeff * 
                                        torch.norm(self._goal_state[0::2] - 
                                                       new_state[0::2], p=2)))
        # reward = torch.as_tensor(-torch.norm(self._goal_state[0::2] - new_state[0::2], 
        #                                      p=2)/self._max_distance + 1.0)
        info = {"success": torch.as_tensor(torch.norm(self._goal_state[0::2] - 
                                                          new_state[0::2], p=2) < 0.25),
                "cost": cost}
        terminated = truncated = torch.as_tensor((self._timestep >= self.MAX_TIME_STEPS) or crash) 
        if truncated:
            info["final_observation"] = new_state
        # print(f"state: {new_state}, reward: {reward}, cost: {cost}, info: {info}")
        return new_state, reward, cost, terminated, truncated, info

    def render(self):
        lava_1_center = np.array([(self._context[0]-self._context[1]/2+self.ROOM_WIDTH/2)/2, 0.0])
        lava_1_width = self._context[0]-self._context[1]/2+self.ROOM_WIDTH/2
        lava_2_center = np.array([-self._context[0]/2-self._context[1]/4+3*self.ROOM_WIDTH/4, 0.0])
        lava_2_width = self.ROOM_WIDTH/2-self._context[0]-self._context[1]/2
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
        self._viewer.line(np.array([self.ROOM_WIDTH/2-0.1, 0.9]), np.array([self.ROOM_WIDTH/2+0.1, 1.1]),
                          color=(0, 255, 0), width=0.1)
        self._viewer.line(np.array([self.ROOM_WIDTH/2+0.1, 0.9]), np.array([self.ROOM_WIDTH/2-0.1, 1.1]),
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