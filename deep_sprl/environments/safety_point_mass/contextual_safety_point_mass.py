import numpy as np
import time

from gym import Env, spaces
from deep_sprl.util.viewer import Viewer


class ContextualSafetyPointMass(Env):
    ROOM_WIDTH = 8.

    def __init__(self, context=np.array([0., 2., 2.]), cost_coeff=0.1):
        self.action_space = spaces.Box(np.array([-10., -10.]), np.array([10., 10.]))
        self.observation_space = spaces.Box(np.array([-self.ROOM_WIDTH/2, -np.inf, -4., -np.inf]),
                                            np.array([self.ROOM_WIDTH/2, np.inf, 4., np.inf]))
        self._state = None
        self._goal_state = np.array([self.ROOM_WIDTH/2-0.5, 0., -3.5, 0.0])
        self.context = context
        self._dt = 0.01
        self.cost_coeff = cost_coeff
        self.single_lava_pass_cost = 0.1
        self.timestep = 0
        self.lava_passes = []
        self._viewer = Viewer(self.ROOM_WIDTH, 8, background=(255, 255, 255))

    def reset(self):
        # print(f"RESET AT {self.timestep} || NUMBER OF LAVA PASSES: {self.lava_passes}")
        # print("********** RESET **********")
        self.timestep = 0
        self.lava_passes = []
        self._state = np.array([-self.ROOM_WIDTH/2+0.5, 0., 3.5, 0.])
        return np.copy(self._state)

    def _step_internal(self, state, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        state_der = np.zeros(4)
        state_der[0::2] = state[1::2]
        friction_param = self.context[2]
        state_der[1::2] = 1.5 * action - friction_param * state[1::2] + np.random.normal(0, 0.05, (2,))
        new_state = np.clip(state + self._dt * state_der, self.observation_space.low,
                            self.observation_space.high)

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
            new_state[0] >= -self.ROOM_WIDTH/2 and new_state[0] <= self.context[0]):
            on_lava = True
        # Lava 2 (right)
        if (new_state[2] >= -3.0 and new_state[2] <= -1.0) and (
            new_state[0] <= self.ROOM_WIDTH/2 and new_state[0] >= self.context[1]):
            on_lava = True
        return new_state, on_lava

    def step(self, action):
        self.timestep += 1
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")

        action = np.clip(action, self.action_space.low, self.action_space.high)

        new_state = self._state
        num_lava_passes = 0
        for i in range(0, 10):
            new_state, on_lava = self._step_internal(new_state, action)
            if on_lava:
                num_lava_passes += 1
        if num_lava_passes > 0:
            self.lava_passes.append([self.timestep, num_lava_passes])

        self._state = np.copy(new_state)
        cost = num_lava_passes * self.single_lava_pass_cost
        info = {"success": np.linalg.norm(self._goal_state[0::2] - new_state[0::2]) < 0.25,
                "cost": cost}
        r_coeff = 0.6
        reward = np.exp(-r_coeff * np.linalg.norm(self._goal_state[0::2] - new_state[0::2]))
        augmented_reward = reward - self.cost_coeff * cost
        # if info["success"]:
        #     print(f"timestep: {self.timestep} || state: {new_state} || goal: {self._goal_state} || context: {self.context}")
        return new_state, augmented_reward, False, info

    def render(self, mode='human'):
        lava_1_center = np.array([self.context[0]+self.ROOM_WIDTH/2, 6.0])
        lava_1_width = self.context[0]+self.ROOM_WIDTH/2
        lava_2_center = np.array([self.context[1]+self.ROOM_WIDTH/2, 2.0])
        lava_2_width = self.ROOM_WIDTH/2-self.context[1]
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
