from __future__ import annotations

import random
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
import safety_gymnasium
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

from deep_sprl.util.viewer import Viewer

MAX_EPISODE_STEPS = 1000

def register_and_make_env(context, max_episode_steps):
    config = {'agent_name': "Ant", 'context': context}
    safety_gymnasium.__register_helper(
        env_id="SafetyPointContextualAnt1-v0",
        entry_point='deep_sprl.environments.safety_ant.contextual_ant_level1:CustomBuilder',
        spec_kwargs={'config': config, 'task_id': "SafetyPointContextualAnt1-v0" , 
                        'render_mode': 'rgb_array', 'camera_name': 'fixedfar'}, 
        max_episode_steps=max_episode_steps,
        )
    return safety_gymnasium.make("SafetyPointContextualAnt1-v0")

@env_register
class ContextualSafetyAnt3D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyAnt3D-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 250}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    MAX_TIME_STEPS = MAX_EPISODE_STEPS

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
        self._context = np.array([1.75, 2.5, 0.25])
        self._env = register_and_make_env(self._context.copy(), self.MAX_TIME_STEPS)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        
        self._state = None
        self._timestep = 0
        self._hazard_passes = 0
        self._single_pass_cost = 1.0 # 0.25

    @staticmethod
    def sample_initial_state(contexts):
        initial_states = []
        for context in contexts:
            assert context.shape[0] == 3
            _env = register_and_make_env(context, MAX_EPISODE_STEPS)
            initial_states.append(_env.reset()[0])
        return torch.as_tensor(np.stack(initial_states))

    def reset(
            self, 
            seed: int = None,
            options: dict[str, Any] = None):
        if seed is not None:
            self.set_seed(seed)
        self._timestep = 0
        self._hazard_passes = 0
        self._env = register_and_make_env(self._context.copy(), self.MAX_TIME_STEPS)
        self._state, info = self._env.reset()
        return torch.as_tensor(self._state), info

    @staticmethod
    def _is_feasible(context):
        # Check that the context is not in or beyond the outer wall
        if context[1] < -1.5 or context[1] > 1.5 or context[0] < -1.5 or context[0] > 1.5:
            print(f"Context {context} is not feasible: Out of bounds")
            return False
        # Check that the context is not within the inner rectangle (i.e. in [-1., 1.] x [-1., 1.])
        elif -1 < context[0] < 0.5 and -1 < context[1] < 0.5:
            print(f"Context {context} is not feasible: In inner rectangle")
            return False
        else:
            return True

    def step(self, action):
        if self._state is None:
            raise RuntimeError("State is None! Be sure to reset the environment before using it")
        self._timestep += 1
        action = torch.clip(action.detach().cpu(), 
                            torch.as_tensor(self.action_space.low), 
                            torch.as_tensor(self.action_space.high))
        s, r, c, ter, tru, i = self._env.step(action)
        self._state = torch.as_tensor(s)
        success = torch.as_tensor(i.get('goal_met', False))
        # reward = torch.as_tensor(-1.0 * (not success))
        reward = torch.as_tensor(r)
        cost = torch.as_tensor(c)
        # cost = torch.as_tensor(self._single_pass_cost * (c > 0))
        self._hazard_passes += 1 if cost > 0 else 0
        info = {"success": success,
                "reward": torch.as_tensor(1.0) if success else torch.as_tensor(0.), 
                "cost": torch.as_tensor(self._single_pass_cost * (c > 0))}
        terminated = truncated = torch.as_tensor((self._timestep >= self.MAX_TIME_STEPS) or info["success"] or ter)      
        if terminated:
            info["final_observation"] = torch.as_tensor(s)
            print(f"Resetting environment at timestep {self._timestep} with context {self._context} || Success: {success}")
        return torch.as_tensor(s), reward, cost, terminated, truncated, info

    def render(self):
        return self._env.render()

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