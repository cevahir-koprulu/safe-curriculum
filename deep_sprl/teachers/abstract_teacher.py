from typing import Any 
from abc import ABC, abstractmethod

import torch
import numpy as np
import gymnasium

from gymnasium import spaces
from omnisafe.envs.core import CMDP, make
from deep_sprl.teachers.util import Buffer

class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

class BaseWrapper(CMDP):

    def __init__(self, 
                 env_id: str,
                 env_id_actual: str,
                 teacher, 
                 discount_factor, 
                 context_visible=True, 
                 context_post_processing=None, 
                 episodes_per_update=50,
                 **kwargs):
        kwargs.pop('env_id_actual')
        kwargs.pop('teacher')
        kwargs.pop('discount_factor')
        kwargs.pop('context_visible')
        kwargs.pop('context_post_processing')
        kwargs.pop('episodes_per_update')
        
        super().__init__(env_id=env_id, **kwargs)
        self.env_id = env_id
        self.env_id_actual = env_id_actual
        self._env = make(env_id_actual)
        
        self.teacher = teacher
        self.discount_factor = discount_factor

        self.context_visible = context_visible
        self.context_post_processing = context_post_processing
        self.episodes_per_update = episodes_per_update

        self.stats_buffer = Buffer(5, 10000, True)
        self.context_trace_buffer = Buffer(5, 10000, True)

        if context_visible:
            context = self.teacher.sample()
            if context_post_processing is not None:
                context = context_post_processing(context)
            low_ext = np.concatenate((self._env._observation_space.low, -np.inf * np.ones_like(context)))
            high_ext = np.concatenate((self._env._observation_space.high, np.inf * np.ones_like(context)))
            self._observation_space = spaces.Box(low=low_ext, high=high_ext)
        else:
            self._observation_space = self._env._observation_space
        self.action_space = self._env.action_space
        self.metadata = self._env.metadata

        self.episodes_counter = 0
        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.undiscounted_cost = 0.
        self.discounted_cost = 0.
        self.cur_disc = 1.
        self.step_length = 0.

        self.cur_context = None
        self.processed_context = None
        self.cur_initial_state = None

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        pass

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        if self.context_visible:
            obs = torch.cat((obs, self.processed_context))
        self.update((obs, reward, cost, terminated, truncated, info))
        return obs, reward, cost, terminated, truncated, info

    def reset(
            self, 
            seed: int = None,
            options: dict[str, Any] = None):
        if self.cur_context is None:
            self.cur_context = self.teacher.sample()
        if self.context_post_processing is None:
            self.processed_context = self.cur_context.copy()
        else:
            self.processed_context = self.context_post_processing(self.cur_context.copy())
        self._env.context = self.processed_context.copy()
        obs, info = self._env.reset(seed=seed, options=options)

        if self.context_visible:
            obs = torch.cat((obs, self.processed_context))

        self.cur_initial_state = obs.copy()
        return obs, info

    def set_context(self, context):
        self.cur_context = context

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def update(self, step):
        obs, reward, cost, terminated, truncated, info = step
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.undiscounted_cost += cost
        self.discounted_cost += self.cur_disc * cost
        self.cur_disc *= self.discount_factor
        self.step_length += 1.

        if terminated or truncated:
            self.done_callback(step, self.cur_initial_state.copy(), self.cur_context, 
                               self.discounted_reward, self.undiscounted_reward,
                               self.discounted_cost, self.undiscounted_cost)
            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, 
                                             self.undiscounted_cost, self.discounted_cost, self.step_length))
            self.context_trace_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward,
                                                     self.undiscounted_cost, self.discounted_cost,
                                                     self.processed_context.copy()))
            self.episodes_counter += 1
            if self.episodes_counter >= self.episodes_per_update:
                self.episodes_counter = 0

            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.processed_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0
        else:
            rewards, disc_rewards, costs, disc_costs, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_cost = np.mean(costs)
            mean_disc_cost = np.mean(disc_costs)
            mean_step_length = np.mean(steps)
            return mean_reward, mean_disc_reward, mean_cost, mean_disc_cost, mean_step_length

    def get_encountered_contexts(self):
        return self.context_trace_buffer.read_buffer()

    def get_reward_buffer(self):
        if len(self.stats_buffer) == 0:
            return None
        else:
            return self.stats_buffer.read_buffer()[1]
            
    def get_cost_buffer(self):
        if len(self.stats_buffer) == 0:
            return None
        else:
            return self.stats_buffer.read_buffer()[3]