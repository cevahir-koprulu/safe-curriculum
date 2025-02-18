from typing import Any, ClassVar
from abc import ABC, abstractmethod

import os
import time
import torch
import pickle
import numpy as np

from typing import Dict

from deep_sprl.teachers.util import Buffer
from omnisafe.envs.core import CMDP
from omnisafe.typing import DEVICE_CPU

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
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs,
                 ):
        super().__init__(env_id)
        self._env_id = env_id
        self._num_envs = num_envs
        self._device = torch.device(device)

    def initialize_wrapper(self, 
                           log_dir,
                           teacher,
                           discount_factor,
                           context_post_processing=None,
                           episodes_per_update=50,
                           save_interval=5,
                           step_divider=1,
                           value_fn=None,
                           lam=None,
                           use_undiscounted_reward=False,
                           reward_from_info=False,
                           cost_from_info=False,
                           eval_mode=False,
                           penalty_coeff_s=0.,
                           penalty_coeff_t=0.,
                           wait_until_policy_update=False,
                           ):
        self.log_dir = log_dir
        self.teacher = teacher
        self.discount_factor = discount_factor

        self.context_post_processing = context_post_processing
        self.episodes_per_update = episodes_per_update
        self.save_interval = save_interval
        self.step_divider = step_divider
        self.value_fn = value_fn
        self.lam = lam
        self.use_undiscounted_reward = use_undiscounted_reward
        self.reward_from_info = reward_from_info
        self.cost_from_info = cost_from_info
        self.eval_mode = eval_mode
        self.penalty_coeff_s = penalty_coeff_s
        self.penalty_coeff_t = penalty_coeff_t
        self.wait_until_policy_update = wait_until_policy_update

        self.stats_buffer = Buffer(6, 10000, True)
        self.context_trace_buffer = Buffer(7, 10000, True)

        self.algorithm_iteration = 0
        self.iteration = 0
        self.episodes_counter = 0
        self.undiscounted_reward = 0.
        self.discounted_reward = 0.
        self.undiscounted_cost = 0.
        self.discounted_cost = 0.
        self.undiscounted_reward_for_teacher = 0.
        self.discounted_reward_for_teacher = 0.
        self.undiscounted_cost_for_teacher = 0.
        self.discounted_cost_for_teacher = 0.
        self.successful = False
        self.cur_disc = 1.
        self.step_length = 0.

        self.cur_context = None
        self.processed_context = None
        self.cur_initial_state = None

    def init_step_callback(self):
        self.last_time = None
        self.format = "   %4d    | %.1E |   %3d    |  %.2E  |  %.2E  |  %.2E  |  %.2E  |  %.2E  "
        if self.penalty_coeff_s != 0.:
            self.format += "|  %.2E  |  %.2E  "
        if self.penalty_coeff_t != 0.: 
            self.format += "|  %.2E  |  %.2E  "
        if self.teacher is not None:
            if str(self.teacher)=="self_paced" or str(self.teacher)=="constrained_self_paced":
                context_dim = self.teacher.context_dim
                text = "| [%.2E"
                for i in range(0, context_dim - 1):
                    text += ", %.2E"
                text += "] "
                self.format += text + text
        header = " Iteration |  Time   | Ep. Len. | Mean Reward | Mean Disc. Reward "+\
            "| Mean Cost | Mean Disc. Cost | Mean Success "
        if self.penalty_coeff_s != 0.:
            header += "| Mean PenRewS | Mean Disc. PenRewS "
        if self.penalty_coeff_t != 0.:
            header += "| Mean PenRewT | Mean Disc. PenRewT "
        if self.teacher is not None:
            if str(self.teacher)=="self_paced":
                header += "|     Context mean     |      Context std     "
        print(header)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        pass

    def step_callback(self):
        if self.eval_mode:
            return
        
        if self.algorithm_iteration == 0:
            self.init_step_callback()

        if self.algorithm_iteration % self.step_divider == 0:
            data_tpl = (self.iteration,)

            t_new = time.time()
            dt = np.nan
            if self.last_time is not None:
                dt = t_new - self.last_time
            data_tpl += (dt,)

            mean_rew, mean_disc_rew, mean_cost, mean_disc_cost, mean_success, mean_length = self.get_statistics()
            data_tpl += (int(mean_length), mean_rew, mean_disc_rew, mean_cost, mean_disc_cost, mean_success,)
            if self.penalty_coeff_s != 0.:
                data_tpl += (mean_rew - self.penalty_coeff_s * mean_cost,
                             mean_disc_rew - self.penalty_coeff_s * mean_disc_cost,)
            if self.penalty_coeff_t != 0.:
                data_tpl += (mean_rew - self.penalty_coeff_t * mean_cost,
                             mean_disc_rew - self.penalty_coeff_t * mean_disc_cost,)

            if str(self.teacher)=="self_paced" or str(self.teacher)=="constrained_self_paced":
                context_mean = self.teacher.context_dist.mean()
                context_std = np.sqrt(np.diag(self.teacher.context_dist.covariance_matrix()))
                data_tpl += tuple(context_mean.tolist())
                data_tpl += tuple(context_std.tolist())

            print(self.format % data_tpl)

            if self.iteration % self.save_interval == 0:
                iter_log_dir = os.path.join(self.log_dir, "iteration-" + str(self.iteration))
                os.makedirs(iter_log_dir, exist_ok=True)

                with open(os.path.join(iter_log_dir, "context_trace.pkl"), "wb") as f:
                    pickle.dump(self.get_encountered_contexts(), f)

                self.teacher.save(iter_log_dir)
            self.last_time = time.time()
            self.iteration += 1

        self.algorithm_iteration += 1

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        if info["success"]:
            self.successful = True
        obs = torch.cat((obs, torch.as_tensor(self.processed_context))).float()
        self.update((obs, reward, cost, terminated, truncated, info))
        self.step_callback()
        return obs, reward, cost, terminated, truncated, info

    def reset(
            self, 
            seed: int = None,
            options: Dict[str, Any] = None):
        if self.cur_context is None:
            self.cur_context = self.teacher.sample()
        if self.context_post_processing is None:
            self.processed_context = self.cur_context.copy()
        else:
            self.processed_context = self.context_post_processing(self.cur_context.copy())
        self._env.context = self.processed_context.copy()
        obs, info = self._env.reset(seed=seed, options=options)
        obs = torch.cat((obs, torch.as_tensor(self.processed_context))).float()
        self.cur_initial_state = obs.detach().clone()
        return obs, info

    def set_context(self, context):
        self.cur_context = context

    def get_context(self):
        return self.cur_context

    def render(self, mode='human'):
        return self._env.render()

    def update(self, step):
        obs, reward, cost, terminated, truncated, info = step
        if self.reward_from_info:
            self.undiscounted_reward_for_teacher += info["reward"]
            self.discounted_reward_for_teacher += self.cur_disc * info["reward"]
        else:
            self.undiscounted_reward_for_teacher += reward
            self.discounted_reward_for_teacher += self.cur_disc * reward
        if self.cost_from_info:
            self.undiscounted_cost_for_teacher += info["cost"]
            self.discounted_cost_for_teacher += self.cur_disc * info["cost"]
        else:
            self.undiscounted_cost_for_teacher += cost
            self.discounted_cost_for_teacher += self.cur_disc * cost
        if "final_observation" in info:
            info["final_observation"] = torch.cat((info["final_observation"], torch.as_tensor(self.processed_context))).float()
        if "success" in info:
            if info["success"]:
                self.successful = True 
    
        self.undiscounted_reward += reward
        self.discounted_reward += self.cur_disc * reward
        self.undiscounted_cost += cost
        self.discounted_cost += self.cur_disc * cost
        self.cur_disc *= self.discount_factor
        self.step_length += 1.
        if terminated or truncated:
            self.done_callback(step, self.cur_initial_state.detach().clone(), self.cur_context, 
                               self.discounted_reward_for_teacher, self.undiscounted_reward_for_teacher,
                               self.discounted_cost_for_teacher, self.undiscounted_cost_for_teacher)
            self.stats_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward, 
                                             self.undiscounted_cost, self.discounted_cost, 
                                             self.successful, self.step_length))
            self.context_trace_buffer.update_buffer((self.undiscounted_reward, self.discounted_reward,
                                                     self.undiscounted_cost, self.discounted_cost,
                                                     self.successful, self.step_length, 
                                                     self.processed_context.copy()))
            self.episodes_counter += 1
            if self.episodes_counter >= self.episodes_per_update and (
                self.algorithm_iteration % self.step_divider <= (
                    self.algorithm_iteration-self.step_length) % self.step_divider):
                self.episodes_counter = 0

            self.undiscounted_reward = 0.
            self.discounted_reward = 0.
            self.undiscounted_cost = 0.
            self.discounted_cost = 0.
            self.undiscounted_reward_for_teacher = 0.
            self.discounted_reward_for_teacher = 0.
            self.undiscounted_cost_for_teacher = 0.
            self.discounted_cost_for_teacher = 0.
            self.successful = False
            self.cur_disc = 1.
            self.step_length = 0.

            self.cur_context = None
            self.processed_context = None
            self.cur_initial_state = None

    def get_statistics(self):
        if len(self.stats_buffer) == 0:
            return 0., 0., 0., 0., 0., 0
        else:
            rewards, disc_rewards, costs, disc_costs, success, steps = self.stats_buffer.read_buffer()
            mean_reward = np.mean(rewards)
            mean_disc_reward = np.mean(disc_rewards)
            mean_cost = np.mean(costs)
            mean_disc_cost = np.mean(disc_costs)
            mean_success = np.mean(success)
            mean_step_length = np.mean(steps)
            return mean_reward, mean_disc_reward, mean_cost, mean_disc_cost, mean_success, mean_step_length

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

    def set_seed(self, seed: int):
        self._env.set_seed(seed)

    def sample_action(self):
        return torch.as_tensor(self._env._action_space.sample())

    def close(self):
        self._env._viewer.close()

    @property
    def context(self):
        return self._env.context
    
    @context.setter
    def context(self, context):
        self._env.context = context