from typing import ClassVar, List, Dict, Any

import torch
import numpy as np
from deep_sprl.teachers.abstract_teacher import AbstractTeacher, BaseWrapper

from gymnasium import spaces
from omnisafe.envs.core import make, env_register, support_envs
from omnisafe.typing import DEVICE_CPU

class PLR(AbstractTeacher):

    def __init__(self, context_lb, context_ub, replay_rate, buffer_size, beta, rho, is_discrete=False):
        self.context_lb = context_lb
        self.context_ub = context_ub
        self.is_discrete = is_discrete
        self.replay_rate = replay_rate
        self.contexts = []
        self.scores = np.zeros(buffer_size)
        self.max_returns = np.zeros(buffer_size)
        self.stalenesses = np.zeros(buffer_size)
        self.max_buffer_size = buffer_size
        self.beta = beta
        self.rho = rho

        self.sample_from_buffer = None

    def __str__(self) -> str:
        return "plr"

    def sample_uniform(self):
        if self.is_discrete:
            to_sample = set(np.arange(self.context_lb, self.context_ub)) - set(self.contexts)
            if len(to_sample) == 0:
                return self.sample_prioritized()
            else:
                return np.random.choice(list(to_sample))
        else:
            return np.random.uniform(self.context_lb, self.context_ub)

    def logsumexp(self, log_x):
        log_x_max = np.max(log_x)
        return np.log(np.sum(np.exp(log_x - log_x_max))) + log_x_max

    def sample_prioritized(self):
        self.sample_from_buffer = True
        cur_size = len(self.contexts)
        if cur_size == 1:
            return 0
        else:
            # Compute the ranking
            tmp = np.argsort(self.scores[:cur_size])[::-1]
            log_score_probs = np.zeros(cur_size)
            log_score_probs[tmp] = -np.log(np.arange(cur_size) + 1) / self.beta
            score_probs = np.exp(log_score_probs - self.logsumexp(log_score_probs))

            # Compute the staleness probability
            staleness_probs = self.stalenesses[:cur_size] / np.sum(self.stalenesses[:cur_size])

            sample_probs = (1 - self.rho) * score_probs + self.rho * staleness_probs
            return np.random.choice(len(self.contexts), p=sample_probs)

    def update(self, task, r, value_trace):
        if self.sample_from_buffer is None:
            raise RuntimeError("Update called without previously sampling")

        # If the task is not from the buffer, check if we should add it
        if self.sample_from_buffer:
            self.max_returns[task] = max(self.max_returns[task], r)
            score = np.mean(np.clip(self.max_returns[task] - value_trace, 0, np.inf))
        else:
            score = np.mean(np.clip(r - value_trace, 0, np.inf))
            if len(self.contexts) < self.max_buffer_size:
                self.contexts.append(task)
                task = len(self.contexts) - 1
                self.max_returns[task] = r
            else:
                min_score_idx = np.argmin(self.scores)
                if score > self.scores[min_score_idx]:
                    self.contexts[min_score_idx] = task
                    self.max_returns[min_score_idx] = r
                    task = min_score_idx
                else:
                    # Nothing to do
                    task = -1

        self.sample_from_buffer = None

        # Update the score
        self.stalenesses[0:len(self.contexts)] += 1
        if task >= 0:
            self.scores[task] = score
            self.stalenesses[task] = 0

    def sample(self):
        self.sample_from_buffer = False
        if np.random.uniform(0, 1) > self.replay_rate or len(self.contexts) == 0:
            # This may override the sample from buffer flag by calling sample_prioritzed
            return self.sample_uniform()
        else:
            return self.sample_prioritized()

    def post_process(self, task):
        if self.sample_from_buffer:
            return self.contexts[task]
        else:
            return task

    def save(self, path):
        pass

    def load(self, path):
        pass


@env_register
class PLRWrapper(BaseWrapper):
    _support_envs: ClassVar[List[str]] = [f'PLR-{env_id}'
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
        self._env = make(env_id[len('PLR-'):])
        self.context_dim = self.context.shape[0]
        low_ext = np.concatenate((self._env._observation_space.low, -np.inf * np.ones(self.context_dim)))
        high_ext = np.concatenate((self._env._observation_space.high, np.inf * np.ones(self.context_dim)))
        self._observation_space = spaces.Box(low=low_ext, high=high_ext)
        self._action_space = self._env.action_space
        self._metadata = self._env.metadata

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
                           eval_mode=False,
                           penalty_coeff=0.,
                           ):
        super().initialize_wrapper(log_dir, teacher, discount_factor, context_post_processing, 
                                   episodes_per_update, save_interval, step_divider, value_fn, lam,
                                   use_undiscounted_reward, eval_mode, penalty_coeff)
        self.state_trace = []
        self.reward_trace = []
        self.step_count = 0
        if self.value_fn is not None:
            self.train_state_buffer = []
            self.train_value_buffer = []

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
        obs = torch.cat((obs, self.processed_context))

        self.state_trace = [obs.copy()]
        self.reward_trace = []
        self.cur_initial_state = obs.copy()
        return obs, info

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self._env.step(action)
        if "final_observation" in info:
            info["final_observation"] = torch.cat((info["final_observation"], torch.as_tensor(self.processed_context))).float()
        self.step_count += 1
        obs = torch.cat((obs, self.processed_context))
        self.state_trace.append(obs.copy())
        self.reward_trace.append(reward)

        # In this case PLR trains its own value function (if e.g. using a different algorithm than PPO)
        if (terminated or truncated) and self.value_fn is not None:
            values = self.value_fn(np.array(self.state_trace))
            advantages = np.zeros((values.shape[0] - 1, 1))
            last_gae_lam = 0
            for cur_step in reversed(range(values.shape[0] - 1)):
                delta = self.reward_trace[cur_step] + self.discount_factor * values[cur_step + 1] - values[cur_step]
                advantages[cur_step] = last_gae_lam = delta + self.discount_factor * self.lam * last_gae_lam
            self.train_state_buffer.append(np.array(self.state_trace[:-1]))
            self.train_value_buffer.append(advantages + values[:-1])

            if self.value_fn.should_train(self.step_count):
                self.value_fn.train(np.concatenate(self.train_state_buffer, axis=0),
                                    np.concatenate(self.train_value_buffer, axis=0))
                self.train_state_buffer.clear()
                self.train_value_buffer.clear()


        self.update((obs, reward, cost, terminated, truncated, info))
        self.step_callback()
        return (obs, reward, cost, terminated, truncated, info)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        # We currently rely on the learner being set on the environment after its creation
        if self.value_fn is None:
            estimated_values = self.learner.estimate_value_r(np.array(self.state_trace))
        else:
            estimated_values = self.value_fn(np.array(self.state_trace))
        self.teacher.update(cur_context, discounted_reward, estimated_values)

class ValueFunction:

    def __init__(self, input_dim, layers, act_func, train_config):
        self.train_config = train_config
        self.model = ValueModel(input_dim, layers, act_func)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_config["lr"], eps=1e-5)
        self.next_training = train_config["steps_per_iter"]

    def train(self, states, values):
        states = torch.from_numpy(states).type(torch.float32)
        values = torch.from_numpy(values).type(torch.float32)
        inds = np.arange(states.shape[0])
        for _ in range(self.train_config["noptepochs"]):
            np.random.shuffle(inds)
            batch_size = states.shape[0] // self.train_config["minibatches"]
            for epoch in range(self.train_config["minibatches"]):
                batch_inds = inds[batch_size * epoch:batch_size * (1 + epoch)]
                predictions = self.model(states[batch_inds, :])
                loss = torch.sum(torch.nn.functional.mse_loss(predictions, values[batch_inds, :]))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.next_training += self.train_config["steps_per_iter"]

    def should_train(self, count):
        return count > self.next_training

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            np_type = x.dtype
            pred = self.model(torch.from_numpy(x).type(torch.float32))
            return pred.detach().numpy().astype(np_type)
        else:
            return self.model(x)


class ValueModel(torch.nn.Module):

    def __init__(self, input_dim, layers, act_func):
        super().__init__()
        layers_ext = [input_dim] + layers + [1]
        torch_layers = []
        for i in range(len(layers_ext) - 1):
            torch_layers.append(torch.nn.Linear(layers_ext[i], layers_ext[i + 1], bias=True))
        self.layers = torch.nn.ModuleList(torch_layers)
        self.act_fun = act_func

    def __call__(self, x):
        h = x
        for l in self.layers[:-1]:
            h = self.act_fun(l(h))

        return self.layers[-1](h)
