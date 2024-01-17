import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, NoReturn
from deep_sprl.teachers.util import RewardEstimatorGP
from deep_sprl.teachers.spl.assignment_solver import AssignmentSolver
from deep_sprl.teachers.abstract_teacher import AbstractTeacher
from deep_sprl.teachers.spl.constrained_wasserstein_interpolation import ConstrainedWassersteinInterpolation
from deep_sprl.util.utils import verbose_print, verbose_input

IS_VERBOSE = False

class ConstrainedCurrOT(AbstractTeacher):

    def __init__(self, context_bounds, init_samples, target_sampler, perf_lb, cost_ub, epsilon, episodes_per_update,
                 callback=None, model_r=None, model_c=None, wait_until_threshold=False, wb_max_reuse=1, 
                 annealing_target_probability=0.75, cost_annealing_steps=10, reward_annealing_steps=10):
        if model_r is None:
            self.model_r = RewardEstimatorGP()
        else:
            self.model_r = model_r

        if model_c is None:
            self.model_c = RewardEstimatorGP()
        else:
            self.model_c = model_c

        # Create an array if we use the same number of bins per dimension
        self.context_bounds = context_bounds
        self.threshold_reached = not wait_until_threshold
        self.teacher = ConstrainedWassersteinInterpolation(init_samples, target_sampler, perf_lb, cost_ub, epsilon, callback=callback)
        self.success_buffer = WassersteinConstrainedSuccessBuffer(perf_lb, cost_ub, init_samples.shape[0], episodes_per_update, epsilon,
                                                       context_bounds=context_bounds, max_reuse=wb_max_reuse, annealing_target_probability=annealing_target_probability,
                                                       cost_annealing_steps=cost_annealing_steps, reward_annealing_steps=reward_annealing_steps,)
        self.fail_context_buffer = []
        self.fail_return_buffer = []
        self.fail_cost_buffer = []
        self.sampler = UniformConstrainedSampler(self.context_bounds)

    def __str__(self) -> str:
        return "constrained_wasserstein"

    def on_rollout_end(self, context, ret, cost):
        self.sampler.update(context, ret, cost)

    def update_distribution(self, contexts, returns, costs):
        fail_contexts, fail_returns, fail_costs = self.success_buffer.update(contexts, returns, costs,
                                                                             self.teacher.target_sampler(
                                                                                 self.teacher.current_samples.shape[0]))
    
        verbose_print(IS_VERBOSE, f"fail_contexts: {fail_contexts}")
        verbose_print(IS_VERBOSE, f"fail_returns: {fail_returns}")
        verbose_print(IS_VERBOSE, f"fail_costs: {fail_costs}")

        if self.threshold_reached:
            verbose_print(IS_VERBOSE, "Extending fail buffer as threshold reached.")
            self.fail_context_buffer.extend(fail_contexts)
            self.fail_context_buffer = self.fail_context_buffer[-self.teacher.n_samples:]
            self.fail_return_buffer.extend(fail_returns)
            self.fail_return_buffer = self.fail_return_buffer[-self.teacher.n_samples:]
            self.fail_cost_buffer.extend(fail_costs)
            self.fail_cost_buffer = self.fail_cost_buffer[-self.teacher.n_samples:]

        success_contexts, success_returns, success_costs = self.success_buffer.read_train()
        if len(self.fail_context_buffer) == 0:
            verbose_print(IS_VERBOSE, "No failed samples yet, only using successful samples.")
            train_contexts = success_contexts
            train_returns = success_returns
            train_costs = success_costs
        else:
            verbose_print(IS_VERBOSE, "Using failed and successful samples.")
            train_contexts = np.concatenate((np.stack(self.fail_context_buffer, axis=0), success_contexts), axis=0)
            train_returns = np.concatenate((np.stack(self.fail_return_buffer, axis=0), success_returns), axis=0)
            train_costs = np.concatenate((np.stack(self.fail_cost_buffer, axis=0), success_costs), axis=0)
            
        print("Updating Reward GP Model")
        self.model_r.update_model(train_contexts, train_returns)
        print("Updating Cost GP Model")
        self.model_c.update_model(train_contexts, train_costs)

        if self.threshold_reached or (np.logical_and(self.model_r(self.teacher.current_samples) >= self.teacher.perf_lb, self.model_c(self.teacher.current_samples) <= self.teacher.cost_ub)):
            verbose_print(IS_VERBOSE, "Updating sampling distribution, as threshold reached (%r) OR performance threshold (%.3e vs %.3e) and cost threshold (%.3e vs %.3e) met!" % (
                self.threshold_reached, self.model_r(self.teacher.current_samples), self.teacher.perf_lb, self.model_c(self.teacher.current_samples), self.teacher.cost_ub))
            self.threshold_reached = True
            self.teacher.update_distribution(self.model_r, self.model_c, self.success_buffer.read_update())
        else:
            verbose_print(IS_VERBOSE, "Not updating sampling distribution, as threshold not reached (%r) AND performance threshold (%.3e vs %.3e) or cost threshold (%.3e vs %.3e) not met!" % (
                self.threshold_reached, self.model_r(self.teacher.current_samples), self.teacher.perf_lb, self.model_c(self.teacher.current_samples), self.teacher.cost_ub))
        verbose_input(IS_VERBOSE, "End of update_distribution()")

    def sample(self):
        sample = self.sampler(self.teacher.current_samples)
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1])

    def save(self, path):
        self.model_r.save(os.path.join(path, "teacher_model_r.pkl"))
        self.model_c.save(os.path.join(path, "teacher_model_c.pkl"))
        self.teacher.save(path)
        self.success_buffer.save(path)
        self.sampler.save(path)

    def load(self, path):
        self.model_r.load(os.path.join(path, "teacher_model_r.pkl"))
        self.model_c.load(os.path.join(path, "teacher_model_c.pkl"))
        self.teacher.load(path)
        self.success_buffer.load(path)
        self.sampler.load(path)


class AbstractConstrainedSuccessBuffer(ABC):

    def __init__(self, delta: float, delta_c: float, n: int, epsilon: float, context_bounds: Tuple[np.ndarray, np.ndarray],
                 annealing_target_probability: float, cost_annealing_steps: int, reward_annealing_steps: int):
        context_exts = context_bounds[1] - context_bounds[0]
        self.delta_stds = context_exts / 4
        self.min_stds = 0.005 * epsilon * np.ones(len(context_bounds[0]))
        self.context_bounds = context_bounds
        self.delta = delta
        self.delta_c = delta_c
        self.max_size = n
        self.annealing_target_probability = annealing_target_probability
        self.cost_annealing_steps = cost_annealing_steps
        self.reward_annealing_steps = reward_annealing_steps
        self.contexts = np.zeros((1, len(context_bounds[0])))
        self.returns = np.array([-np.inf])
        self.costs = np.array([np.inf])
        self.delta_reached = False
        self.delta_c_reached = False
        self.min_ret = np.inf # None
        self.max_cost = -np.inf # None
        self.num_updates_in_cost_annealing = 0
        self.num_updates_in_reward_annealing = 0

    @abstractmethod
    def update_delta_not_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, new_costs: np.ndarray,
                                 current_samples: np.ndarray) -> \
                                    Tuple[bool, np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        pass

    @abstractmethod
    def update_delta_c_not_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, new_costs: np.ndarray,
                                      current_samples: np.ndarray) -> \
                                        Tuple[bool, np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        pass

    @abstractmethod
    def update_deltas_reached(self, new_contexts: np.ndarray, new_returns: np.ndarray, new_costs: np.ndarray,
                              current_samples: np.ndarray) -> \
                                Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
        pass


    def update(self, contexts, returns, costs, current_samples):
        assert contexts.shape[0] < self.max_size

        # if self.min_ret is None:
        #     self.min_ret = np.min(returns)
        # if self.max_cost is None:
        #     self.max_cost = np.max(costs)
        self.min_ret = min(np.min(returns), self.min_ret)
        self.max_cost = max(np.max(costs), self.max_cost) + 1e-4

        if not self.delta_c_reached:
            self.delta_c_reached, self.contexts, self.returns, self.costs, mask = self.update_delta_c_not_reached(
                contexts, returns, costs, current_samples)
        elif not self.delta_reached:
            self.delta_reached, self.contexts, self.returns, self.costs, mask = self.update_delta_not_reached(
                contexts, returns, costs, current_samples)
        else:
            self.contexts, self.returns, self.costs, mask = self.update_deltas_reached(contexts, returns, costs, current_samples)

        return contexts[mask, :], returns[mask], costs[mask]

    def read_train(self):
        return self.contexts.copy(), self.returns.copy(), self.costs.copy()

    def read_update(self):
        verbose_print(IS_VERBOSE, "reading update...")
        # Compute the Gaussian search noise that we add to the samples
        if not self.delta_c_reached:
            var_scales = np.clip(self.costs - self.delta_c, 0., np.inf) / (self.max_cost - self.delta_c)
        else:
            var_scales = np.clip(self.delta - self.returns, 0., np.inf) / (self.delta - self.min_ret)
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_c_reached:
            verbose_print(IS_VERBOSE, f"delta_c not reached")
            offset = self.costs.shape[0] // 2
            sub_costs = self.costs[offset:]
            sub_contexts = self.contexts[offset:, :]
            sub_stds = stds[offset:, :]
            verbose_print(IS_VERBOSE, f"offset: {offset}")
            verbose_print(IS_VERBOSE, f"sub_costs: {sub_costs}")
            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = self.costs[offset - 1] - sub_costs
            verbose_print(IS_VERBOSE, f"probs: {probs}")
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_costs.shape[0]) / sub_costs.shape[0]
            else:
                probs = probs / norm
            verbose_print(IS_VERBOSE, f"probs: {probs}")

            sub_returns = self.returns[offset:]
            verbose_print(IS_VERBOSE, f"sub_returns: {sub_returns}")
            probs_r = sub_returns - np.min(sub_returns)
            verbose_print(IS_VERBOSE, f"probs_r: {probs_r}")
            norm_r = np.sum(probs_r)
            if norm_r == 0:
                probs_r = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs_r = probs_r / norm_r
            verbose_print(IS_VERBOSE, f"probs_r: {probs_r}")

            w = max(1.0 - (1.0-self.annealing_target_probability) * (self.num_updates_in_cost_annealing / self.cost_annealing_steps), 
                    self.annealing_target_probability)
            verbose_print(IS_VERBOSE, f"w: {w} || num_updates_in_cost_annealing: {self.num_updates_in_cost_annealing}")
            probs = (w * probs + (1 - w) * probs_r)
            verbose_print(IS_VERBOSE, f"probs: {probs}")
            self.num_updates_in_cost_annealing += 1

            sample_idxs = np.random.choice(sub_costs.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_stds = sub_stds[sample_idxs, :]
        elif not self.delta_reached:
            verbose_print(IS_VERBOSE, f"delta not reached")
            offset = self.returns.shape[0] // 2
            sub_returns = self.returns[offset:]
            sub_contexts = self.contexts[offset:, :]
            sub_stds = stds[offset:, :]
            verbose_print(IS_VERBOSE, f"offset: {offset}")
            verbose_print(IS_VERBOSE, f"sub_returns: {sub_returns}")
            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_returns - self.returns[offset - 1]
            verbose_print(IS_VERBOSE, f"probs: {probs}")
            norm = np.sum(probs)
            if norm == 0:
                probs = np.ones(sub_returns.shape[0]) / sub_returns.shape[0]
            else:
                probs = probs / norm
            verbose_print(IS_VERBOSE, f"probs: {probs}")

            sub_costs = self.costs[offset:]
            verbose_print(IS_VERBOSE, f"sub_costs: {sub_costs}")
            probs_c = np.max(sub_costs) - sub_costs
            verbose_print(IS_VERBOSE, f"probs_c: {probs_c}")
            norm_c = np.sum(probs_c)
            if norm_c == 0:
                probs_c = np.ones(sub_costs.shape[0]) / sub_costs.shape[0]
            else:
                probs_c = probs_c / norm_c
            verbose_print(IS_VERBOSE, f"probs_c: {probs_c}")

            w = min(self.annealing_target_probability + (1.0 - self.annealing_target_probability) * (self.num_updates_in_reward_annealing / self.reward_annealing_steps),
                    1.0)
            verbose_print(IS_VERBOSE, f"w: {w} || num_updates_in_reward_annealing: {self.num_updates_in_reward_annealing}")
            probs = (w * probs + (1 - w) * probs_c)
            verbose_print(IS_VERBOSE, f"probs: {probs}")
            self.num_updates_in_reward_annealing += 1

            sample_idxs = np.random.choice(sub_returns.shape[0], self.max_size, p=probs)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_stds = sub_stds[sample_idxs, :]
        else:
            to_fill = self.max_size - self.returns.shape[0]
            add_idxs = np.random.choice(self.returns.shape[0], to_fill)
            sampled_contexts = np.concatenate((self.contexts, self.contexts[add_idxs, :]), axis=0)
            sampled_stds = np.concatenate((stds, stds[add_idxs, :]), axis=0)

        contexts = sampled_contexts + np.random.normal(0, sampled_stds, size=(self.max_size, self.contexts.shape[1]))
        invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                       contexts > self.context_bounds[1][None, :]), axis=-1)
        count = 0
        while np.any(invalid) and count < 10:
            new_noise = np.random.normal(0, sampled_stds[invalid, :], size=(np.sum(invalid), self.contexts.shape[1]))
            contexts[invalid, :] = sampled_contexts[invalid, :] + new_noise
            invalid = np.any(np.logical_or(contexts < self.context_bounds[0][None, :],
                                           contexts > self.context_bounds[1][None, :]), axis=-1)
            count += 1

        return np.clip(contexts, self.context_bounds[0], self.context_bounds[1])

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> NoReturn:
        pass

    def save(self, path):

        with open(os.path.join(path, "teacher_constrained_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.delta, self.delta_c, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns,
                         self.costs, self.delta_reached, self.delta_c_reached, self.min_ret, self.max_cost, self.get_data()), f)

    def load(self, path):
        with open(os.path.join(path, "teacher_constrained_success_buffer.pkl"), "rb") as f:
            self.delta, self.delta_c, self.max_size, self.min_stds, self.delta_stds, self.contexts, self.returns, \
            self.costs, self.delta_reached, self.delta_c_reached, self.min_ret, self.max_cost, data = pickle.load(f)
        self.set_data(data)


class WassersteinConstrainedSuccessBuffer(AbstractConstrainedSuccessBuffer):

    def __init__(self, delta: float, delta_c: float, n: int, ep_per_update: int, epsilon: float,
                 context_bounds: Tuple[np.ndarray, np.ndarray], max_reuse=3, annealing_target_probability=0.75,
                 cost_annealing_steps=10, reward_annealing_steps=10):
        super().__init__(delta, delta_c, n, epsilon, context_bounds, 
                         annealing_target_probability, cost_annealing_steps, reward_annealing_steps)
        self.max_reuse = max_reuse
        self.ep_per_update = ep_per_update
        self.solver = AssignmentSolver(ep_per_update, n, max_reuse=self.max_reuse, verbose=False)
        self.last_assignments = None


    def update_delta_not_reached(self, contexts: np.ndarray, returns: np.ndarray, costs: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        med_idx = self.returns.shape[0] // 2
        mask = returns >= self.returns[med_idx]
        n_new = np.sum(mask)
        verbose_print(IS_VERBOSE, "DELTA NOT REACHED! Improving buffer quality with %d samples" % n_new)
        verbose_print(IS_VERBOSE, f"self.returns: {self.returns}")
        verbose_print(IS_VERBOSE, f"self.costs: {self.costs}")
        verbose_print(IS_VERBOSE, f"returns: {returns}")
        verbose_print(IS_VERBOSE, f"Mask: {mask} || n_new: {n_new} || med_idx: {med_idx}")
        
        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_costs = np.concatenate((costs[mask], self.costs[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(new_returns)

        verbose_print(IS_VERBOSE, f"new_returns: {new_returns}")
        verbose_print(IS_VERBOSE, f"sort_idxs: {sort_idxs}")

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0] == new_costs.shape[0]

        new_delta_reached = self.returns[self.returns.shape[0] // 2] > self.delta
        verbose_print(IS_VERBOSE, f"new_delta_reached: {new_delta_reached}")

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_returns.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]
        verbose_print(IS_VERBOSE, f"rem_mask: {rem_mask}")
        return new_delta_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], new_costs[sort_idxs], rem_mask

    def update_delta_c_not_reached(self, contexts: np.ndarray, returns: np.ndarray, costs: np.ndarray,
                                 current_samples: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, List[bool]]:
        # Only add samples that have a lower cost than the median cost in the buffer (we do >= here to allow
        # for binary costs to work)
        med_idx = self.costs.shape[0] // 2
        mask = costs <= self.costs[med_idx]
        n_new = np.sum(mask)
        verbose_print(IS_VERBOSE, "DELTA_C NOT REACHED! Improving buffer quality with %d samples" % n_new)
        verbose_print(IS_VERBOSE, f"self.returns: {self.returns}")
        verbose_print(IS_VERBOSE, f"self.costs: {self.costs}")
        verbose_print(IS_VERBOSE, f"costs: {costs}")
        verbose_print(IS_VERBOSE, f"Mask: {mask} || n_new: {n_new} || med_idx: {med_idx}")
        
        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_returns = np.concatenate((returns[mask], self.returns[offset_idx:]), axis=0)
        new_costs = np.concatenate((costs[mask], self.costs[offset_idx:]), axis=0)
        new_contexts = np.concatenate((contexts[mask, :], self.contexts[offset_idx:, :]), axis=0)
        sort_idxs = np.argsort(-new_costs)

        verbose_print(IS_VERBOSE, f"new_costs: {new_costs}")
        verbose_print(IS_VERBOSE, f"sort_idxs: {sort_idxs}")
        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_returns.shape[0] == new_costs.shape[0]

        new_delta_c_reached = self.costs[self.costs.shape[0] // 2] < self.delta_c
        verbose_print(IS_VERBOSE, f"new_delta_c_reached: {new_delta_c_reached}")
        # Since delta_c is reached, sort by returns to avoid negative probability calculation in read_update()
        # Note: delta_c must be reached before delta. Once either is reached, they keep being reached.
        if new_delta_c_reached:
            sort_idxs = sort_idxs[np.argsort(new_returns[sort_idxs])]
            verbose_print(IS_VERBOSE, f"sort_idxs (sorted wrt returns): {sort_idxs}")
            verbose_print(IS_VERBOSE, f"new_returns (sorted wrt returns): {new_returns[sort_idxs]}")
            verbose_print(IS_VERBOSE, f"new costs (sorted wrt returns): {new_costs[sort_idxs]}")
        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_costs.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            rem_mask[mask] = [i not in sort_idxs for i in np.arange(n_new)]
        verbose_print(IS_VERBOSE, f"rem_mask: {rem_mask}")
        return new_delta_c_reached, new_contexts[sort_idxs, :], new_returns[sort_idxs], new_costs[sort_idxs], rem_mask

    def update_deltas_reached(self, contexts: np.ndarray, returns: np.ndarray, costs: np.ndarray, current_samples: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, List[bool]]:
        # Compute the new successful samples
        mask = np.logical_and(returns >= self.delta, costs <= self.delta_c)
        n_new = np.sum(mask)

        if n_new > 0:
            remove_mask = np.logical_or(self.returns < self.delta, self.costs > self.delta_c)
            if not np.any(remove_mask) and self.max_reuse * self.returns.shape[0] >= current_samples.shape[0]:
                if n_new + self.contexts.shape[0] > self.max_size + self.ep_per_update:
                    print(f"self.contexts: {self.contexts.shape} || contexts: {contexts.shape} || "+\
                    f"contexts[mask,:]: {contexts[mask,:].shape} || current_samples: {current_samples.shape}")
                    if self.last_assignments is not None:
                        print(f"self.last_assignments[0]: {self.last_assignments[0].shape} || self.last_assignments[1]: {self.last_assignments[1].shape}")
                    print("Source sample size (self.contexts + contexts[mask]) is larger than the target sample size (max_size + ep_per_update)")
                    # pick a random index from mask and make it False
                    mask[np.random.choice(np.where(mask)[0])] = False
                    n_new -= 1
                    print(f"NEW! context[mask] shape: {contexts[mask].shape}")

                
                # At this stage we use the optimizer
                assignments = self.solver(self.contexts, contexts[mask], current_samples, self.last_assignments)
                source_idxs, target_idxs = np.where(assignments)

                # Select the contexts using the solution from the MIP solver. The unique functions sorts the data
                ret_idxs = np.unique(source_idxs)
                new_contexts = np.concatenate((self.contexts, contexts[mask, :]), axis=0)[ret_idxs, :]
                new_returns = np.concatenate((self.returns, returns[mask]), axis=0)[ret_idxs]
                new_costs = np.concatenate((self.costs, costs[mask]), axis=0)[ret_idxs]

                # We update the mask to indicate only the kept samples
                mask[mask] = [idx in (source_idxs - self.contexts.shape[0]) for idx in np.arange(n_new)]

                # We need to relabel the assignments
                up_ret_idxs = np.select([source_idxs == idx for idx in ret_idxs], np.arange(ret_idxs.shape[0]).tolist(),
                                        source_idxs)
                self.last_assignments = (up_ret_idxs, target_idxs)
                avg_dist = np.mean(np.linalg.norm(new_contexts[up_ret_idxs] - current_samples[target_idxs], axis=-1))
                print("Updated success buffer with %d samples. New Wasserstein distance: %.3e" % (n_new, avg_dist))
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < np.sum(remove_mask):
                    remove_idxs = np.argpartition(self.returns, kth=n_new)[:n_new]
                    remove_mask = np.zeros(self.returns.shape[0], dtype=bool)
                    remove_mask[remove_idxs] = True

                new_returns = np.concatenate((returns[mask], self.returns[~remove_mask]), axis=0)
                new_costs = np.concatenate((costs[mask], self.costs[~remove_mask]), axis=0)
                new_contexts = np.concatenate((contexts[mask, :], self.contexts[~remove_mask, :]), axis=0)

                if new_returns.shape[0] > self.max_size:
                    new_returns = new_returns[:self.max_size]
                    new_costs = new_costs[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_returns.shape[0] == new_costs.shape[0]
        else:
            new_contexts = self.contexts
            new_returns = self.returns
            new_costs = self.costs

        return new_contexts, new_returns, new_costs, ~mask


class AbstractConstrainedSampler(ABC):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        self.noise = 1e-3 * (context_bounds[1] - context_bounds[0])

    def update(self, context: np.ndarray, ret: float, cost: float) -> NoReturn:
        pass

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        return self.select(samples) + np.random.uniform(-self.noise, self.noise)

    @abstractmethod
    def select(self, samples: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> NoReturn:
        pass

    def load(self, path: str) -> NoReturn:
        pass


class UniformConstrainedSampler(AbstractConstrainedSampler):

    def __init__(self, context_bounds: Tuple[np.ndarray, np.ndarray]):
        super(UniformConstrainedSampler, self).__init__(context_bounds)

    def select(self, samples: np.ndarray) -> np.ndarray:
        return samples[np.random.randint(0, samples.shape[0]), :]
