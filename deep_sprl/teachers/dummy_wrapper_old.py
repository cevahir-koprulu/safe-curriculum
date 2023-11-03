import numpy as np
from deep_sprl.teachers.util import Buffer
from deep_sprl.teachers.abstract_teacher import BaseWrapper


class DummyWrapper(BaseWrapper):

    def __init__(self, env, sp_teacher, discount_factor, context_visible, reward_from_info=False,
                 use_undiscounted_reward=False, episodes_per_update=50):
        self.use_undiscounted_reward = use_undiscounted_reward

        BaseWrapper.__init__(self, env, sp_teacher, discount_factor, context_visible,
                             reward_from_info=reward_from_info,
                             episodes_per_update=episodes_per_update)
        self.context_buffer = Buffer(3, episodes_per_update + 1, True)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        ret = undiscounted_reward if self.use_undiscounted_reward else discounted_reward
        self.context_buffer.update_buffer((cur_initial_state, cur_context, ret))
        if len(self.context_buffer) >= self.episodes_per_update:
            __, contexts, returns = self.context_buffer.read_buffer()

    def get_context_buffer(self):
        ins, cons, disc_rews = self.context_buffer.read_buffer()
        return np.array(ins), np.array(cons), np.array(disc_rews)

    def get_aux_context_buffer(self):
        aux_ins, aux_cons, aux_disc_rews = self.aux_context_buffer.read_buffer()
        return np.array(aux_ins), np.array(aux_cons), np.array(aux_disc_rews)