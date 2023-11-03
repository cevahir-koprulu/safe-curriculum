from typing import ClassVar

from deep_sprl.teachers.abstract_teacher import BaseWrapper
from omnisafe.envs.core import env_register

@env_register
class ALPGMMWrapper(BaseWrapper):
    _support_envs: ClassVar[list[str]] = ['ALPGMMWrapper-v0']
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True

    def __init__(self, 
                 env_id: str, 
                 env_id_actual: str, 
                 teacher, discount_factor, 
                 context_visible=True, 
                 reward_from_info=False,
                 context_post_processing=None, 
                 episodes_per_update=50, 
                 **kwargs):
        super().__init__(env_id, env_id_actual, teacher, discount_factor, context_visible, 
                         reward_from_info, context_post_processing, episodes_per_update, **kwargs)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward,
                      discounted_cost, undiscounted_cost):
        self.teacher.update(cur_context, discounted_reward)
