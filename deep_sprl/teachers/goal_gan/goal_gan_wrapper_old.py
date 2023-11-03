from deep_sprl.teachers.abstract_teacher import BaseWrapper


class GoalGANWrapper(BaseWrapper):

    def __init__(self, env, goal_gan, discount_factor, context_visible, context_post_processing=None):
        BaseWrapper.__init__(self, env, goal_gan, discount_factor, context_visible,
                             context_post_processing=context_post_processing)

    def done_callback(self, step, cur_initial_state, cur_context, discounted_reward, undiscounted_reward):
        self.teacher.update(cur_context, float(step[-1]["success"]))
