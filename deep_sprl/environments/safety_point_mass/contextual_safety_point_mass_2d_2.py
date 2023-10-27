import numpy as np
from gym import Env
from .contextual_safety_point_mass import ContextualSafetyPointMass


class ContextualSafetyPointMass2D2(Env):

    def __init__(self, context=np.array([0., 0.])):
        self.env = ContextualSafetyPointMass(context=np.concatenate((context, [0.])),
                                             cost_coeff=10.0)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def set_context(self, context):
        self.env.context = np.concatenate((context, [0.]))

    def get_context(self):
        return self.env.context.copy()

    context = property(get_context, set_context)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode)
