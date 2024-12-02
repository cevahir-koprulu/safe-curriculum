from __future__ import annotations

from typing import Dict
import numpy as np
import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards, Goal, Pillars
from safety_gymnasium.assets.free_geoms import PushBox
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name
        
HAZARD_LOCATIONS = []
HAZARD_SIZE = 0.25
for x in range(7):
    pos_x = -2.25 + HAZARD_SIZE*x
    for j in range(5):
        pos_y = 0. - HAZARD_SIZE*j
        HAZARD_LOCATIONS.append((pos_x, pos_y))

PILLAR_LOCATIONS = []
PILLAR_SIZE = 0.25
for x in range(3):
    pos_x = -0.5 + 2*PILLAR_SIZE*x
    for j in range(3):
        pos_y = 0.25 - 2*PILLAR_SIZE*j
        PILLAR_LOCATIONS.append((pos_x, pos_y))
for j in range(11):
    pos_y = 2.25 - 2*PILLAR_SIZE*j
    PILLAR_LOCATIONS.append((-2.5, pos_y))
    PILLAR_LOCATIONS.append((2.5, pos_y))
for i in range(9):
    pos_x = -2 + 2*PILLAR_SIZE*i
    PILLAR_LOCATIONS.append((pos_x, 2.5))
    PILLAR_LOCATIONS.append((pos_x, -3))

class ContextualPushBoxLevel1(BaseTask):
    """
    Custom safety gym environment
    """
    def __init__(
        self, 
        config:Dict=dict(),
    ):
        if 'context' not in config:
            raise ValueError("Context must be specified in config")
        else:
            assert config['context'].shape[0] == 3
            self._context = config['context']
            config.pop('context', None)
        super().__init__(config=config)

        # - in x is to the right
        # - in y is to the top
        # (0, 0) is in the middle

        # Agent start on the bottom left
        # self.placements_conf.extents = np.array([-1.5, -2, -1.5, -2])

        # Agent placed with respect to the goal
        if self._context[0] < 0 and self._context[1] < -0.25:
            # self.placements_conf.extents = np.array([1.5, -2, 1.5, -2])
            self.placements_conf.extents = np.array([1.5, -1.75, 1.5, -1.75])
        else:
            # self.placements_conf.extents = np.array([-1.5, -2, -1.5, -2])
            self.placements_conf.extents = np.array([-1.5, -1.75, -1.5, -1.75])

        # Context determines the position of the goal and its size, i.e., tolerance for success
        self._add_geoms(Goal(size = self._context[2], keepout = 0, 
                             locations=[(self._context[0], self._context[1])])) 
        self._add_free_geoms(PushBox(null_dist=0, 
                                     size=0.125, 
                                    #  locations=[(-1, -2)], 
                                    #  locations=[(0., -2)], 
                                     locations=[(0., -1.75)], 
                                    # placements=[(-0.1, -1.7, 0.1, -1.5)],
                                     keepout=0., 
                                     density=0.0005))
        # Hazards are on the left side
        self._add_geoms(Hazards(size = HAZARD_SIZE, keepout = 0, is_constrained=True,
                                num = len(HAZARD_LOCATIONS), locations=HAZARD_LOCATIONS))
        # Pillars are all around and in the center
        self._add_geoms(Pillars(size = PILLAR_SIZE, height=0.15, keepout = 0, is_constrained=False,
                              num = len(PILLAR_LOCATIONS), locations=PILLAR_LOCATIONS))
        self.last_dist_box = None
        self.last_box_goal = None
        self.last_dist_goal = None

        # self.render(width=100, height=100, mode='rgb_array', camera_name="fixedfar")
        self._is_load_static_geoms = False
           
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0
        # Distance from agent to box
        dist_box = self.dist_box()
        # pylint: disable-next=no-member
        gate_dist_box_reward = self.last_dist_box > self.push_box.null_dist * self.push_box.size
        reward += (
            # pylint: disable-next=no-member
            (self.last_dist_box - dist_box)
            * self.push_box.reward_box_dist  # pylint: disable=no-member
            * gate_dist_box_reward
        )
        self.last_dist_box = dist_box

        # Distance from box to goal
        dist_box_goal = self.dist_box_goal()
        # pylint: disable-next=no-member
        reward += (self.last_box_goal - dist_box_goal) * self.push_box.reward_box_goal
        self.last_box_goal = dist_box_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal  # pylint: disable=no-member

        return reward
   
    def dist_box(self):
        """Return the distance. from the agent to the box (in XY plane only)"""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.agent.pos)))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position."""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.goal.pos)))
            
    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()
        self.push_box_rotation_matrix = self.push_box.engine.data.body(self.push_box.name).xmat.copy()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_box_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        # Define the location of the target
        # If there is a goal in the environment, the same position as the goal
        # Can be undefined
        return self.goal.pos

tasks = {
    "ContextualPushBoxLevel1": ContextualPushBoxLevel1
}
class CustomBuilder(Builder):
   def _get_task(self):
        class_name = get_task_class_name(self.task_id)
        if class_name in tasks:
            task_class = tasks[class_name]
            task = task_class(config=self.config)
            task.build_observation_space()
        else:
            task = super()._get_task()    
        return task
   

