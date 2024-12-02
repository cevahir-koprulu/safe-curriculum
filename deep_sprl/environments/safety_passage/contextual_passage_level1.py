from __future__ import annotations

from typing import Dict
import numpy as np
import safety_gymnasium
from safety_gymnasium.assets.geoms import Hazards, Goal, Pillars
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.builder import Builder
from safety_gymnasium.utils.task_utils import get_task_class_name

HAZARD_LOCATIONS = []
HAZARD_SIZE = 0.25
for x in range(5):
    pos_x = -0.5 + HAZARD_SIZE*x
    # for j in range(5):
    #     pos_y = 0. - HAZARD_SIZE*j
    for j in range(7):
        pos_y = 0.5 - HAZARD_SIZE*j
        HAZARD_LOCATIONS.append((pos_x, pos_y))

PILLAR_LOCATIONS = []
PILLAR_SIZE = 0.25
# for j in range(3):
#     pos_y = 0.25 - 2*PILLAR_SIZE*j
for j in range(4):
    pos_y = 0.75 - 2*PILLAR_SIZE*j
    PILLAR_LOCATIONS.append((-.75, pos_y))
    PILLAR_LOCATIONS.append((.75, pos_y))

# for j in range(8):
#     pos_y = 1.5 - 2*PILLAR_SIZE*j
for j in range(9):
    pos_y = 2. - 2*PILLAR_SIZE*j
    PILLAR_LOCATIONS.append((-2., pos_y))
    PILLAR_LOCATIONS.append((2., pos_y))
for i in range(8):
    pos_x = -1.75 + 2*PILLAR_SIZE*i
    # PILLAR_LOCATIONS.append((pos_x, 2.))
    PILLAR_LOCATIONS.append((pos_x, 2.5))
    PILLAR_LOCATIONS.append((pos_x, -2.5))


class ContextualPassageLevel1(BaseTask):
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
        # self.placements_conf.extents = np.array([-0.1, -1.7, 0.1, -1.5])
        self.placements_conf.extents = np.array([-0.1, -2.2, 0.1, -2])

        # Context determines the position of the goal and its size, i.e., tolerance for success
        self._add_geoms(Goal(size = self._context[2], keepout = 0, 
                             locations=[(self._context[0], self._context[1])])) 
        # Hazards are on the left side
        self._add_geoms(Hazards(size = HAZARD_SIZE, keepout = 0, is_constrained=True,
                                num = len(HAZARD_LOCATIONS), locations=HAZARD_LOCATIONS))
        # Pillars are all around and in the center
        self._add_geoms(Pillars(size = PILLAR_SIZE, height=0.15, keepout = 0, is_constrained=False,
                              num = len(PILLAR_LOCATIONS), locations=PILLAR_LOCATIONS))
        # Calculate the specific data members needed for the reward
        self.last_dist_goal = None
        # self.render(width=100, height=100, mode='rgb_array', camera_name="fixedfar")
        self._is_load_static_geoms = False
           
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward
            
    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        # Define the location of the target
        # If there is a goal in the environment, the same position as the goal
        # Can be undefined
        return self.goal.pos

tasks = {
    "ContextualPassageLevel1": ContextualPassageLevel1
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
   

