# Do this import to ensure that the Gym environments get registered properly
import deep_sprl.environments
from .safety_door_2d_experiment import SafetyDoor2DExperiment
from .safety_maze_3d_experiment import SafetyMaze3DExperiment
from .safety_goal_3d_experiment import SafetyGoal3DExperiment
from .safety_goal_noconflict_3d_experiment import SafetyGoalNoConflict3DExperiment
from .safety_goal_with_vases_3d_experiment import SafetyGoalWithVases3DExperiment
from .safety_passage_3d_experiment import SafetyPassage3DExperiment
from .safety_passage_push_3d_experiment import SafetyPassagePush3DExperiment
from .safety_push_box_3d_experiment import SafetyPushBox3DExperiment
from .safety_push_4d_experiment import SafetyPush4DExperiment
from .safety_push_3d_experiment import SafetyPush3DExperiment
from .safety_reach_3d_experiment import SafetyReach3DExperiment
from .safety_cartpole_2d_experiment import SafetyCartpole2DExperiment
from .safety_point_mass_2d_experiment import SafetyPointMass2DExperiment
from .safety_point_mass_1d_experiment import SafetyPointMass1DExperiment
from .safety_ant_3d_experiment import SafetyAnt3DExperiment
from .safety_doggo_3d_experiment import SafetyDoggo3DExperiment
from .abstract_experiment import CurriculumType, Learner

__all__ = ['CurriculumType', 'Learner',
           'SafetyPointMass2DExperiment',
           'SafetyPointMass1DExperiment',
           'SafetyCartpole2DExperiment',
           'SafetyDoor2DExperiment',
           'SafetyMaze3DExperiment',
           'SafetyGoal3DExperiment',
           'SafetyGoalNoConflict3DExperiment',
           'SafetyGoalWithVases3DExperiment',
           'SafetyPassage3DExperiment',
           'SafetyPassagePush3DExperiment',
           'SafetyPushBox3DExperiment',
           'SafetyPush4DExperiment',
           'SafetyPush3DExperiment',
           'SafetyReach3DExperiment',
            'SafetyAnt3DExperiment',
            'SafetyDoggo3DExperiment'
           ] 
