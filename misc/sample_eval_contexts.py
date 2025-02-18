import sys
sys.path.insert(0, '..')
import os
import math
import numpy as np
from deep_sprl.experiments.safety_point_mass_2d_experiment import SafetyPointMass2DExperiment
from deep_sprl.experiments.safety_point_mass_1d_experiment import SafetyPointMass1DExperiment
from deep_sprl.experiments.safety_cartpole_2d_experiment import SafetyCartpole2DExperiment
from deep_sprl.experiments.safety_door_2d_experiment import SafetyDoor2DExperiment
from deep_sprl.experiments.safety_maze_3d_experiment import SafetyMaze3DExperiment
from deep_sprl.experiments.safety_goal_3d_experiment import SafetyGoal3DExperiment
from deep_sprl.experiments.safety_passage_3d_experiment import SafetyPassage3DExperiment
from deep_sprl.experiments.safety_passage_push_3d_experiment import SafetyPassagePush3DExperiment
from deep_sprl.experiments.safety_push_3d_experiment import SafetyPush3DExperiment
from deep_sprl.experiments.safety_reach_3d_experiment import SafetyReach3DExperiment
from deep_sprl.experiments.safety_doggo_3d_experiment import SafetyDoggo3DExperiment
from pathlib import Path


def sample_contexts(target_sampler, bounds, num_contexts):
    lower_bounds = bounds["lower_bounds"]
    upper_bounds = bounds["upper_bounds"]
    contexts = np.clip(target_sampler(n=num_contexts), lower_bounds, upper_bounds)
    return contexts

def sample_contexts_hom(bounds, num_per_axis):
    lower_bounds = bounds["target_lower_bounds"]
    upper_bounds = bounds["target_upper_bounds"]
    dim = lower_bounds.shape[0]
    if dim == 1:
        contexts = np.linspace(lower_bounds[0], upper_bounds[0], num=num_per_axis)
    elif dim == 2:
        x, y = np.meshgrid(np.linspace(lower_bounds[0], upper_bounds[0], num=num_per_axis),
                           np.linspace(lower_bounds[1], upper_bounds[1], num=num_per_axis))
        x_ = x.reshape(-1, 1)
        y_ = y.reshape(-1, 1)
        contexts = np.concatenate((x_, y_), axis=1)
    elif dim == 3:
        x, y = np.meshgrid(np.linspace(lower_bounds[0], upper_bounds[0], num=num_per_axis[0]),
                            np.linspace(lower_bounds[1], upper_bounds[1], num=num_per_axis[1]),
                              )
        x_ = x.reshape(-1, 1)
        y_ = y.reshape(-1, 1)
        # contexts = np.concatenate((x_, y_, np.repeat(lower_bounds[2], x_.shape[0]).reshape(-1,1)), axis=1)
        contexts = np.concatenate((x_, y_, np.repeat(upper_bounds[2], x_.shape[0]).reshape(-1,1)), axis=1)
    return contexts

def main():
    ##################################
    num_contexts = 20
    eval_context_dir = f"{Path(os.getcwd()).parent}/eval_contexts"
    target_type = "narrow"
    # env = f"safety_door_2d_{target_type}"
    # env = "safety_maze_3d"
    # env = "safety_goal_3d"
    # env = "safety_passage_3d"
    # env = "safety_passage_push_3d"
    # env = "safety_push_3d"
    # env = "safety_reach_3d"
    env = "safety_doggo_3d"
    all_contexts = True
    all_contexts_hom = False
    num_per_axis = [25,5]
    ##################################

    if not os.path.exists(eval_context_dir):
        os.makedirs(eval_context_dir)

    if env[:-len(target_type) - 1] == "safety_point_mass_2d":
        exp = SafetyPointMass2DExperiment(base_log_dir="logs", curriculum_name="self_paced", 
                                          learner_name="PPO", parameters={"TARGET_TYPE": target_type},
                                          seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "safety_point_mass_1d":
        exp = SafetyPointMass1DExperiment(base_log_dir="logs", curriculum_name="self_paced", 
                                          learner_name="PPO", parameters={"TARGET_TYPE": target_type},
                                          seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "safety_cartpole_2d":
        exp = SafetyCartpole2DExperiment(base_log_dir="logs", curriculum_name="self_paced", 
                                         learner_name="PPO", parameters={"TARGET_TYPE": target_type},
                                         seed=1, device="cpu")
    elif env[:-len(target_type) - 1] == "safety_door_2d":
        exp = SafetyDoor2DExperiment(base_log_dir="logs", curriculum_name="self_paced", 
                                     learner_name="PPO", parameters={"TARGET_TYPE": target_type},
                                     seed=1, device="cpu")
    elif env == "safety_maze_3d":
        exp = SafetyMaze3DExperiment(base_log_dir="logs", curriculum_name="self_paced", 
                                     learner_name="PPO", parameters={},
                                     seed=1, device="cpu")
    elif env == "safety_goal_3d":
        exp = SafetyGoal3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                     learner_name="PPO", parameters={},
                                     seed=1, device="cpu")
    elif env == "safety_passage_3d":
        exp = SafetyPassage3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                        learner_name="PPO", parameters={},
                                        seed=5, device="cpu")
    elif env == "safety_passage_push_3d":
        exp = SafetyPassagePush3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                        learner_name="PPO", parameters={},
                                        seed=5, device="cpu")
    elif env == "safety_push_3d":
        exp = SafetyPush3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                        learner_name="PPO", parameters={},
                                        seed=5, device="cpu")
    elif env == "safety_reach_3d":
        exp = SafetyReach3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                        learner_name="PPO", parameters={},
                                        seed=5, device="cpu")
    elif env == "safety_doggo_3d":
        exp = SafetyDoggo3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                        learner_name="PPO", parameters={},
                                        seed=5, device="cpu")
    else:
        raise ValueError("Invalid environment")

    bounds = {
        "lower_bounds": exp.LOWER_CONTEXT_BOUNDS,
        "upper_bounds": exp.UPPER_CONTEXT_BOUNDS,
        "target_lower_bounds": exp.TARGET_LOWER_CONTEXT_BOUNDS,
        "target_upper_bounds": exp.TARGET_UPPER_CONTEXT_BOUNDS,
    }

    if all_contexts:
        contexts = sample_contexts(target_sampler=exp.target_sampler,
                                   bounds=bounds,
                                   num_contexts=num_contexts,)
        print(contexts)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_contexts"), contexts)

    if all_contexts_hom:
        contexts = sample_contexts_hom(bounds=bounds,
                                       num_per_axis=num_per_axis,)
        print(contexts)
        np.save(os.path.join(eval_context_dir,f"{env}_eval_hom_contexts"), contexts)


if __name__ == "__main__":
    main()