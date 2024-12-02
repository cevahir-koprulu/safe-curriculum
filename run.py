import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import deep_sprl.environments
import torch


def main():
    parser = argparse.ArgumentParser("Safe curriculum learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="self_paced",
                        choices=["default", "random", 
                                 "constrained_self_paced", "self_paced", 
                                 "wasserstein", "constrained_wasserstein",  "wasserstein4cost",
                                 "alp_gmm", "goal_gan", "acl", "plr", "vds"])
    parser.add_argument("--learner", type=str, default="PPOLag", 
                        choices=["PPO", "SAC", "PPOLag", "CPO", "FOCOPS", "PCPO"])
    parser.add_argument("--env", type=str, default="safety_door_2d",
                        choices=["safety_point_mass_1d", "safety_point_mass_2d", "safety_cartpole_2d",
                                 "safety_door_2d", "safety_maze_3d", 
                                 "safety_goal_3d", "safety_goal_noconflict_3d", "safety_goal_with_vases_3d",
                                 "safety_passage_3d", "safety_passage_push_3d",
                                 "safety_push_box_3d", "safety_push_4d", "safety_push_3d", "safety_reach_3d",
                                 "safety_ant_3d", "safety_doggo_3d"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--eval_type", type=int, default=0, choices=[0, 1],
                        help="0: target distribution, 1: target distribution but homogeneous")
    parser.add_argument('--eval_training', action='store_true')
    parser.add_argument("--device", type=str, default="cuda:0")

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    torch.set_num_threads(args.n_cores)

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.env == "safety_point_mass_2d":
        from deep_sprl.experiments import SafetyPointMass2DExperiment
        exp = SafetyPointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_point_mass_1d":
        from deep_sprl.experiments import SafetyPointMass1DExperiment
        exp = SafetyPointMass1DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_cartpole_2d":
        from deep_sprl.experiments import SafetyCartpole2DExperiment
        exp = SafetyCartpole2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_door_2d":
        from deep_sprl.experiments import SafetyDoor2DExperiment
        exp = SafetyDoor2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_maze_3d":
        from deep_sprl.experiments import SafetyMaze3DExperiment
        exp = SafetyMaze3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_goal_3d":
        from deep_sprl.experiments import SafetyGoal3DExperiment
        exp = SafetyGoal3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_goal_noconflict_3d":
        from deep_sprl.experiments import SafetyGoalNoConflict3DExperiment
        exp = SafetyGoalNoConflict3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_goal_with_vases_3d":
        from deep_sprl.experiments import SafetyGoalWithVases3DExperiment
        exp = SafetyGoalWithVases3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_passage_3d":
        from deep_sprl.experiments import SafetyPassage3DExperiment
        exp = SafetyPassage3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_passage_push_3d":
        from deep_sprl.experiments import SafetyPassagePush3DExperiment
        exp = SafetyPassagePush3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_push_box_3d":
        from deep_sprl.experiments import SafetyPushBox3DExperiment
        exp = SafetyPushBox3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_push_4d":
        from deep_sprl.experiments import SafetyPush4DExperiment
        exp = SafetyPush4DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_push_3d":
        from deep_sprl.experiments import SafetyPush3DExperiment
        exp = SafetyPush3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_reach_3d":
        from deep_sprl.experiments import SafetyReach3DExperiment
        exp = SafetyReach3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_ant_3d":
        from deep_sprl.experiments import SafetyAnt3DExperiment
        exp = SafetyAnt3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "safety_doggo_3d":
        from deep_sprl.experiments import SafetyDoggo3DExperiment
        exp = SafetyDoggo3DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)   
    else:
        raise RuntimeError("Unknown environment '%s'!" % args.env)

    if args.train:
        exp.train()

    if args.eval:
        exp.evaluate(eval_type=args.eval_type)

    if args.eval_training:
        exp.evaluate_training_performance()


if __name__ == "__main__":
    main()
