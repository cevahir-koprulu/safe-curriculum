import argparse
from deep_sprl.util.parameter_parser import parse_parameters
import deep_sprl.environments
import torch


def main():
    parser = argparse.ArgumentParser("Self-Paced Learning experiment runner")
    parser.add_argument("--base_log_dir", type=str, default="logs")
    parser.add_argument("--type", type=str, default="self_paced",
                        choices=["default", "random", "self_paced", "wasserstein", "alp_gmm",
                                 "goal_gan", "acl", "plr", "vds"])
    parser.add_argument("--learner", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--env", type=str, default="safety_point_mass_2d",
                        choices=["safety_point_mass_2d", "point_mass_2d", "point_mass_2d_heavytailed", "lunar_lander_2d_heavytailed"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_cores", type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument("--eval_type", type=int, default=0, choices=[0, 1],
                        help="0: target distribution, 1: all contexts")
    parser.add_argument('--eval_training', action='store_true')
    parser.add_argument("--device", type=str, default="cuda")

    args, remainder = parser.parse_known_args()
    parameters = parse_parameters(remainder)

    torch.set_num_threads(args.n_cores)

    if args.device != "cpu" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.env == "point_mass_2d":
        from deep_sprl.experiments import PointMass2DExperiment
        exp = PointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
    elif args.env == "point_mass_2d_heavytailed":
        from deep_sprl.experiments import PointMass2DHeavyTailedExperiment
        exp = PointMass2DHeavyTailedExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed,
                                               args.device)
    elif args.env == "lunar_lander_2d_heavytailed":
        from deep_sprl.experiments import LunarLander2DHeavyTailedExperiment
        exp = LunarLander2DHeavyTailedExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed,
                                                   args.device)        
    elif args.env == "safety_point_mass_2d":
        from deep_sprl.experiments import SafetyPointMass2DExperiment
        exp = SafetyPointMass2DExperiment(args.base_log_dir, args.type, args.learner, parameters, args.seed, args.device)
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
