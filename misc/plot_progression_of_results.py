import os
import sys
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def update_results(res_dict, res, use_mean_and_std=False):
    if use_mean_and_std:
        res_dict["mid"].append(np.mean(res, axis=0))
        res_dict["qlow"].append(np.mean(res, axis=0)-np.std(res, axis=0))
        res_dict["qhigh"].append(np.mean(res, axis=0)+np.std(res, axis=0))
    else:
        res_dict["mid"].append(np.median(res, axis=0))
        res_dict["qlow"].append(np.quantile(res, 0.25, axis=0))
        res_dict["qhigh"].append(np.quantile(res, 0.75, axis=0))
    res_dict["min"].append(np.min(res, axis=0))
    res_dict["max"].append(np.max(res, axis=0))

def load_training_results(filepath):
    with open(filepath, "rb") as f:
       _rew, disc_rew, _cost, disc_cost, succ, step_len, context_trace = pickle.load(f)
    return disc_rew, disc_cost, succ

def load_eval_results(filepath):
    results = np.load(filepath)
    return results[:, 1], results[:, -1], results[:, -2]

def get_results(base_dir, seeds, iterations, cost_th=0., use_mean_and_std=False, results_from="eval"):
    res_dict = {
        "return": {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []},
        "cost": {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []},
        "success": {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []},
        "regret": {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []},
    }
    regret = {seed: 0. for seed in seeds}
    ignore_seed = {seed: False for seed in seeds}
    max_iter = 0
    final_rets = None
    final_costs = None
    final_succs = None
    for iteration in iterations:
        rets = []
        costs = []
        succs = []
        for seed in seeds:
            if ignore_seed[seed]:
                continue
            path = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}")
            if results_from == "training":
                if not os.path.exists(os.path.join(path, "context_trace.pkl")):
                    ignore_seed[seed] = True
                    print(f"ignore seed {seed}")
                    continue
                disc_rewards, cost, success = (0., 0., 0.) if iteration == 0 \
                    else load_training_results(os.path.join(path, "context_trace.pkl"))
            elif results_from == "eval":
                if not os.path.exists(os.path.join(path, "performance.npy")):
                    ignore_seed[seed] = True
                    continue
                disc_rewards, cost, success = load_eval_results(os.path.join(path, "performance.npy"))
            else:
                raise ValueError("results_from should be either 'training' or 'eval'")
            max_iter = max(max_iter, iteration)
            rets.append(np.mean(disc_rewards))
            costs.append(np.mean(cost))
            succs.append(np.mean(success))
            regret[seed] += max(np.mean(cost)-cost_th, 0.)
        if len(rets) > 0:
            final_rets, final_costs, final_succs = rets, costs, succs
            update_results(res_dict["return"], np.array(rets), use_mean_and_std)
            update_results(res_dict["cost"], np.array(costs), use_mean_and_std)
            update_results(res_dict["success"], np.array(succs), use_mean_and_std)
            update_results(res_dict["regret"], np.array(list(regret.values())), use_mean_and_std)
    print(final_rets, final_costs, final_succs)
    print(f"success: {res_dict['success']['mid']}")
    return res_dict, max_iter

def plot_for_separate(plot_for, results_from, algo_res_dict, algos, labels, colors):
    for algo_i in range(len(algos)):
        algo = algos[algo_i]
        label = labels[algo_i]
        color = colors[algo_i]
        average_mid = algo_res_dict[algo]['res_dict'][plot_for]["mid"]
        average_qlow =  algo_res_dict[algo]['res_dict'][plot_for]["qlow"]
        average_qhigh =  algo_res_dict[algo]['res_dict'][plot_for]["qhigh"]
        average_min =  algo_res_dict[algo]['res_dict'][plot_for]["min"]
        average_max =  algo_res_dict[algo]['res_dict'][plot_for]["max"]
        s_idx = 1 if results_from == "training" else 0
        plt.plot(algo_res_dict[algo]['iterations_step'][s_idx:], average_mid[s_idx:], color=color, linewidth=2.0, label=f"{label}", marker=".")
        plt.fill_between(algo_res_dict[algo]['iterations_step'][s_idx:], average_qlow[s_idx:], average_qhigh[s_idx:], color=color, alpha=0.4)
        # plt.fill_between(iterations_step[s_idx:], average_min[s_idx:], average_max[s_idx:], color=color, alpha=0.2)


def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra, 
                 use_mean_and_std, plot_for_list, results_from):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    plot_settings = setting["plot_settings"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    final_iterations_step = iterations*steps_per_iter

    algos = list(algorithms.keys())
    algos.reverse()

    algo_res_dict = {}
    colors = []
    labels = []
    for cur_algo in algos:
        iterations_step = iterations*steps_per_iter

        algorithm = algorithms[cur_algo]["algorithm"]
        model = algorithms[cur_algo]["model"]
        labels.append(algorithms[cur_algo]["label"])
        colors.append(algorithms[cur_algo]["color"])
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        res_dict, max_iter = get_results(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
            cost_th=setting["cost_threshold"],
            use_mean_and_std=use_mean_and_std,
            results_from=results_from,
        )
        print(f"max_iter: {max_iter}")
        iterations_step = iterations_step[:(max_iter//num_updates_per_iteration+1)]

        algo_res_dict[cur_algo] = {"res_dict": res_dict, "iterations_step": iterations_step}

    num_alg = len(algos)
    lines = [Line2D([0], [0], color=colors[i], linestyle="-", marker="", linewidth=2.0)
            for i in range(num_alg)]
    for plot_for in plot_for_list:
        fig = plt.figure(figsize=figsize)
        plot_for_separate(plot_for, results_from, algo_res_dict, algos, labels, colors)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        plt.xlabel("Number of environment interactions")
        plt.ylabel(plot_settings[plot_for]["ylabel"])
        plt.xlim([final_iterations_step[0], final_iterations_step[-1]])
        plt.ylim(plot_settings[plot_for]["ylim"][results_from])
        plt.grid(True)

        lgd = fig.legend(lines[::-1], labels[::-1], ncol=num_alg//2+1, loc="upper center", bbox_to_anchor=bbox_to_anchor,
                        fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

        plot_info = f"_{results_from}_prog4{plot_for}"
        plot_info += "_mean" if use_mean_and_std else ""
        figname = ""
        for cur_algo_i, cur_algo in enumerate(algorithms):
            figname += cur_algo
            if cur_algo_i < len(algorithms)-1:
                figname += "_vs_"
        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{plot_info}{figname_extra}.pdf")
        print(figpath)
        plt.savefig(figpath, dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))
        plt.close()

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_updates_per_iteration = 10

    # seeds = [str(i) for i in range(1, 11)]
    # env = "safety_maze_3d"
    # figname_extra = ""

    # seeds = [str(i) for i in range(1, 11)]
    # env = "safety_door_2d_narrow"
    # figname_extra = ""

    seeds = [str(i) for i in range(1, 6)]
    env = "safety_goal_3d"
    figname_extra = ""

    use_mean_and_std = False
    
    # results_from = "eval"
    results_from = "training"

    plot_for_list = ["return", "cost", "regret", "success"]
    
    algorithms = {
        "safety_maze_3d": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=0.25_METRIC_EPS=1.25_RAS=10",
                "color": "red",
            },
            "CRT": {
                "algorithm": "wasserstein",
                # "label": "CURROT",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=0.0",
                "color": "blue",
            },
            "NSCRT": {
                "algorithm": "wasserstein",
                # "label": "NaiveSafeCURROT",
                "label": "NSCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=0.25_METRIC_EPS=1.25",
                "color": "lawngreen",
            },
            "DEF": {
                "algorithm": "default",
                # "label": "DEFAULT",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                # "label": "ALP-GMM",
                "label": "ALP",
                "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=200_AG_MAX_SIZE=500_AG_P_RAND=0.2",
                "color": "purple",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.15_PLR_REPLAY_RATE=0.55_PLR_RHO=0.45",
                "color": "darkcyan",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_DIST_TYPE=gaussian_INIT_VAR=0.0_KL_EPS=0.25_PEN_COEFT=0.0",
                "color": "cyan",
            },
            "GGAN": {
                "algorithm": "goal_gan",
                # "label": "GoalGAN",
                "label": "GGAN",
                "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "gold",
            },
        },
        "safety_door_2d_narrow": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=1.5_METRIC_EPS=0.5_RAS=10",
                "color": "red",
            },
            "CRT": {
                "algorithm": "wasserstein",
                # "label": "CURROT",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
            },
            "NSCRT": {
                "algorithm": "wasserstein",
                # "label": "NaiveSafeCURROT",
                "label": "NSCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.5_METRIC_EPS=0.5",
                "color": "lawngreen",
            },
            "DEF": {
                "algorithm": "default",
                # "label": "DEFAULT",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                # "label": "ALP-GMM",
                "label": "ALP",
                "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=100_AG_MAX_SIZE=500_AG_P_RAND=0.1",
                "color": "purple",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
                "color": "darkcyan",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_DIST_TYPE=gaussian_INIT_VAR=0.5_KL_EPS=0.25_PEN_COEFT=0.0",
                "color": "cyan",
            },
            "GGAN": {
                "algorithm": "goal_gan",
                # "label": "GoalGAN",
                "label": "GGAN",
                "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "gold",
            },
        },
        "safety_goal_3d": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.5_RAS=10",
                "color": "red",
            },
            "CRT": {
                "algorithm": "wasserstein",
                # "label": "CURROT",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
            },  
            "NSCRT": {
                "algorithm": "wasserstein",
                # "label": "NaiveSafeCURROT",
                "label": "NSCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.5",
                "color": "lawngreen",
            },
            "DEF": {
                "algorithm": "default",
                # "label": "DEFAULT",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
                # "label": "ALP-GMM",
                "label": "ALP",
                "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=200_AG_MAX_SIZE=500_AG_P_RAND=0.2",
                "color": "purple",
            },
            "PLR": {
                "algorithm": "plr",
                "label": "PLR",
                "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.15_PLR_REPLAY_RATE=0.55_PLR_RHO=0.45",
                "color": "darkcyan",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_DIST_TYPE=gaussian_INIT_VAR=0.0_KL_EPS=0.25_PEN_COEFT=0.0",
                "color": "cyan",
            },
            "GGAN": {
                "algorithm": "goal_gan",
                # "label": "GoalGAN",
                "label": "GGAN",
                "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "gold",
            },
        },
    }

    settings = {
        "safety_door_2d_narrow":{
            "cost_threshold": 0.0,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 22,
            "figsize": (10, 4),
            "bbox_to_anchor": (.5, 1.15),
            "plot_settings": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": {
                        'eval': [-5., 70.],
                        'training': [-5., 70.],
                    }
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": {
                        'eval': [-5.0, 30.],
                        'training': [-5.0, 20.],
                    }
                },
                "regret": {
                    "ylabel": 'Constraint Violation Regret',
                    "ylim": {
                        'eval': [-5.0, 1000.],
                        'training': [-5.0, 200.],
                    }
                },
                "success": {
                    "ylabel": 'Success Rate',
                    "ylim": {
                        'eval': [-0.1, 1.1],
                        'training': [-0.1, 1.1],
                    }
                },
            },
        },
        "safety_maze_3d":{
            "cost_threshold": 0.,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 22,
            "figsize": (10, 4),
            "bbox_to_anchor": (.5, 1.15),
            "plot_settings": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": {
                        'eval': [-90., -40.],
                        'training': [-90., -0.],
                    }
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": {
                        'eval': [-1.0, 8.],
                        'training': [-0.25, 4.],
                    }
                },
                "regret": {
                    "ylabel": 'Constraint Violation Regret',
                    "ylim": {
                        'eval': [0.0, 100.],
                        'training': [0.0, 40.],
                    }
                },
                "success": {
                    "ylabel": 'Success Rate',
                    'ylim': {
                        'eval': [-0.1, 1.1],
                        'training': [-0.1, 1.1],
                    }
                },
            },
        },
        "safety_goal_3d":{
            "cost_threshold": 0.,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "fontsize": 22,
            "figsize": (10, 4),
            "bbox_to_anchor": (.5, 1.15),
            "plot_settings": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": {
                        'eval': [0., 1.],
                        'training': [0., 1.],
                    }
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": {
                        'eval': [-0.5, 10.],
                        'training': [-0.5, 10.],
                    }
                },
                "regret": {
                    "ylabel": 'Constraint Violation Regret',
                    "ylim": {
                        'eval': [0., 50.],
                        'training': [0., 50.],
                    }
                },
                "success": {
                    "ylabel": 'Success Rate',
                    'ylim': {
                        'eval': [-0.1, 1.1],
                        'training': [-0.1, 1.1],
                    }
                },
            },
        },
    }

    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env],
        figname_extra=figname_extra,
        use_mean_and_std=use_mean_and_std,
        plot_for_list=plot_for_list,
        results_from=results_from,
        )

if __name__ == "__main__":
    main()
