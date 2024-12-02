import os
import sys
import math
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from statannotations.Annotator import Annotator
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def load_training_results(filepath):
    with open(filepath, "rb") as f:
       _rew, disc_rew, _cost, disc_cost, succ, step_len, context_trace = pickle.load(f)
    return disc_rew, disc_cost, succ

def load_eval_results(filepath):
    results = np.load(filepath)
    return results[:, 1], results[:, -1], results[:, -2]

def get_results(base_dir, seeds, iterations, cost_th=0., results_from=set(["eval", "train"]), 
                only_last=False, plot_seeds=True):
    res_dict = {}
    for result_from in results_from:
        regret = {seed: 0. for seed in seeds}
        ignore_seed = {seed: False for seed in seeds}
        for iteration in iterations:
            if only_last and iteration != iterations[-1]:
                continue
            rets = []
            costs = []
            succs = []
            for seed in seeds:
                if ignore_seed[seed]:
                    continue
                path = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}")
                if result_from == "train":
                    if not os.path.exists(os.path.join(path, "context_trace.pkl")):
                        ignore_seed[seed] = True
                        del regret[seed]
                        print(f"ignore seed {seed}")
                        continue
                    disc_rewards, cost, success = (0., 0., 0.) if iteration == 0 \
                        else load_training_results(os.path.join(path, "context_trace.pkl"))
                elif result_from == "eval":
                    if not os.path.exists(os.path.join(path, "performance.npy")):
                        ignore_seed[seed] = True
                        continue
                    disc_rewards, cost, success = load_eval_results(os.path.join(path, "performance.npy"))
                else:
                    raise ValueError(f"{result_from} should be either 'train' or 'eval'")
                if iteration == iterations[-1]:
                    rets.append(np.mean(disc_rewards))
                    costs.append(np.mean(cost))
                    succs.append(np.mean(success))
                regret[seed] += max(np.mean(cost)-cost_th, 0.)
        if plot_seeds:
            res_dict[result_from] = {
                "return": np.array(rets),
                "cost": np.array(costs),
                "success": np.array(succs),
                "regret": np.array([regret[seed] for seed in seeds]),
            }
        else:
            res_dict[result_from] = {
                "return": np.median(np.array(rets)),
                "cost": np.median(np.array(costs)),
                "success": np.median(np.array(succs)),
                "regret": np.median(np.array([regret[seed] for seed in seeds])),
            }
        print(f"result from: {result_from}")
        print(res_dict[result_from])
    return res_dict

def plot_scatter(plot_for, algo_res_dict, algorithms, figpath, figsize, setting, fontsize):
    f, ax = plt.subplots(figsize=figsize)
    for algo in algorithms:
        x = algo_res_dict[algo][plot_for["x"]["from"]][plot_for["x"]["metric"]]
        y = algo_res_dict[algo][plot_for["y"]["from"]][plot_for["y"]["metric"]]
        ax.scatter(x, y, color=algorithms[algo]["color"], label=algorithms[algo]["label"], 
                   marker=algorithms[algo]["marker"])
    ax.set(xlabel=f"{plot_for['x']['from']}-{setting[plot_for['x']['metric']]['label']}",
           ylabel=f"{plot_for['y']['from']}-{setting[plot_for['y']['metric']]['label']}")
    ax.set_xlim(setting[plot_for["x"]["metric"]]['lim'][plot_for["x"]["from"]])      
    ax.set_ylim(setting[plot_for["y"]["metric"]]['lim'][plot_for["y"]["from"]])
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    # legend
    ax.legend(loc='lower right', fontsize=fontsize)
    plt.savefig(figpath, dpi=500, bbox_inches='tight')

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra,
                 plot_for_list, plot_seeds):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    figsize = setting["figsize"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    results_from = set()
    only_last = True
    for plot_for in plot_for_list:
        results_from.add(plot_for["x"]["from"])
        results_from.add(plot_for["y"]["from"])
        if "regret" in plot_for["x"]["metric"] or "regret" in plot_for["y"]["metric"]:
            only_last = False
    print(results_from)

    algo_res_dict = {}
    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        model = algorithms[cur_algo]["model"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        res_dict = get_results(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
            cost_th=setting["cost_threshold"],
            results_from=results_from,
            only_last=only_last,
            plot_seeds=plot_seeds,
        )
        algo_res_dict[cur_algo] = res_dict

    for plot_for in plot_for_list:
        plot_info = f"_{plot_for['x']['from']}{plot_for['x']['metric']}_vs_{plot_for['y']['from']}{plot_for['y']['metric']}"
        plot_info_seeds = f"_seeds" if plot_seeds else ""
        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{plot_info}{plot_info_seeds}{figname_extra}.png")
        plot_scatter(plot_for, algo_res_dict, algorithms, figpath, figsize, setting['metrics'], setting["fontsize"])

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 11)]
    # env = "safety_maze_3d"
    # figname_extra = ""

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 11)]
    # env = "safety_door_2d_narrow"
    # figname_extra = ""

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 6)]
    # env = "safety_goal_3d"
    # figname_extra = ""

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 6)]
    # env = "safety_goal_noconflict_3d"
    # figname_extra = "_s1-5"

    num_updates_per_iteration = 5
    seeds = [str(i) for i in range(1, 6)]
    env = "safety_passage_3d"
    figname_extra = "_s1-5"

    plot_seeds = True

    plot_for_list = [
        {'x': {'from': 'eval', 'metric': 'cost'},
         'y': {'from': 'eval', 'metric': 'success'}},
        {'x': {'from': 'train', 'metric': 'regret'},
         'y': {'from': 'eval', 'metric': 'success'}},
        {'x': {'from': 'train', 'metric': 'regret'},
         'y': {'from': 'eval', 'metric': 'cost'}},
    ]

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
                "label": "NS\nCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT\n4C",
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
                "label": "Goal\nGAN",
                # "label": "GGAN",
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
                "label": "NS\nCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT\n4C",
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
                "label": "Goal\nGAN",
                # "label": "G\nGAN",
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
                "label": "NS\nCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                # "label": "CURROT4Cost",
                "label": "CRT\n4C",
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
                "label": "Goal\nGAN",
                # "label": "GGAN",
                "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "gold",
            },
        },
        "safety_goal_noconflict_3d": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.3_RAS=10",
                "color": "red",
            }, 
            "CRT": {
                "algorithm": "wasserstein",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
            },  
            "CRT4Cost": {
                "algorithm": "wasserstein4cost",
                "label": "CRT4Cost",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.5",
                "color": "limegreen",
            }, 
            "NSCRT": {
                "algorithm": "wasserstein",
                "label": "NSCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            }, 
        },
        "safety_passage_3d": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.5_RAS=10",
                "color": "red",
                "marker": "o",
            },
            "CRT": {
                "algorithm": "wasserstein",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
                "marker": "s",
            },  
            "NSCRT": {
                "algorithm": "wasserstein",
                "label": "NSCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
                "marker": "D",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                "label": "CRT4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.5",
                "color": "lawngreen",
                "marker": "X",
            },
            "DEF": {
                "algorithm": "default",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
                "marker": "P",
            },
            # "ALP": {
            #     "algorithm": "alp_gmm",
            #     "label": "ALP",
            #     "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=200_AG_MAX_SIZE=500_AG_P_RAND=0.2",
            #     "color": "purple",
            # },
            # "PLR": {
            #     "algorithm": "plr",
            #     "label": "PLR",
            #     "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.15_PLR_REPLAY_RATE=0.55_PLR_RHO=0.45",
            #     "color": "darkcyan",
            # },
            # "SPDL": {
            #     "algorithm": "self_paced",
            #     "label": "SPDL",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_DIST_TYPE=gaussian_INIT_VAR=0.0_KL_EPS=0.25_PEN_COEFT=0.0",
            #     "color": "cyan",
            # },
            # "GGAN": {
            #     "algorithm": "goal_gan",
            #     "label": "Goal\nGAN",
            #     "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
            #     "color": "gold",
            # },
        },
    }

    settings = {
        "safety_maze_3d":{
            "cost_threshold": 0.,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "figsize": (10, 4),
            "fontsize": 22,
            "ylabel": {
                "return": 'Cumulative Reward',
                "cost": 'Cumulative Cost',
                "regret": 'CV Regret',
                "success": 'Success Rate',
            },
        },
        "safety_door_2d_narrow":{
            "cost_threshold": 0.,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "figsize": (10, 4),
            "fontsize": 22,
            "ylabel": {
                "return": 'Cumulative Reward',
                "cost": 'Cumulative Cost',
                "regret": 'CV Regret',
                "success": 'Success Rate',
            },
        },
        "safety_goal_3d":{
            "cost_threshold": 0.,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "figsize": (10, 4),
            "fontsize": 22,
            "ylabel": {
                "return": 'Cumulative Reward',
                "cost": 'Cumulative Cost',
                "regret": 'CV Regret',
                "success": 'Success Rate',
            },
        },
        "safety_goal_noconflict_3d":{
            "cost_threshold": 0.,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "figsize": (10, 4),
            "fontsize": 22,
            "metrics": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": [1., 1.6],
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": [-0.1, 1.],
                },
                "regret": {
                    "ylabel": 'CV Regret',
                    "ylim": [-0.1, 2.5],#20],
                },
                "success": {
                    "ylabel": 'Success Rate',
                    "ylim": [0.5, 1.1],
                },
            },
            # "ylabel": {
            #     "return": 'Cumulative Reward',
            #     "cost": 'Cumulative Cost',
            #     "regret": 'CV Regret',
            #     "success": 'Success Rate',
            # },
        },
        "safety_passage_3d":{
            "cost_threshold": 0.,
            "num_iters": 200,
            "steps_per_iter": 10000,
            "figsize": (10, 10),
            "fontsize": 22,
            "metrics": {
                "return": {
                    "label": 'Reward',
                    "lim": {
                        'eval': [0., 1.25],
                        'train': [0., 1.],
                    }
                },
                "cost": {
                    "label": 'Cost',
                    "lim": {
                        'eval': [-0.5, 3.],
                        'train': [-0.5, 10.],
                    }
                },
                "regret": {
                    "label": 'CV Regret',
                    "lim": {
                        'eval': [0., 75.],
                        'train': [0., 75.],
                    }
                },
                "success": {
                    "label": 'Success',
                    'lim': {
                        'eval': [-0.1, 1.1],
                        'train': [-0.1, 1.1],
                    }
                },
            },
        },
    }

    plot_results(
        base_log_dir=base_log_dir,
        num_updates_per_iteration=num_updates_per_iteration,
        seeds=seeds,
        env=env,
        setting=settings[env],
        algorithms=algorithms[env],
        figname_extra=figname_extra,
        plot_for_list=plot_for_list,
        plot_seeds=plot_seeds,
        )


if __name__ == "__main__":
    main()
