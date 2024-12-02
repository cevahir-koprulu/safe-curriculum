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

def get_results(base_dir, seeds, iterations, cost_th=0., results_from="eval", only_last=False):
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
            if results_from == "training":
                if not os.path.exists(os.path.join(path, "context_trace.pkl")):
                    ignore_seed[seed] = True
                    del regret[seed]
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
            if iteration == iterations[-1]:
                rets.append(np.mean(disc_rewards))
                costs.append(np.mean(cost))
                succs.append(np.mean(success))
            regret[seed] += max(np.mean(cost)-cost_th, 0.)
    res_dict = {
        "return": np.array(rets),
        "cost": np.array(costs),
        "success": np.array(succs),
        "regret": np.array([regret[seed] for seed in seeds if seed in regret]),
    }
    print(res_dict)
    return res_dict

def plot_bar_for_separate(plot_for, algo_res_dict, algorithms, figpath, figsize, ylabel, ylim):
    df = pd.DataFrame()
    color_palette = {}
    labels = []
    for algo in algorithms:
        labels.append(algorithms[algo]["label"])
        color_palette[algorithms[algo]["label"]] = algorithms[algo]["color"]
        result = algo_res_dict[algo]["res_dict"]
        df_alg = pd.DataFrame()
        df_alg["algorithm"] = list([algorithms[algo]["label"] for _ in range(result[plot_for].shape[0])])
        df_alg[plot_for] = result[plot_for].tolist()
        df = pd.concat([df, df_alg])
    

    # for annotate in [False, True]:
    for annotate in [False]:
        f, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=df, x="algorithm", y=plot_for, order=labels, palette=color_palette, showfliers=False)
        ax.xaxis.grid(True)
        # ax.set(ylabel=ylabel, xlabel="Algorithm")
        ax.set(ylabel=ylabel, xlabel=None)
        ax.set_ylim(ylim)
        sns.despine(trim=True, left=True)
        if annotate:
            print("ANNOTATING!!!!")
            pairs = list(comb for comb in combinations(labels, r=2) if "SCG" in comb)
            annotator = Annotator(ax, pairs, data=df, x="algorithm", y=plot_for, order=labels)
            annotator.configure(
                # test="Mann-Whitney",
                test="t-test_welch",
                loc='outside',
                text_format="star",
                line_height=0.01,
                fontsize="small",
                # pvalue_thresholds=[[0.0001, '****'], [0.001, '***'], [0.01, '**'], [0.05, '*'], [1, 'ns']],
            )
            print(annotator.get_configuration())
            annotator.apply_and_annotate()
            dot_idx = figpath.rfind(".")
            figpath = figpath[:dot_idx]+f"_annotated{figpath[dot_idx:]}"
        else:
            print("NOT ANNOTATING!!!!")
    
        plt.savefig(figpath, dpi=500, bbox_inches='tight')
        # plt.close()

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra,
                 plot_for_list, results_from):
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
            only_last="regret" not in plot_for_list,
        )
        algo_res_dict[cur_algo] = {"res_dict": res_dict}

    for plot_for in plot_for_list:
        plot_info = f"_{results_from}_prog4{plot_for}"
        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{plot_info}_finalbar{figname_extra}.pdf")
        plot_bar_for_separate(plot_for, algo_res_dict, algorithms, figpath, figsize, setting["metrics"][plot_for]["ylabel"], setting["metrics"][plot_for]["ylim"])

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
    # figname_extra = "_s1-5"

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(1, 6)]
    # env = "safety_goal_noconflict_3d"
    # figname_extra = "_s1-5"

    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 6)]
    # env = "safety_passage_3d"
    # figname_extra = ""

    num_updates_per_iteration = 5
    seeds = ["1", "3", "4", "5", "6"]
    # seeds = [str(i) for i in range(1, 6)]
    env = "safety_passage_push_3d"
    figname_extra = "_s13456"

    results_from = "eval"
    # results_from = "training"

    # plot_for_list = ["cost", "success"]
    # plot_for_list = ["regret"]
    # plot_for_list = ["success"]
    # plot_for_list = ["return", "cost",]
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
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
            },  
            "NSCRT": {
                "algorithm": "wasserstein",
                "label": "NS\nCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                "label": "CRT\n4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.5",
                "color": "lawngreen",
            },
            "DEF": {
                "algorithm": "default",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
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
            },
            "CRT": {
                "algorithm": "wasserstein",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
            },  
            "NSCRT": {
                "algorithm": "wasserstein",
                "label": "NS\nCRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=1.0",
                "color": "magenta",
            },
            "CRT4C": {
                "algorithm": "wasserstein4cost",
                "label": "CRT\n4C",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.5",
                "color": "lawngreen",
            },
            "DEF": {
                "algorithm": "default",
                "label": "DEF",
                "model": "PPOLag_DELTA_CS=0.0",
                "color": "maroon",
            },
            "ALP": {
                "algorithm": "alp_gmm",
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
                "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
                "color": "gold",
            },
        },
        "safety_passage_push_3d": {
            "DCT=1": {
                "algorithm": "constrained_wasserstein",
                "label": r"$\tilde{D}=1$",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
                "color": "red",
            },
            "DCT=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": r"$\tilde{D}=0.5$",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=0.5_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
                "color": "blue",
            },
            "DCT=2": {
                "algorithm": "constrained_wasserstein",
                "label": r"$\tilde{D}=2$",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=2.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
                "color": "green",
            },
            # "AS=10": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$K_{ann}=10$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "red",
            # },
            # "AS=5": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$K_{ann}=5$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=5_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=5",
            #     "color": "blue",
            # },
            # "AS=20": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$K_{ann}=20$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=20_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=20",
            #     "color": "green",
            # },
            # "ATP=0.75": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$\alpha=0.75$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "red",
            # },
            # "ATP=0.5": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$\alpha=0.5$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.5_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "blue",
            # },
            # "ATP=0.875": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$\alpha=0.875$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.875_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "green",
            # },
            # "ATP=1": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": r"$\alpha=1$",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "cyan",
            # },  
            # "SCG": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "red",
            # },
            # "SCG_NoAnn": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG\nNoAnn",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
            #     "color": "blue",
            # },  
            # "SCG_NoAnn_NoPP": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG\nNoPP",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=False_PS=True_RAS=10",
            #     "color": "magenta",
            # },
            # "SCG_NoAnn_NoPS": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG\nNoPS",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=False_RAS=10",
            #     "color": "lawngreen",
            # },
            # "SCG_NoAnn_NoPPPS": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG\nNoPPPS",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=False_PS=False_RAS=10",
            #     "color": "maroon",
            # },
            # "PPOLag": {
            #     "algorithm": "default",
            #     "label": "DEF\nPPOLag",
            #     "model": "PPOLag_DELTA_CS=0.0",
            #     "color": "blue",
            # },  
            # "CPO": {
            #     "algorithm": "default",
            #     "label": "DEF\nCPO",
            #     "model": "CPO_DELTA_CS=0.0",
            #     "color": "magenta",
            # },
            # "PCPO": {
            #     "algorithm": "default",
            #     "label": "DEF\nPCPO",
            #     "model": "PCPO_DELTA_CS=0.0",
            #     "color": "lawngreen",
            # },
            # "FOCOPS": {
            #     "algorithm": "default",
            #     "label": "DEF\nFOCOPS",
            #     "model": "FOCOPS_DELTA_CS=0.0",
            #     "color": "maroon",
            # },
            # "CRT": {
            #     "algorithm": "wasserstein",
            #     "label": "CRT",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=0.0",
            #     "color": "blue",
            # },  
            # "NSCRT": {
            #     "algorithm": "wasserstein",
            #     "label": "NS\nCRT",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=1.0",
            #     "color": "magenta",
            # },
            # "CRT4C": {
            #     "algorithm": "wasserstein4cost",
            #     "label": "CRT\n4C",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.25",
            #     "color": "lawngreen",
            # },
            # "DEF": {
            #     "algorithm": "default",
            #     "label": "DEF",
            #     "model": "PPOLag_DELTA_CS=0.0",
            #     "color": "maroon",
            # },
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
        }
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
            "metrics": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": [0., 1.25],
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": [-0.1, 6],
                },
                "regret": {
                    "ylabel": 'CV Regret',
                    "ylim": [-1, 80],
                },
                "success": {
                    "ylabel": 'Success Rate',
                    "ylim": [-0.1, 1.1],
                },
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
            "figsize": (10, 4),
            "fontsize": 22,
            "metrics": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": [0., 1.]#1.25],
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": [-0.1, 7.5],#0.75],
                },
                "regret": {
                    "ylabel": 'CV Regret',
                    "ylim": [-1, 175],#20],
                },
                "success": {
                    "ylabel": 'Success Rate',
                    "ylim": [-0.1, 1.1],
                },
            },
        },
        "safety_passage_push_3d":{
            "cost_threshold": 0.,
            "num_iters": 300,
            "steps_per_iter": 10000,
            "figsize": (10, 4),
            "fontsize": 22,
            "metrics": {
                "return": {
                    "ylabel": 'Cumulative Reward',
                    "ylim": [0., 1],#0.8],#1.1],
                },
                "cost": {
                    "ylabel": 'Cumulative Cost',
                    "ylim": [-0.1, 1.5],#0.75],#5],#3],
                },
                "regret": {
                    "ylabel": 'CV Regret',
                    "ylim": [-1, 100],#150],#200],
                },
                "success": {
                    "ylabel": 'Success Rate',
                    "ylim": [-0.1, 1.1],
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
        results_from=results_from,
        )


if __name__ == "__main__":
    main()
