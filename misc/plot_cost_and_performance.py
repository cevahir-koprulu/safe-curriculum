import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def update_results(res_dict, res):
    res_dict["mid"].append(np.mean(res, axis=0))
    res_dict["qlow"].append(np.quantile(res, 0.25, axis=0))
    res_dict["qhigh"].append(np.quantile(res, 0.75, axis=0))
    res_dict["min"].append(np.min(res, axis=0))
    res_dict["max"].append(np.max(res, axis=0))

def get_results(base_dir, seeds, iterations):
    ret_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    cost_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    succ_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    for iteration in iterations:
        rets = []
        costs = []
        succs = []
        for seed in seeds:
            perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "performance.npy")
            if os.path.exists(perf_file):
                results = np.load(perf_file)
                disc_rewards = results[:, 1]
                cost = results[:, -1]
                success = results[:, -2]
                rets.append(np.mean(disc_rewards))
                costs.append(np.mean(cost))
                succs.append(np.mean(success))
        if len(rets) > 0:
            update_results(ret_dict, np.array(rets))
            update_results(cost_dict, np.array(costs))
            update_results(succ_dict, np.array(succs))
    return ret_dict, cost_dict, succ_dict

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    plot_success = setting["plot_success"]
    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    subplot_settings = setting["subplot_settings"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(2+int(plot_success),1, figsize=figsize, constrained_layout=True)
    alg_exp_mid = {}
    plt.suptitle("Evaluation wrt target distribution")

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        ret, cost, succ = get_results(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        expected_return_mid = ret["mid"]
        expected_return_qlow = ret["qlow"]
        expected_return_qhigh = ret["qhigh"]
        expected_return_min = ret["min"]
        expected_return_max = ret["max"]

        expected_cum_cost_mid = cost["mid"]
        expected_cum_cost_qlow = cost["qlow"]
        expected_cum_cost_qhigh = cost["qhigh"]
        expected_cum_cost_min = cost["min"]
        expected_cum_cost_max = cost["max"]

        alg_exp_mid[cur_algo] = expected_return_mid[-1]

        axes[0].plot(iterations_step, expected_return_mid, color=color, linewidth=2.0, label=f"{label}",marker=".")
        # axes[0].fill_between(iterations_step, expected_return_qlow, expected_return_qhigh, color=color, alpha=0.4)
        axes[0].fill_between(iterations_step, expected_return_min, expected_return_max, color=color, alpha=0.4)
        axes[1].plot(iterations_step, expected_cum_cost_mid, color=color, linewidth=2.0, marker=".")
        # axes[1].fill_between(iterations_step, expected_cum_cost_qlow, expected_cum_cost_qhigh, color=color, alpha=0.4)
        axes[1].fill_between(iterations_step, expected_cum_cost_min, expected_cum_cost_max, color=color, alpha=0.4)
        if plot_success:            
            expected_success_mid = succ["mid"]
            expected_success_qlow = succ["qlow"]
            expected_success_qhigh = succ["qhigh"]
            expected_success_min = succ["min"]
            expected_success_max = succ["max"]
            axes[2].plot(iterations_step, expected_success_mid, color=color, linewidth=2.0, marker=".")
            # axes[2].fill_between(iterations_step, expected_success_qlow, expected_success_qhigh, color=color, alpha=0.4)
            axes[2].fill_between(iterations_step, expected_success_min, expected_success_max, color=color, alpha=0.4)

    for i, ax in enumerate(axes):
        ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        # if i != len(axes) - 1:
        #     ax.set_xticks([])
        # else:
        #     ax.set_xlabel("Number of environment interactions")
        if i == len(axes)-1:
            ax.set_xlabel("Number of environment interactions")
        ax.set_xlim([iterations_step[0], iterations_step[-1]])
        ax.set_ylim(subplot_settings[i]["ylim"])
        ax.set_ylabel(subplot_settings[i]["ylabel"])
        ax.grid(True)


    sorted_alg_exp_mid = [b[0] for b in sorted(enumerate(list(alg_exp_mid.values()), ), key=lambda i: i[1])]
    colors = []
    labels = []
    num_alg = len(algorithms)
    for alg_i in sorted_alg_exp_mid:
        cur_algo = list(alg_exp_mid.keys())[alg_i]
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])
    # for cur_algo in algorithms:
    #     colors.append(algorithms[cur_algo]["color"])
    #     labels.append(algorithms[cur_algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=num_alg, loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, cur_algo in enumerate(algorithms):
        figname += cur_algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))

    figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{figname_extra}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_updates_per_iteration = 5
    seeds = [str(i) for i in range(1, 4)]
    env = "safety_door_2d_narrow"
    figname_extra = "_D=30_rExp0.2_lBorder"
    # env = "safety_point_mass_2d_narrow"
    # figname_extra = "KL_EPS=1.0"
    # env = "safety_cartpole_2d_narrow"
    # figname_extra = "_KL_EPS=1.0_im=0.7xth_0.2xth"

    algorithms = {
        "safety_door_2d_narrow": {
            "CSPDL2_KL=1.0": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_KL=1.0",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=1.0_KL_EPS=1.0",
                "color": "gray",
            },
            "SPDL2_KL=1.0": {
                "algorithm": "constrained_self_paced",
                "label": "SPDL2_KL=1.0",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=1.0_KL_EPS=1.0",
                "color": "tan",
            },
            "CSPDL2_KL=0.5": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_KL=0.5",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=1.0_KL_EPS=0.5",
                "color": "blue",
            },
            "SPDL2_KL=0.5": {
                "algorithm": "constrained_self_paced",
                "label": "SPDL2_KL=0.5",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=1.0_KL_EPS=0.5",
                "color": "green",
            },
            "DEF_Lag": {
                "algorithm": "default",
                "label": "DEF_Lag",
                "model": "PPOLag",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "PPO",
                "color": "magenta",
            },
        },
        "safety_cartpole_2d_narrow": {
            "CSPDL_D=10": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=10",
                "model": "PPO_DELTA=40.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "gray",
            },
            "CSPDL2_D=10": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=10",
                "model": "PPOLag_DELTA=10.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "tan",
            },
            "CSPDL_D=15": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=15",
                "model": "PPO_DELTA=15.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "blue",
            },
            "CSPDL2_D=15": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=15",
                "model": "PPOLag_DELTA=15.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "green",
            },
            "CSPDL_D=25": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=25",
                "model": "PPO_DELTA=25.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "red",
            },
            "CSPDL2_D=25": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=25",
                "model": "PPOLag_DELTA=25.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "brown",
            },
            # "SPDL_D=50": {
            #     "algorithm": "self_paced",
            #     "label": "SPDL_D=50",
            #     "model": "PPO_DELTA=50.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
            #     "color": "blue",
            # },
            # "SPDL2_D=50": {
            #     "algorithm": "self_paced",
            #     "label": "SPDL2_D=50",
            #     "model": "PPOLag_DELTA=50.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
            #     "color": "green",
            # },
            "DEF_Lag": {
                "algorithm": "default",
                "label": "DEF_Lag",
                "model": "PPOLag",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "PPO",
                "color": "magenta",
            },
        },
        "safety_point_mass_2d_narrow": {
            "CSPDL": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=30",
                "model": "PPO_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "gray",
            },
            "CSPDL2": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=30",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "tan",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL_D=30",
                "model": "PPO_DELTA=30.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "blue",
            },
            "SPDL2": {
                "algorithm": "self_paced",
                "label": "SPDL2_D=30",
                "model": "PPOLag_DELTA=30.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "green",
            },
            "DEF_Lag": {
                "algorithm": "default",
                "label": "DEF_Lag",
                "model": "PPOLag",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "PPO",
                "color": "magenta",
            },
        },
    }

    settings = {
        "safety_door_2d_narrow":{
            "plot_success": True,
            "num_iters": 300,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Ave. return',
                    "ylim": [-5., 200.],
                },
                1: {
                    "ylabel": 'Ave. cum. cost',
                    "ylim": [-5.0, 200.],
                },
                2: {
                    "ylabel": 'Ave. succ. rate',
                    "ylim": [-0.1, 1.1],
                },
            },
        },
        "safety_cartpole_2d_narrow":{
            "plot_success": False,
            "num_iters": 200,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 6),
            "bbox_to_anchor": (.5, 1.1),
            "subplot_settings": {
                0: {
                    "ylabel": 'Ave. return',
                    "ylim": [-10., 100.],
                },
                1: {
                    "ylabel": 'Ave. cum. cost',
                    "ylim": [-10.0, 100.],
                },
            },
        },
        "safety_point_mass_2d_narrow":{
            "plot_success": True,
            "num_iters": 150,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Ave. return',
                    "ylim": [-10., 80.],
                },
                1: {
                    "ylabel": 'Ave. cum. cost',
                    "ylim": [-5.0, 90.],
                },
                2: {
                    "ylabel": 'Ave. succ. rate',
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
        )

if __name__ == "__main__":
    main()
