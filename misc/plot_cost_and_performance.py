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

def get_results(base_dir, iterations):
    expected_return = []
    expected_success = []
    expected_cum_cost = []
    for iteration in iterations:
        perf_file = os.path.join(base_dir, f"iteration-{iteration}", "performance.npy")
        if os.path.exists(perf_file):
            results = np.load(perf_file)
            disc_rewards = results[:, 1]
            cost = results[:, -1]
            success = results[:, -2]
            expected_return.append(np.mean(disc_rewards))
            expected_cum_cost.append(np.mean(cost))
            expected_success.append(np.mean(success))
        else:
            print(f"No evaluation data found: {perf_file}")
            expected_return = []
            expected_cum_cost = []
            expected_success = []
            break
    return expected_return, expected_cum_cost, expected_success

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    subplot_settings = setting["subplot_settings"]
    iterations = np.arange(0, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(3,1, figsize=figsize, constrained_layout=True)
    alg_exp_mid = {}
    plt.suptitle("Evaluation wrt target distribution")

    for cur_algo in algorithms:
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        expected_return = []
        expected_cum_cost = []
        expected_success = []
        for seed in seeds:
            base_dir = os.path.join(base_log_dir, env, algorithm, model, f"seed-{seed}")
            print(base_dir)
            expected_return_seed, expected_cum_cost_seed, expected_success_seed = get_results(
                base_dir=base_dir,
                iterations=iterations,
            )
            if len(expected_return_seed) == 0:
                continue

            expected_return.append(expected_return_seed)
            expected_cum_cost.append(expected_cum_cost_seed)
            expected_success.append(expected_success_seed)

        expected_return = np.array(expected_return)
        # expected_return_mid = np.median(expected_return, axis=0)
        # expected_return_qlow = np.quantile(expected_return, 0.25, axis=0)
        # expected_return_qhigh = np.quantile(expected_return, 0.75, axis=0)
        expected_return_mid = np.mean(expected_return, axis=0)
        expected_return_qlow = np.min(expected_return, axis=0)
        expected_return_qhigh = np.max(expected_return, axis=0)

        expected_cum_cost = np.array(expected_cum_cost)
        # expected_cum_cost_mid = np.median(expected_cum_cost, axis=0)
        # expected_cum_cost_qlow = np.quantile(expected_cum_cost, 0.25, axis=0)
        # expected_cum_cost_qhigh = np.quantile(expected_cum_cost, 0.75, axis=0)
        expected_cum_cost_mid = np.mean(expected_cum_cost, axis=0)
        expected_cum_cost_qlow = np.min(expected_cum_cost, axis=0)
        expected_cum_cost_qhigh = np.max(expected_cum_cost, axis=0)

        expected_success = np.array(expected_success)
        # expected_success_mid = np.median(expected_success, axis=0)
        # expected_success_qlow = np.quantile(expected_success, 0.25, axis=0)
        # expected_success_qhigh = np.quantile(expected_success, 0.75, axis=0)
        expected_success_mid = np.mean(expected_success, axis=0)
        expected_success_qlow = np.min(expected_success, axis=0)
        expected_success_qhigh = np.max(expected_success, axis=0)


        alg_exp_mid[cur_algo] = expected_return_mid[-1]

        axes[0].plot(iterations_step, expected_return_mid, color=color, linewidth=2.0, label=f"{label}",marker=".")
        axes[0].fill_between(iterations_step, expected_return_qlow, expected_return_qhigh, color=color, alpha=0.4)
        axes[1].plot(iterations_step, expected_cum_cost_mid, color=color, linewidth=2.0, marker=".")
        axes[1].fill_between(iterations_step, expected_cum_cost_qlow, expected_cum_cost_qhigh, color=color, alpha=0.4)
        axes[2].plot(iterations_step, expected_success_mid, color=color, linewidth=2.0, marker=".")
        axes[2].fill_between(iterations_step, expected_success_qlow, expected_success_qhigh, color=color, alpha=0.4)

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
    env = "safety_point_mass_2d_narrow"
    figname_extra = "_KL_EPS=1.0_D30"

    algorithms = {
        "safety_point_mass_2d_narrow": {
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL",
                "model": "ppo_DELTA=30.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "blue",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "ppo",
                "color": "magenta",
            },
        },
    }

    settings = {
        "safety_point_mass_2d_narrow":{
            "num_iters": 1000,
            "steps_per_iter": 4000,
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
