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

def load_success_buffer(base_dir, seeds, iterations, constrained=True):
    all_info = {}
    for seed in seeds:
        seed_info = {'delta_reached': [],
                     'delta_c_reached': [],
                     'min_ret': [],
                     'max_cost': [],}
        for iteration in iterations:
            if constrained:
                perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "teacher_constrained_success_buffer.pkl")
                if os.path.exists(perf_file):
                    with open(perf_file, "rb") as f:
                        delta, delta_c, max_size, min_stds, delta_stds, contexts, returns, \
                            costs, delta_reached, delta_c_reached, min_ret, max_cost, data = pickle.load(f)
                        seed_info['delta_reached'].append(delta_reached*1.0)
                        seed_info['delta_c_reached'].append(delta_c_reached*1.0)
                        seed_info['min_ret'].append(min_ret)
                        seed_info['max_cost'].append(max_cost)
                else:
                    break
            else:
                perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "teacher_success_buffer.pkl")
                if os.path.exists(perf_file):
                    with open(perf_file, "rb") as f:
                        delta, max_size, min_stds, delta_stds, contexts, returns, delta_reached, min_ret, data = pickle.load(f)
                        seed_info['delta_reached'].append(delta_reached*1.0)
                        seed_info['min_ret'].append(min_ret)
                else:
                    break
        if seed_info is not None:
            all_info[seed] = seed_info
    return all_info

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    context_dim = setting["context_dim"]
    num_iters = setting["num_iters"]
    steps_per_iter = setting["steps_per_iter"]
    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]
    subplot_settings = setting["subplot_settings"]
    iterations = np.arange(num_updates_per_iteration, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(4, len(seeds), figsize=figsize, constrained_layout=True)
    infokey_to_axis = {
        'delta_reached': 0,
        'delta_c_reached': 1,
        'min_ret': 2,
        'max_cost': 3,
    }
    infokey_to_marker = {
        'delta_reached': 'o',
        'delta_c_reached': 'o',
        'min_ret': 'o',
        'max_cost': 'o',
    }
    plt.suptitle("Progression of sampled contexts during training")

    x_axis_splitter = np.arange(0.0, 4.0, 4/(len(algorithms)*len(seeds)))*steps_per_iter
    for cur_algo_i, cur_algo in enumerate(algorithms):
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        all_info = load_success_buffer(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
            constrained=True if "constrained" in algorithm else False,
        )
        if len(all_info) == 0: continue
        for seed_i in range(len(seeds)):
            if seeds[seed_i] not in all_info: continue
            seed = seeds[seed_i]
            for infokey in all_info[seed]:
                ax_i = infokey_to_axis[infokey]
                axes[ax_i,seed_i].plot(iterations_step[:len(all_info[seed][infokey])],
                    all_info[seed][infokey], color=color, marker=infokey_to_marker[infokey], alpha=0.5, markersize=3.0)

    for i, infokey in enumerate(infokey_to_axis):
        for seed_i in range(len(seeds)):
            ax = axes[i,seed_i]
            # ax.xaxis.set_major_locator(ticker.FixedLocator(x_axis_splitter))
            # ax.xaxis.set_major_formatter(ticker.FixedFormatter([f"{int(x)}" for x in x_axis_splitter]))
            if i == 0:
                ax.set_title(f"Seed {seeds[seed_i]}")
            if i == len(axes)-1:
                ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
                if seed_i == len(seeds)//2:
                    ax.set_xlabel("Number of environment interactions")
            ax.set_xlim([iterations_step[0], iterations_step[-1]])
            ax.set_ylim(subplot_settings[i]["ylim"])
            if seed_i == 0:
                ax.set_ylabel(subplot_settings[i]["ylabel"])
            ax.grid(True)

    colors = []
    labels = []
    num_alg = len(algorithms)
    for cur_algo in algorithms:
        colors.append(algorithms[cur_algo]["color"])
        labels.append(algorithms[cur_algo]["label"])

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
    seeds = [str(i) for i in range(1, 16)]
    env = "safety_door_2d_narrow"
    figname_extra = "_rExp0.8_lBorder=0.01_slp=0.5_walled_training_progression_ANNEALED"

    algorithms = {
        "safety_door_2d_narrow": {
            "CCURROTL_D=25_DCE=1.25_ATP=0.75_MEPS=0.25": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_D=25_DCE=1.25_ATP=0.75_MEPS=0.25",
                "model": "PPOLag_ATP=0.75_CAS=10_DELTA=25.0_DELTA_C=0.0_DELTA_C_EXT=1.25_METRIC_EPS=0.25_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "CURROTL_D=25_MEPS=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": "CURROTL_D=25_MEPS=0.5",
                "model": "PPOLag_DELTA=25.0_METRIC_EPS=0.25",
                "color": "red",
                "cmap": "Reds",
            },           
        },
    }

    settings = {
        "safety_door_2d_narrow": {
            "context_dim": 2,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Delta Reached',
                    "ylim": [-.1, 1.1],
                },
                1: {
                    "ylabel": 'Delta C Reached',
                    "ylim": [-.1, 1.1],
                },
                2: {
                    "ylabel": 'Min Return',
                    "ylim": [-1, 10.],
                },
                3: {
                    "ylabel": 'Max Cost',
                    "ylim": [0, 100.],
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
