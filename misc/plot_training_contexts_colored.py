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

def update_results(res_dict, res):
    res_dict["mid"].append(np.mean(res, axis=0))
    res_dict["qlow"].append(np.quantile(res, 0.25, axis=0))
    res_dict["qhigh"].append(np.quantile(res, 0.75, axis=0))
    res_dict["min"].append(np.min(res, axis=0))
    res_dict["max"].append(np.max(res, axis=0))

def load_context_traces(base_dir, seeds, iterations):
    all_context_traces = {}
    all_disc_rew = {}
    all_disc_cost = {}
    for iteration in iterations:
        # iter_context_traces = None
        iter_context_traces = {}
        iter_disc_rew = {}
        iter_disc_cost = {}
        for seed in seeds:
            perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "context_trace.pkl")
            if os.path.exists(perf_file):
                with open(perf_file, "rb") as f:
                    rew, disc_rew, cost, disc_cost, context_trace = pickle.load(f)
                    iter_context_traces[seed] = np.array(context_trace)
                    iter_disc_rew[seed] = np.array(disc_rew)
                    iter_disc_cost[seed] = np.array(disc_cost)
        if iter_context_traces is not None:
            all_context_traces[iteration] = iter_context_traces
            all_disc_rew[iteration] = iter_disc_rew
            all_disc_cost[iteration] = iter_disc_cost
    return all_context_traces, all_disc_rew, all_disc_cost

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra, color_type):
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
    return_bounds = setting["return_bounds"]
    cost_bounds = setting["cost_bounds"]
    iterations = np.arange(num_updates_per_iteration, num_iters, num_updates_per_iteration, dtype=int)
    iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(context_dim, len(seeds), figsize=figsize, constrained_layout=True)
    suptitle = f"Progression of sampled contexts during training with " + "return" if color_type == "return" else "cost"
    plt.suptitle(suptitle)

    x_axis_splitter = np.arange(0.0, 4.0, 4/len(algorithms))*steps_per_iter
    for cur_algo_i, cur_algo in enumerate(algorithms):
        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        cmap = algorithms[cur_algo]["cmap"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        context_traces, disc_rew, disc_cost = load_context_traces(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        if len(context_traces) == 0: continue
        for cdim in range(context_dim):
            for key in context_traces:
                for seed_i in range(len(seeds)):
                    seed = seeds[seed_i]
                    if seed not in context_traces[key]: continue
                    idx = int(key/num_updates_per_iteration)
                    c = disc_rew[key][seed] if color_type == "return" else -disc_cost[key][seed]
                    vmin = return_bounds[0] if color_type == "return" else -cost_bounds[1]
                    vmax = return_bounds[1] if color_type == "return" else -cost_bounds[0]
                    axes[cdim, seed_i].scatter(
                        np.ones(context_traces[key][seed][:,cdim].shape[0]
                                )*iterations_step[idx-1] + x_axis_splitter[cur_algo_i],
                        context_traces[key][seed][:,cdim], c=c, alpha=0.5, s=1.0, 
                        cmap=cmap, vmin=vmin, vmax=vmax)

    for cdim in range(context_dim):
        for seed_i in range(len(seeds)):
            ax = axes[cdim,seed_i]
            if cdim == 0:
                ax.set_title(f"Seed {seeds[seed_i]}")
            if cdim == context_dim-1:
                ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
                if seed_i == len(seeds)//2:
                    ax.set_xlabel("Number of environment interactions")
            ax.set_xlim([iterations_step[0], iterations_step[-1]])
            ax.set_ylim(subplot_settings[cdim]["ylim"])
            if seed_i == 0:
                ax.set_ylabel(subplot_settings[cdim]["ylabel"])
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

    figname_extra += f"_{color_type}"
    figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{figname_extra}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_updates_per_iteration = 5
    seeds = [str(i) for i in range(1, 6)]
    env = "safety_door_2d_narrow"
    figname_extra = "_DCE=1.25_MEPS=0.25_C-R-AS=10_rExp0.8_lBorder=0.01_slp=0.5_walled_training_ANNEALED"
    # color_type = "return"
    color_type = "cost"

    algorithms = {
        "safety_door_2d_narrow": {
            "CCURROTL_D=20_ATP=0.75": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_D=20_ATP=0.75",
                "model": "PPOLag_ATP=0.75_CAS=10_DELTA=20.0_DELTA_C=0.0_DELTA_C_EXT=1.25_METRIC_EPS=0.25_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "CCURROTL_D=30_ATP=0.75": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_D=30_ATP=0.75",
                "model": "PPOLag_ATP=0.75_CAS=10_DELTA=30.0_DELTA_C=0.0_DELTA_C_EXT=1.25_METRIC_EPS=0.25_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            "CCURROTL_D=20_ATP=1.0": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_D=20_ATP=1.0",
                "model": "PPOLag_ATP=1.0_CAS=10_DELTA=20.0_DELTA_C=0.0_DELTA_C_EXT=1.25_METRIC_EPS=0.25_RAS=10",
                "color": "red",
                "cmap": "Reds",
            },               
            "CCURROTL_D=30_ATP=1.0": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_D=30_ATP=1.0",
                "model": "PPOLag_ATP=1.0_CAS=10_DELTA=30.0_DELTA_C=0.0_DELTA_C_EXT=1.25_METRIC_EPS=0.25_RAS=10",
                "color": "purple",
                "cmap": "Purples",
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
                    "ylabel": 'Door Position',
                    "ylim": [-4., 4.],
                },
                1: {
                    "ylabel": 'Door Width',
                    "ylim": [0., 8.],
                },
            },
            "return_bounds": [25., 70.],
            "cost_bounds": [0., 10.],
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
        color_type=color_type,
        )

if __name__ == "__main__":
    main()
