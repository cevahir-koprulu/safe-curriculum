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
    for iteration in iterations:
        # iter_context_traces = None
        iter_context_traces = {}
        for seed in seeds:
            perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "context_trace.pkl")
            if os.path.exists(perf_file):
                with open(perf_file, "rb") as f:
                    context_trace = pickle.load(f)[-1]
                    # if iter_context_traces is None:
                    #     iter_context_traces = context_trace
                    # else:
                    #     iter_context_traces = np.vstack((iter_context_traces, context_trace))
                    iter_context_traces[seed] = np.array(context_trace)
        if iter_context_traces is not None:
            all_context_traces[iteration] = iter_context_traces
    return all_context_traces

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

    fig, axes = plt.subplots(context_dim,1, figsize=figsize, constrained_layout=True)
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
        context_traces = load_context_traces(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        if len(context_traces) == 0: continue
        for ax_i in range(context_dim):
            for key in context_traces:
                for seed in context_traces[key]:
                    idx = int(key/num_updates_per_iteration)
                    splitter_i = int(cur_algo_i*len(seeds) + seeds.index(seed))
                    axes[ax_i].scatter(
                        np.ones(context_traces[key][seed][:,ax_i].shape[0])*iterations_step[idx-1] + x_axis_splitter[splitter_i],
                        context_traces[key][seed][:,ax_i], color=color, alpha=0.5, s=1.0/len(seeds))

    for i, ax in enumerate(axes):
        ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        if i == len(axes)-1:
            ax.set_xlabel("Number of environment interactions")
        ax.set_xlim([iterations_step[0], iterations_step[-1]])
        ax.set_ylim(subplot_settings[i]["ylim"])
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
    seeds = [str(i) for i in range(1, 4)]
    env = "safety_door_2d_narrow"
    figname_extra = "_D=30_rExp0.2_lBorder_training_contexts"
    # env = "safety_cartpole_2d_narrow"
    # figname_extra = "_KL_EPS=1.0_im=0.7xth_0.2xth_training_contexts"

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
        },
        "safety_point_mass_2d_narrow": {
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
        },
    }

    settings = {
        "safety_door_2d_narrow": {
            "context_dim": 2,
            "num_iters": 300,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 10),
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
        },
        "safety_cartpole_2d_narrow": {
            "context_dim": 2,
            "num_iters": 200,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Left Bar Position',
                    "ylim": [-3., 0.],
                },
                1: {
                    "ylabel": 'Right Bar Position',
                    "ylim": [0., 3.],
                },
            },
        },
        "safety_point_mass_2d_narrow":{
            "context_dim": 2,
            "num_iters": 150,
            "steps_per_iter": 2000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Left Lava Position',
                    "ylim": [-4., 4.],
                },
                1: {
                    "ylabel": 'Right Lava Position',
                    "ylim": [-4., 4.],
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
