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

def load_context_traces(base_dir, seeds, iterations):
    all_context_traces = {}
    all_disc_rew = {}
    all_disc_cost = {}
    max_iter = 0
    for iteration in iterations:
        # iter_context_traces = None
        iter_context_traces = {}
        iter_disc_rew = {}
        iter_disc_cost = {}
        for seed in seeds:
            perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "context_trace.pkl")
            if os.path.exists(perf_file):
                max_iter = max(max_iter, iteration)
                with open(perf_file, "rb") as f:
                    rew, disc_rew, cost, disc_cost, succ, step_len, context_trace = pickle.load(f)
                    iter_context_traces[seed] = np.array(context_trace)
                    iter_disc_rew[seed] = np.array(disc_rew)
                    iter_disc_cost[seed] = np.array(disc_cost)
        if iter_context_traces is not None:
            all_context_traces[iteration] = iter_context_traces
            all_disc_rew[iteration] = iter_disc_rew
            all_disc_cost[iteration] = iter_disc_cost
    return all_context_traces, all_disc_rew, all_disc_cost, max_iter

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
    final_iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(context_dim, len(seeds), figsize=figsize, constrained_layout=True)
    axes = np.array(axes).reshape(context_dim, len(seeds))
    suptitle = f"Progression of sampled contexts during training with " + "return" if color_type == "return" else "cost"
    plt.suptitle(suptitle)

    x_axis_splitter = np.arange(0.0, 4.0, 4/len(algorithms))*steps_per_iter
    for cur_algo_i, cur_algo in enumerate(algorithms):
        iterations_step = iterations*steps_per_iter

        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        cmap = algorithms[cur_algo]["cmap"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        context_traces, disc_rew, disc_cost, max_iter = load_context_traces(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        iterations_step = iterations_step[:(max_iter//num_updates_per_iteration+1)]

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
            ax.set_xlim([final_iterations_step[0], final_iterations_step[-1]])
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

    # num_updates_per_iteration = 10
    # seeds = [str(i) for i in range(6, 11)]
    # env = "safety_maze_3d"
    # figname_extra = "_DCS=0_training_contexts_s6-10"

    # num_updates_per_iteration = 10    
    # seeds = [str(i) for i in range(6, 11)]
    # env = "safety_goal_3d"
    # figname_extra = "_MEPS=0.5_DCS=0_training_contexts_s1-5_new"

    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 4)]
    # env = "safety_goal_noconflict_3d"
    # figname_extra = "_DCS=0_training_contexts_s1-3"

    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 4)]
    # env = "safety_passage_3d"
    # figname_extra = "_s1-3"

    # num_updates_per_iteration = 5
    # seeds = [str(i) for i in range(1, 6)]
    # env = "safety_passage_push_3d"
    # figname_extra = "_MEPS=0.25_gs=15_bs=128_s1-5_CAR_InnerWallv6.2_SPI=300"

    num_updates_per_iteration = 5
    seeds = [str(i) for i in range(1, 4)]
    env = "safety_doggo_3d"
    figname_extra = "_SCG_RCAS=15_act=relu_SMALLv5_h3x256_LOWCOST"

    # color_type = "return"
    color_type = "cost"

    algorithms = {
        "safety_maze_3d": {
            "SCG": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=0.25_METRIC_EPS=1.25_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "CURROT": {
                "algorithm": "wasserstein",
                "label": "CURROT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=0.0",
                "color": "red",
                "cmap": "Reds",
            },
            "NaiveSafeCURROT": {
                "algorithm": "wasserstein",
                "label": "NaiveSafeCURROT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=1.0",
                "color": "green",
                "cmap": "Greens",
            },
            "CURROT4Cost": {
                "algorithm": "wasserstein4cost",
                "label": "CURROT4Cost",
                "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=0.25_METRIC_EPS=1.25",
                "color": "purple",
                "cmap": "Purples",
            },
        },
        "safety_goal_3d": {
            "SCG_D=0.6_DCT=1": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.6_DCT=1",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.5_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "SCG_D=0.6_DCT=1.5": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.6_DCT=1.5",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.5_METRIC_EPS=0.5_RAS=10",
                "color": "red",
                "cmap": "Reds",
            },
            "SCG_D=0.5_DCT=1": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.5_DCT=1",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.5_DELTA_CT=1.0_METRIC_EPS=0.5_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            # "CURROT_D=0.5_PENCOEFT=1": {
            #     "algorithm": "wasserstein",
            #     "label": "CURROT_D=0.5_PENCOEFT=1",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.5_METRIC_EPS=0.5_PEN_COEFT=1.0",
            #     "color": "red",
            #     "cmap": "Reds",
            # },
            # "CURROT_D=0.6_PENCOEFT=0": {
            #     "algorithm": "wasserstein",
            #     "label": "CURROT_D=0.6_PENCOEFT=0",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
            #     "color": "green",
            #     "cmap": "Greens",
            # },     
            # "DEFAULT": {
            #     "algorithm": "default",
            #     "label": "DEFAULT",
            #     "model": "PPOLag_DELTA_CS=0.0",
            #     "color": "purple",
            #     "cmap": "Purples",
            # },
        },
        "safety_goal_noconflict_3d": {
            "SCG_MEPS=0.4": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_MEPS=0.4",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.4_RAS=10",
                "color": "red",
                "cmap": "Reds",
            },
            "SCG_MEPS=0.3": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_MEPS=0.3",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.3_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            "CRT": {
                "algorithm": "wasserstein",
                "label": "CRT",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.5_PEN_COEFT=0.0",
                "color": "blue",
                "cmap": "Blues",
            },
        },
        "safety_passage_3d": {
            "SCG_MEPS=0.1": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_MEPS=0.1",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.1_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            "SCG_MEPS=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_MEPS=0.5",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.5_RAS=10",
                "color": "magenta",
                "cmap": "Purples",
            },
        },
        "safety_passage_push_3d": {
            "SCG_D=0.6": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.6",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_RAS=10",
                "color": "red",
                "cmap": "Reds",
            },
            "SCG_D=0.55": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.55",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.55_DELTA_CT=1.0_METRIC_EPS=0.25_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "SCG_D=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": "SCG_D=0.5",
                "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.5_DELTA_CT=1.0_METRIC_EPS=0.25_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            # "SCG_DCT=1.0": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG_DCT=1.0",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_RAS=10",
            #     "color": "red",
            #     "cmap": "Reds",
            # },
            # "SCG_DCT=0.75": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG_DCT=0.75",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=0.75_METRIC_EPS=0.25_RAS=10",
            #     "color": "blue",
            #     "cmap": "Blues",
            # },
            # "SCG_DCT=0.5": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG_DCT=0.5",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=0.5_METRIC_EPS=0.25_RAS=10",
            #     "color": "green",
            #     "cmap": "Greens",
            # },
            # "SCG": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "SCG",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_RAS=10",
            #     "color": "red",
            #     "cmap": "Reds",
            # },
            # "NSCRT": {
            #     "algorithm": "wasserstein",
            #     "label": "NSCRT",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=1.0",
            #     "color": "blue",
            #     "cmap": "Blues",
            # },
            # "CRT": {
            #     "algorithm": "wasserstein",
            #     "label": "CRT",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=0.0",
            #     "color": "green",
            #     "cmap": "Greens",
            # },
            # "CRT4Cost": {
            #     "algorithm": "wasserstein4cost",
            #     "label": "CRT4Cost",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.25",
            #     "color": "magenta",
            #     "cmap": "Purples",
            # },
        },
    }

    settings = {
        "safety_maze_3d": {
            "context_dim": 3,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Goal Position X',
                    "ylim": [-9., 9.],
                },
                1: {
                    "ylabel": 'Goal Position Y',
                    "ylim": [-9., 9.],
                },
                2: {
                    "ylabel": 'Tolerance',
                    "ylim": [0., 18.],
                },
            },
            "return_bounds": [25., 70.],
            "cost_bounds": [0., 10.],
        },
        "safety_goal_3d": {
            "context_dim": 3,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Goal Position X',
                    "ylim": [-1.5, 1.5],
                },
                1: {
                    "ylabel": 'Goal Position Y',
                    "ylim": [-1.5, 1.5],
                },
                2: {
                    "ylabel": 'Tolerance',
                    "ylim": [0.25, 1.],
                },
            },
            "return_bounds": [25., 70.],
            "cost_bounds": [0., 10.],
        },
        "safety_goal_noconflict_3d": {
            "context_dim": 3,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Goal Position X',
                    "ylim": [-1.5, 1.5],
                },
                1: {
                    "ylabel": 'Goal Position Y',
                    "ylim": [-1.5, 1.5],
                },
                2: {
                    "ylabel": 'Tolerance',
                    "ylim": [0.25, 1.],
                },
            },
            "return_bounds": [25., 70.],
            "cost_bounds": [0., 10.],
        },
        "safety_passage_3d": {
            "context_dim": 3,
            "num_iters": 150,
            "steps_per_iter": 10000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Goal Position X',
                    "ylim": [-2, 2],
                },
                1: {
                    "ylabel": 'Goal Position Y',
                    "ylim": [-1.5, 1.5],
                },
                2: {
                    "ylabel": 'Tolerance',
                    "ylim": [0.25, 1.],
                },
            },
            "return_bounds": [25., 70.],
            "cost_bounds": [0., 10.],
        },
        "safety_passage_push_3d": {
            "context_dim": 3,
            "num_iters": 300,
            "steps_per_iter": 10000,
            "fontsize": 16,
            "figsize": (30, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Goal Position X',
                    "ylim": [-2, 2],
                },
                1: {
                    "ylabel": 'Goal Position Y',
                    "ylim": [-2, 2],
                },
                2: {
                    "ylabel": 'Tolerance',
                    "ylim": [0.25, .75],
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
