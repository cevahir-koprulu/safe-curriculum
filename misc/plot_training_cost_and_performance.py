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

def get_results(base_dir, seeds, iterations, cost_th=0., from_traces=False, use_mean_and_std=False):
    ret_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    cost_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    succ_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    acc_ex_cost_dict = {"mid": [], "qlow": [], "qhigh": [], "min": [], "max": []}
    acc_ex_costs = {seed: 0. for seed in seeds}
    max_iter = 0
    final_rets = None
    final_costs = None
    final_succs = None
    for iteration in iterations:
        rets = []
        costs = []
        succs = []
        for seed in seeds:
            if from_traces:
                perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "context_trace.pkl")
                if os.path.exists(perf_file):
                    if iteration == 0:
                        rets.append(0.)
                        costs.append(0.)
                        succs.append(0.)
                        continue
                    max_iter = max(max_iter, iteration)
                    with open(perf_file, "rb") as f:
                        _rew, disc_rew, _cost, disc_cost, succ, step_len, context_trace = pickle.load(f)
                    rets.append(np.mean(disc_rew))
                    costs.append(np.mean(disc_cost))
                    succs.append(np.mean(succ))
                    # succs.append(np.mean(rew)) # works for maze_3d, otherwise meaningless
                    acc_ex_costs[seed] += max(costs[-1]-cost_th, 0.)
            else:
                perf_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "performance_training.npy")
                if os.path.exists(perf_file):
                    max_iter = max(max_iter, iteration)
                    results = np.load(perf_file)
                    disc_rewards = results[:, 1]
                    cost = results[:, -1]
                    success = results[:, -2]
                    rets.append(np.mean(disc_rewards))
                    costs.append(np.mean(cost))
                    succs.append(np.mean(success))
                    acc_ex_costs[seed] += max(np.mean(cost)-cost_th, 0.)
            # else:
            #     acc_ex_costs.pop(seed, None)
        if len(rets) > 0:
            final_rets, final_costs, final_succs = rets, costs, succs
            update_results(ret_dict, np.array(rets), use_mean_and_std)
            update_results(cost_dict, np.array(costs), use_mean_and_std)
            update_results(succ_dict, np.array(succs), use_mean_and_std)
            update_results(acc_ex_cost_dict, np.array(list(acc_ex_costs.values())), use_mean_and_std)
    print(final_rets, final_costs, final_succs)
    print(f"succ_dict: {succ_dict['mid']}")
    return ret_dict, cost_dict, succ_dict, acc_ex_cost_dict, max_iter

def plot_results(base_log_dir, num_updates_per_iteration, seeds, env, setting, algorithms, figname_extra, from_traces, use_mean_and_std):
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
    final_iterations_step = iterations*steps_per_iter

    fig, axes = plt.subplots(3+int(plot_success),1, figsize=figsize, constrained_layout=True)
    alg_exp_mid = {}
    plt.suptitle("Evaluation wrt distributions in the curriculum")

    for cur_algo in algorithms:
        iterations_step = iterations*steps_per_iter

        algorithm = algorithms[cur_algo]["algorithm"]
        label = algorithms[cur_algo]["label"]
        model = algorithms[cur_algo]["model"]
        color = algorithms[cur_algo]["color"]
        print(algorithm)

        base_dir = os.path.join(base_log_dir, env, algorithm, model)
        print(base_dir)
        ret, cost, succ, acc_cost, max_iter = get_results(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
            cost_th=setting["cost_threshold"],
            from_traces=from_traces,
            use_mean_and_std=use_mean_and_std,
        )
        iterations_step = iterations_step[:(max_iter//num_updates_per_iteration+1)]

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

        expected_acc_ex_cost_mid = acc_cost["mid"]
        expected_acc_ex_cost_qlow = acc_cost["qlow"]
        expected_acc_ex_cost_qhigh = acc_cost["qhigh"]
        expected_acc_ex_cost_min = acc_cost["min"]
        expected_acc_ex_cost_max = acc_cost["max"]

        alg_exp_mid[cur_algo] = expected_return_mid[-1]

        axes[0].plot(iterations_step, expected_return_mid, color=color, linewidth=2.0, label=f"{label}",marker=".")
        axes[0].fill_between(iterations_step, expected_return_qlow, expected_return_qhigh, color=color, alpha=0.24)
        axes[0].fill_between(iterations_step, expected_return_min, expected_return_max, color=color, alpha=0.2)
        axes[1].plot(iterations_step, expected_cum_cost_mid, color=color, linewidth=2.0, marker=".")
        axes[1].fill_between(iterations_step, expected_cum_cost_qlow, expected_cum_cost_qhigh, color=color, alpha=0.2)
        axes[1].fill_between(iterations_step, expected_cum_cost_min, expected_cum_cost_max, color=color, alpha=0.4)
        axes[2].plot(iterations_step, expected_acc_ex_cost_mid, color=color, linewidth=2.0, marker=".")
        axes[2].fill_between(iterations_step, expected_acc_ex_cost_qlow, expected_acc_ex_cost_qhigh, color=color, alpha=0.2)
        axes[2].fill_between(iterations_step, expected_acc_ex_cost_min, expected_acc_ex_cost_max, color=color, alpha=0.4)
        if plot_success:            
            expected_success_mid = succ["mid"]
            expected_success_qlow = succ["qlow"]
            expected_success_qhigh = succ["qhigh"]
            expected_success_min = succ["min"]
            expected_success_max = succ["max"]
            axes[-1].plot(iterations_step, expected_success_mid, color=color, linewidth=2.0, marker=".")
            axes[-1].fill_between(iterations_step, expected_success_qlow, expected_success_qhigh, color=color, alpha=0.4)
            axes[-1].fill_between(iterations_step, expected_success_min, expected_success_max, color=color, alpha=0.2)

    for i, ax in enumerate(axes):
        ax.ticklabel_format(axis='x', style='sci', scilimits=(5, 6), useMathText=True)
        # if i != len(axes) - 1:
        #     ax.set_xticks([])
        # else:
        #     ax.set_xlabel("Number of environment interactions")
        if i == len(axes)-1:
            ax.set_xlabel("Number of environment interactions")
        ax.set_xlim([final_iterations_step[0], final_iterations_step[-1]])
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

    figname_extra += "_from_traces" if from_traces else ""
    figname_extra += "_mean_and_std" if use_mean_and_std else ""
    figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"{env}_{figname}{figname_extra}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500, bbox_inches='tight', bbox_extra_artists=(lgd,))

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    num_updates_per_iteration = 10
    seeds = [str(i) for i in range(1, 6)]
    env = "safety_door_2d_narrow"
    figname_extra = "_D=25_DCS=2.5_training_s1-5"
    # env = "safety_maze_3d"
    # figname_extra = "_D=0.6_MEPS=1.25_DCS=0_training_s1-10_spc0.25"
    from_traces = True
    use_mean_and_std = True
    algorithms = {
        "safety_door_2d_narrow": {
            "CCURROTL_DCT=2.0_MEPS=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_DCT=2.0_MEPS=0.5",
                "model": "PPOLag_DELTA_CS=2.5_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=2.0_METRIC_EPS=0.5_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "CCURROTL_DCT=2.0_MEPS=0.75": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_DCT=2.0_MEPS=0.75",
                "model": "PPOLag_DELTA_CS=2.5_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=2.0_METRIC_EPS=0.75_RAS=10",
                "color": "red",
                "cmap": "Reds",
            },
            "CCURROTL_DCT=2.5_MEPS=0.5": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_DCT=2.5_MEPS=0.5",
                "model": "PPOLag_DELTA_CS=2.5_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=2.5_METRIC_EPS=0.5_RAS=10",
                "color": "green",
                "cmap": "Greens",
            },
            # "CURROTL_PEN_COEFT=0.0": {
            #     "algorithm": "wasserstein",
            #     "label": "CURROTL_PEN_COEFT=0.0",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=0.0",
            #     "color": "red",
            #     "cmap": "Reds",
            # },
            # "CURROTL_PEN_COEFT=1.0": {
            #     "algorithm": "wasserstein",
            #     "label": "CURROTL_PEN_COEFT=1.0",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=1.0",
            #     "color": "green",
            #     "cmap": "Greens",
            # },
            # "CCURROTL_DCT=1.5": {
            #     "algorithm": "constrained_wasserstein",
            #     "label": "CCURROTL_DCT=1.5",
            #     "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=1.5_METRIC_EPS=0.5_RAS=10",
            #     "color": "blue",
            #     "cmap": "Blues",
            # },
            # "CURROT4CostL_DCT=1.5": {
            #     "algorithm": "wasserstein4cost",
            #     "label": "CURROT4CostL_DCT=1.5",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.5_METRIC_EPS=0.5",
            #     "color": "purple",
            #     "cmap": "Purples",
            # },
        },
        "safety_maze_3d": {
            "CCURROTL_DCT=0.25": {
                "algorithm": "constrained_wasserstein",
                "label": "CCURROTL_DCT=0.5",
                "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=0.25_METRIC_EPS=1.25_RAS=10",
                "color": "blue",
                "cmap": "Blues",
            },
            "CURROTL": {
                "algorithm": "wasserstein",
                "label": "CURROTL",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=0.0",
                "color": "red",
                "cmap": "Reds",
            },
            "CURROTL_PENCOEFT=1": {
                "algorithm": "wasserstein",
                "label": "CURROTL_PENCOEFT=1",
                "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=1.0",
                "color": "green",
                "cmap": "Greens",
            },
            # "CURROT4CostL_DCT=0.25": {
            #     "algorithm": "wasserstein4cost",
            #     "label": "CURROT4CostL_DCT=0.25",
            #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=0.25_METRIC_EPS=1.25",
            #     "color": "purple",
            #     "cmap": "Purples",
            # },
        },
    }

    settings = {
        "safety_door_2d_narrow":{
            "cost_threshold": 2.5,
            "plot_success": True,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Ave. return',
                    "ylim": [-5., 70.],
                },
                1: {
                    "ylabel": 'Ave. cum. cost',
                    "ylim": [-5.0, 20.],
                },
                2: {
                    "ylabel": 'Ave. acc. ex. cost',
                    "ylim": [-5.0, 100.],
                },
                3: {
                    "ylabel": 'Ave. succ. rate',
                    "ylim": [-0.1, 1.1],
                },
            },
        },
        "safety_maze_3d":{
            "cost_threshold": 0.,
            "plot_success": True,
            "num_iters": 500,
            "steps_per_iter": 4000,
            "fontsize": 16,
            "figsize": (10, 10),
            "bbox_to_anchor": (.5, 1.05),
            "subplot_settings": {
                0: {
                    "ylabel": 'Ave. return',
                    "ylim": [-100., 10.],
                },
                1: {
                    "ylabel": 'Ave. cum. cost',
                    "ylim": [-5.0, 10.],
                },
                2: {
                    "ylabel": 'Ave. acc. ex. cost',
                    "ylim": [-5.0, 50.],
                },
                3: {
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
        from_traces=from_traces,
        use_mean_and_std=use_mean_and_std,
        )

if __name__ == "__main__":
    main()
