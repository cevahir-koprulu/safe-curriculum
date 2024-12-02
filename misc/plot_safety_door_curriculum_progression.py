import os
import sys
import math
import pickle
import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker, colorbar, colors
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.experiments.safety_door_2d_experiment import SafetyDoor2DExperiment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FONT_SIZE = 12 # 20

def get_contexts(base_dir, seeds, iterations):
    contexts = {}
    for iteration in iterations:
        contexts[iteration] = {}
        for seed in seeds:
            context_trace_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "context_trace.pkl")
            if os.path.exists(context_trace_file):
                with open(context_trace_file, "rb") as f:
                    _rew, disc_rew, _cost, disc_cost, succ, step_len, context_trace = pickle.load(f)
                contexts[iteration][seed] = np.array(context_trace)
            else:
                print(f"File {context_trace_file} does not exist")
    return contexts

def sample_contexts():
    return


def get_chosen_context(cs, i):
    chosen_its = {0: "left",
                  1: "top",
                  2: "right",
                  3: "bottom"}
    if chosen_its[i] == "left":
        c = cs[np.argmin(cs[:, 0]), :]
    elif chosen_its[i] == "top":
        c = cs[np.argmax(cs[:, 1]), :]
    elif chosen_its[i] == "right":
        c = cs[np.argmax(cs[:, 1]), :]
    elif chosen_its[i] == "bottom":
        c = cs[np.argmax(cs[:, 0]), :]
    return c

def draw_env(ax, pos, width, context_bounds):
    # Draw a red rectangle above each black line
    w1 = pos - (0.5 * width + 0.5) - context_bounds[0][0] + 1
    w2 = context_bounds[1][0] +1 - (pos + (0.5 + 0.5 * width)) 
    ax.add_patch(Rectangle((context_bounds[0][0]-1, -1.1), w1, 2, color="red", alpha=0.7))
    ax.add_patch(Rectangle((pos + (0.5 + 0.5 * width), -1.1), w2, 2, color="red", alpha=0.7))

    ax.plot([context_bounds[0][0]-1, pos - (0.5 * width + 0.5)], [-1.1, -1.1], linewidth=3, color="black")
    ax.plot([pos + (0.5 + 0.5 * width), context_bounds[1][0]+1], [-1.1, -1.1], linewidth=3, color="black")

    ax.scatter([0.], [3.], s=50, color="black")
    ax.plot([-0.25, 0.25], [-3.25, -2.75], linewidth=5, color="green")
    ax.plot([-0.25, 0.25], [-2.75, -3.25], linewidth=5, color="green")
    ax.set_xlim([context_bounds[0][0], context_bounds[1][0]])
    ax.set_ylim([-4, 4])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params('both', length=0, width=0, which='major')

def plot_curriculum_progression(base_log_dir, seeds, algorithm, iterations, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = FONT_SIZE

    exp = SafetyDoor2DExperiment(base_log_dir="logs", curriculum_name="default", 
                                    learner_name="PPOLag", parameters={"TARGET_TYPE": "narrow"},
                                    seed=1, device="cpu")
    context_bounds = [exp.LOWER_CONTEXT_BOUNDS*2, exp.UPPER_CONTEXT_BOUNDS*2]

    curriculum_colors = plt.get_cmap('viridis')(np.linspace(0., 1.0, len(iterations)))
    cmap_ = colors.ListedColormap(curriculum_colors)
    # pm_boxes_arrows = [
    # ([-0.03, 0.15, 0.21, 0.3], [-6.5, 4.3]),
    # ([-0.03, 0.6, 0.21, 0.3], [-6.5, 10.75]),
    # ([0.82, 0.6, 0.21, 0.3], [6.85, 8.1]),
    # ([0.82, 0.15, 0.21, 0.3], [6.6, 2]),
    # ]
    # pm_boxes_arrows = [
    # ([-0.0, 0.15, 0.21, 0.3], [-8.25, 4.]),
    # ([-0.0, 0.6, 0.21, 0.3], [-8.2, 13.]),
    # ([0.79, 0.6, 0.21, 0.3], [8.3, 13.]),
    # ([0.79, 0.15, 0.21, 0.3], [8.2, 4.]),
    # ]
    pm_boxes_arrows = [
    ([0.04, 0.3, 0.15, 0.2], [-8.1, 4.5]),
    ([0.04, 0.6, 0.15, 0.2], [-8.1, 13.]),
    ([0.705, 0.6, 0.15, 0.2], [8.1, 11.]),
    ([0.705, 0.3, 0.15, 0.2], [8.1, 2.]),
    ]
    for algo_dict in algorithm:
        algorithm_name  = algo_dict["algorithm"]
        label = algo_dict["label"]
        model = algo_dict["model"]
        base_dir = os.path.join(base_log_dir, "safety_door_2d_narrow", algorithm_name, model)
        print(base_dir)
        contexts = get_contexts(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        f = plt.figure(figsize=(10.0, 7.0))
        # ax = plt.Axes(f, [0.178, 0.08, 0.6435, 0.85])
        ax = plt.Axes(f, [0.16, 0.18, 0.57, 0.7])
        f.add_axes(ax)
        ax.spines['bottom'].set_position(('data', 1.0))
        ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks([context_bounds[0][0]+2, 0, context_bounds[1][0]-2])
        ax.set_xticklabels([f'{int(context_bounds[0][0]//2+1)}', '0', f'{int(context_bounds[1][0]//2-1)}'], 
                           fontsize=int(FONT_SIZE*1.5))
        ax.set_xticks(np.arange(int(context_bounds[0][0]), int(context_bounds[1][0]+1)))

        ax.set_yticks([context_bounds[1][1]-6])
        ax.set_yticklabels([f'{int(context_bounds[1][1]//2-3)}'], fontsize=int(FONT_SIZE*1.5))
        ax.set_yticks(np.arange(0., int(context_bounds[1][1]+1)))
        
        ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
        arrow_fmt = dict(markersize=4, color='black', clip_on=False)
        ax.plot((1), (1.0), marker='>', transform=ax.get_yaxis_transform(), **arrow_fmt)
        ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), **arrow_fmt)
        contexts_all = None
        iters = []
        for itx, it in enumerate(iterations):
            contexts_all_ = None
            for seed in seeds:
                if contexts_all_ is None:
                    contexts_all_ = contexts[it][seed]
                else:
                    contexts_all_ = np.vstack((contexts_all_, contexts[it][seed]))
            iters.append(contexts_all_.shape[0])
            if contexts_all is None:
                contexts_all = contexts_all_
            else:
                contexts_all = np.vstack((contexts_all, contexts_all_))
        norm_ = colors.BoundaryNorm(np.arange(-0.5,len(iterations)), len(iterations))
        cax = ax.scatter(contexts_all[:,0]*2, contexts_all[:,1]*2, 
                    c=[i for i, i_num in enumerate(iters) for j in range(i_num)], cmap=cmap_, norm=norm_,
                    # norm=plt.Normalize(0., len(curriculum_colors)+1),
                    s=70)
        ax.annotate("Width", xy=(0.1,16.5), fontsize=int(FONT_SIZE*1.25))
        ax.annotate("Position", xy=(2.,0.22), fontsize=int(FONT_SIZE*1.25))
        for itx, it in enumerate(iterations):
            first_iter, last_iter = 0 if itx == 0 else np.cumsum(iters)[itx-1], np.cumsum(iters)[itx]
            c = get_chosen_context(contexts_all[first_iter:last_iter,:], itx)
            arrow_end = pm_boxes_arrows[itx][1] - c*2
            ax.arrow(c[0]*2, c[1]*2, arrow_end[0], arrow_end[1], head_width=.3, color="r")
            ax_pm = plt.Axes(f, pm_boxes_arrows[itx][0])
            f.add_axes(ax_pm)
            draw_env(ax_pm, c[0], c[1], [context_bounds[0]//2, context_bounds[1]//2])
            ax_pm.set_title(f"Epoch {iterations[itx]}\n"r"$\mathbf{x}=$"+f"({c[0]:.2f}, {c[1]:.2f})")

        cbar = f.colorbar(cax, ax=ax, ticks=[0, 1, 2, 3], norm=norm_, shrink=0.75, anchor=(0.5,2.0),
                          orientation='horizontal', label='Colorbar for epochs')
        cbar.ax.tick_params(length=0)
        cbar.set_ticklabels([str(itx) for itx in iterations ])

        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"safety_door_2d_narrow_curriculum_progressions_{label}_iter={iterations}{figname_extra}.pdf")
        print(figpath)
        plt.savefig(figpath, dpi=500, bbox_inches='tight')

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    # iterations = [5, 30, 100, 450]
    iterations = [10, 50, 150, 300]
    seeds = [str(i) for i in range(1, 11)]
    figname_extra = "_legend"
    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))
    algorithm = [
        # {
        #     "algorithm": "default",
        #     "label": "DEFAULT",
        #     "model": "PPOLag_DELTA_CS=0.0",
        # },
        {
            "algorithm": "constrained_wasserstein",
            "label": "SCG",
            "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=25.0_DELTA_CT=1.5_METRIC_EPS=0.5_RAS=10",
        },
        # {
        #     "algorithm": "wasserstein",
        #     "label": "NaiveSafeCURROT",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=1.0",
        # },
        # {
        #     "algorithm": "wasserstein",
        #     "label": "CURROT",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_METRIC_EPS=0.5_PEN_COEFT=0.0",
        # },
        # {
        #     "algorithm": "wasserstein4cost",
        #     "label": "CURROT4Cost",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.5_METRIC_EPS=0.5",
        # },
        # {
        #     "algorithm": "alp_gmm",
        #     "label": "ALP-GMM",
        #     "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=100_AG_MAX_SIZE=500_AG_P_RAND=0.1",
        # },
        # {
        #     "algorithm": "plr",
        #     "label": "PLR",
        #     "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.45_PLR_REPLAY_RATE=0.85_PLR_RHO=0.15",
        # },
        # {
        #     "algorithm": "self_paced",
        #     "label": "SPDL",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=25.0_DIST_TYPE=gaussian_INIT_VAR=0.5_KL_EPS=0.25_PEN_COEFT=0.0",
        # },
        # {
        #     "algorithm": "goal_gan",
        #     "label": "GoalGAN",
        #     "model": "PPOLag_DELTA_CS=0.0_GG_FIT_RATE=200_GG_NOISE_LEVEL=0.1_GG_P_OLD=0.2",
        # },
    ]

    plot_curriculum_progression(
        base_log_dir=base_log_dir,
        seeds=seeds,
        algorithm=algorithm,
        iterations=iterations,
        figname_extra=figname_extra,
        )

if __name__ == "__main__":
    main()
