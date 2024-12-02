import os
import sys
import math
import pickle
import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.experiments.safety_passage_push_3d_experiment import SafetyPassagePush3DExperiment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FONT_SIZE = 8
TICK_SIZE = 6

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

def get_circle(x, y, r=0.25, color='slateblue'):
    return Circle((x, y), r, facecolor=color, linewidth=3, alpha=1.0)

def plot_curriculum_progression(base_log_dir, seeds, algorithm, iterations, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = FONT_SIZE

    exp = SafetyPassagePush3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                    learner_name="PPOLag", parameters={},
                                    seed=1, device="cpu")
    norm = matplotlib.colors.Normalize(vmin=exp.LOWER_CONTEXT_BOUNDS[-1], vmax=exp.UPPER_CONTEXT_BOUNDS[-1], clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")

    for algo_dict in algorithm:
        algorithm_name  = algo_dict["algorithm"]
        label = algo_dict["label"]
        model = algo_dict["model"]
        base_dir = os.path.join(base_log_dir, "safety_passage_push_3d", algorithm_name, model)
        print(base_dir)
        contexts = get_contexts(
            base_dir=base_dir,
            seeds=seeds,
            iterations=iterations,
        )
        f = plt.figure(figsize=(1.5, 1.45))
        cax = f.add_axes([0.79, 0.05, 0.04, 0.9])
        contexts_all = {}
        for i, iteration in enumerate(iterations):
            contexts_all = None
            for seed in seeds:
                # print(contexts[iteration][seed].shape)
                if seed not in contexts[iteration]:
                    continue
                if contexts_all is None:
                    contexts_all = contexts[iteration][seed]
                else:
                    contexts_all = np.vstack((contexts_all, contexts[iteration][seed]))
            ax = plt.Axes(f, [0.39 * (i % 2) + 0.005, 0.5 * (1 - (i // 2)) + 0.005, 0.38, 0.42])
            f.add_axes(ax)
            # add_map(ax)
            im_frame = Image.open('/home/ck28372/safe-curriculum/safety_passage_push_birdeye.png').convert('RGB')
            # save the frame
            np_frame = np.array(im_frame)
            ax.imshow(np_frame, extent=(-3, 3, -3.25, 3.25))
            # print(contexts_all)
            # sort context_all wrt to the radius in descending order
            contexts_all = contexts_all[contexts_all[:, 2].argsort()[::-1]]
            scat = ax.scatter(contexts_all[:, 0], contexts_all[:, 1], alpha=0.3,
                            c=mapper.to_rgba(contexts_all[:, 2]), 
                            s=0,
                            # s=ax.get_window_extent().width, #((ax.get_window_extent().width  / (contexts_all[:,2]+1) * 72./f.dpi) ** 2),
                            # linewidth=0,
                            )
            for context in contexts_all:
                xi, yi, rad = context
                ax.add_patch(plt.Circle((xi, yi), rad*2, color=mapper.to_rgba(rad), alpha=0.3))

            ax.set_xlim(-3,  3)
            ax.set_ylim(-3.25,  3.25)

            ax.set_title(r"Epoch $%d$" % iteration, fontsize=TICK_SIZE, pad=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params('both', length=0, width=0, which='major')
            ax.set_rasterized(True)

            f.colorbar(scat, cax=cax, orientation='vertical')
            cax.tick_params('y', length=2, width=0.5, which='major', pad=1)
            cax.set_yticks([0.0, 0.5, 1.0])
            cax.set_yticklabels([r"$0.25$", r"$0.5$", r"$0.75$"], fontsize=TICK_SIZE)
        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"safety_passage_push_3d_curriculum_progressions_{label}_iter={iterations}{figname_extra}.pdf")
        print(figpath)
        plt.savefig(figpath, dpi=500, bbox_inches='tight')

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    iterations = [5, 40, 75, 150]
    # seeds = ["1", "3", "4", "5", "6"]
    # figname_extra = "_s13456"
    seeds = ["1"]
    figname_extra = "_s1"
    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))
    algorithm = [
    #    {
    #         "algorithm": "constrained_wasserstein",
    #         "label": "SCG-NoAnn",
    #         "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
    #     },  
    #     {
    #         "algorithm": "constrained_wasserstein",
    #         "label": "SCG-NoPP",
    #         "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=False_PS=True_RAS=10",
    #     },
    #     {
    #         "algorithm": "constrained_wasserstein",
    #         "label": "SCG-NoPS",
    #         "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=False_RAS=10",
    #     },
    #     {
    #         "algorithm": "constrained_wasserstein",
    #         "label": "SCG-NoPPPS",
    #         "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=False_PS=False_RAS=10",
    #     },
        # {
        #     "algorithm": "default",
        #     "label": "DEFAULT",
        #     "model": "PPOLag_DELTA_CS=0.0",
        # },
        {
            "algorithm": "constrained_wasserstein",
            "label": "SCG",
            "model": "PPOLag_DELTA_CS=0.0_ATP=0.75_CAS=10_DELTA=0.6_DELTA_CT=1.0_METRIC_EPS=0.25_PP=True_PS=True_RAS=10",
        },
        # {
        #     "algorithm": "wasserstein",
        #     "label": "NaiveSafeCURROT",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=1.0",
        # },
        {
            "algorithm": "wasserstein",
            "label": "CURROT",
            "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=0.25_PEN_COEFT=0.0",
        },
        # {
        #     "algorithm": "wasserstein4cost",
        #     "label": "CURROT4Cost",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=1.0_METRIC_EPS=0.25",
        # },
        # {
        #     "algorithm": "alp_gmm",
        #     "label": "ALP-GMM",
        #     "model": "PPOLag_DELTA_CS=0.0_AG_FIT_RATE=200_AG_MAX_SIZE=500_AG_P_RAND=0.2",
        # },
        # {
        #     "algorithm": "plr",
        #     "label": "PLR",
        #     "model": "PPOLag_DELTA_CS=0.0_PLR_BETA=0.15_PLR_REPLAY_RATE=0.55_PLR_RHO=0.45",
        # },
        # {
        #     "algorithm": "self_paced",
        #     "label": "SPDL",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_DIST_TYPE=gaussian_INIT_VAR=0.0_KL_EPS=0.25_PEN_COEFT=0.0",
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
