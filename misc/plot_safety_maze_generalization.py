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
from pathlib import Path

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from deep_sprl.experiments.safety_maze_3d_experiment import SafetyMaze3DExperiment

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

FONT_SIZE = 8
TICK_SIZE = 6

def get_results(base_dir, seeds, iteration):
    results = None
    contexts = None
    seeds_available = []
    for seed in seeds:
        res_seed_file = os.path.join(base_dir, f"seed-{seed}", f"iteration-{iteration}", "performance_hom.npy")
        if os.path.exists(res_seed_file):
            #load numpy as float
            res_seed = np.load(res_seed_file).astype(float)
            ret_seed = res_seed[:,1:2]
            contexts_seed = res_seed[:, 2:-3]
            success_seed = res_seed[:, -2:-1]
            cost_seed = res_seed[:, -1:]
            if results is None:
                # concatenate the results with the seed on the first column
                results = np.concatenate((np.array([int(seed)]*ret_seed.shape[0]).reshape(-1, 1), 
                                          contexts_seed, 
                                          ret_seed, 
                                          success_seed, 
                                          cost_seed), axis=1)
                contexts = contexts_seed
            else:
                results = np.vstack((results, 
                                     np.concatenate((np.array([int(seed)]*ret_seed.shape[0]).reshape(-1, 1), 
                                                             contexts_seed, 
                                                             ret_seed, 
                                                             success_seed, 
                                                             cost_seed), axis=1)))
            seeds_available.append(seed)
        else:
            print(f"File {res_seed_file} does not exist")
    # Compute average results among seeds for each context in contexts
    mean_results = None
    for context in contexts:
        context_results = results[(results[:, 1] == context[0]) * (results[:, 2] == context[1]) * (results[:, 3] == context[2])]
        mean_result = np.median(context_results[:, 1:], axis=0)
        if mean_results is None:
            mean_results = mean_result
        else:
            mean_results = np.vstack((mean_results, mean_result))
        # print mean results with 2 decimanl points
        print(f"{mean_result.round(2)}")
    return mean_results

def sample_contexts():
    return

def maze():
    # Maze
    maze = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 0, 0, 0, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 'h', 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ])
    # We draw a black white image
    maze_image = 255 * np.ones(maze.shape + (3,))
    x, y = np.where(maze == '1')
    maze_image[x, y, :] = 0.
    x, y = np.where(maze == "r")
    maze_image[x, y, 0::2] = 0.
    x, y = np.where(maze == "h")
    maze_image[x, y, 1:] = 0.
    return maze_image

def plot_curriculum_progression(base_log_dir, seeds, algorithm, iteration, figname_extra, results_for):
    results_indices = {
        'return': 0,
        'success': 1,
        'cost': 2,
    }
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = FONT_SIZE

    exp = SafetyMaze3DExperiment(base_log_dir="logs", curriculum_name="default", 
                                    learner_name="PPOLag", parameters={},
                                    seed=1, device="cpu")
    norm = matplotlib.colors.Normalize(vmin=exp.LOWER_CONTEXT_BOUNDS[-1], vmax=3.35, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")

    for algo_dict in algorithm:
        algorithm_name  = algo_dict["algorithm"]
        label = algo_dict["label"]
        model = algo_dict["model"]
        base_dir = os.path.join(base_log_dir, "safety_maze_3d", algorithm_name, model)
        print(base_dir)
        results = get_results(
            base_dir=base_dir,
            seeds=seeds,
            iteration=iteration,
        )
        f = plt.figure(figsize=(1.5, 1.45))
        plt.imshow(maze(), extent=(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0], exp.LOWER_CONTEXT_BOUNDS[1],
                                    exp.UPPER_CONTEXT_BOUNDS[1]), origin="lower")
        print(results[:, -3+results_indices[results_for]])
        scat = plt.scatter(results[:, 0], results[:, 1], alpha=0.3,
                            c=results[:, -3+results_indices[results_for]], s=2, vmin=0, vmax=1)
                            # c=mapper.to_rgba(results[:, -3+results_indices[results_for]]), s=2)
        plt.xlim(exp.LOWER_CONTEXT_BOUNDS[0], exp.UPPER_CONTEXT_BOUNDS[0])
        plt.ylim(exp.LOWER_CONTEXT_BOUNDS[1], exp.UPPER_CONTEXT_BOUNDS[1])
        plt.title(results_for.capitalize(), fontsize=TICK_SIZE, pad=0)
        plt.xticks([])
        plt.yticks([])
        plt.tick_params('both', length=0, width=0, which='major')
        plt.gca().set_rasterized(True)
        f.colorbar(scat, orientation='vertical')
        figpath = os.path.join(Path(os.getcwd()).parent, "figures", f"safety_maze_3d_generalization_{label}_iter={iteration}_{results_for}{figname_extra}.pdf")
        print(figpath)
        plt.savefig(figpath, dpi=500, bbox_inches='tight')

def main():

    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    # iterations = [5, 30, 100, 450]
    # iterations = [10, 30, 50, 150]
    iteration = 495
    seeds = [str(i) for i in range(1, 11)]
    results_for = "success"
    figname_extra = ""
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
            "model": "PPOLag_DELTA_CS=0.0_ATP=1.0_CAS=10_DELTA=0.6_DELTA_CT=0.25_METRIC_EPS=1.25_PP=True_PS=True_RAS=10",
        },
        # {
        #     "algorithm": "wasserstein",
        #     "label": "NaiveSafeCURROT",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=1.0",
        # },
        # {
        #     "algorithm": "wasserstein",
        #     "label": "CURROT",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA=0.6_METRIC_EPS=1.25_PEN_COEFT=0.0",
        # },
        # {
        #     "algorithm": "wasserstein4cost",
        #     "label": "CURROT4Cost",
        #     "model": "PPOLag_DELTA_CS=0.0_DELTA_CT=0.25_METRIC_EPS=1.25",
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
        iteration=iteration,
        figname_extra=figname_extra,
        results_for=results_for,
        )

if __name__ == "__main__":
    main()
