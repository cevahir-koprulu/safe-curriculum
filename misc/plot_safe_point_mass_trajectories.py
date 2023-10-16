import os
import sys
import math
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.patches import Rectangle
from stable_baselines3.ppo import PPO
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import deep_sprl.environments

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def rollout_policy(policy, env):
    obs = env.reset()
    done = False
    observations = []
    actions = []
    rewards = []
    costs = []
    success = []
    while not done:
        obs_ = np.concatenate((obs, env.context[:-1]))
        action = policy(obs_)
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        costs.append(info["cost"])
        success.append(info["success"]*1)
    return np.array(observations), np.array(actions), np.array(rewards), np.array(costs), np.array(success)

def load_policy(env, algo, policy_path, device='cpu'):
    if os.path.exists(policy_path):
        policy = algo.load(policy_path, device=device)
        return lambda x: policy.predict(x, state=None, deterministic=True)[0]
    else:
        raise ValueError(f"No policy found at path: {policy_path}")
    
def load_eval_contexts(experiment_name):
    return np.load(os.path.join(Path(os.getcwd()).parent, "eval_contexts", f"{experiment_name}_eval_contexts.npy"))

def plot_trajectories(base_log_dir, policy_from_iteration, seeds, env_name, experiment_name, rl_algorithm,
                      discount_factor, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]

    env = gym.make(env_name)
    context = load_eval_contexts(experiment_name)[0]

    fig, axes = plt.subplots(1, len(seeds), figsize=figsize, constrained_layout=True)
    plt.suptitle(f"Context: ({context[0]},{context[1]}) || Iteration: {policy_from_iteration}")
    axes = axes.flatten()
    for ax_i, ax in enumerate(axes):
        ax.set_xlim([0., 8.])
        ax.set_ylim([0., 8.])
        ax.set_xticks([0.0, context[0]+4.0, context[1]+4.0, 8.0])
        ax.set_yticks([0.0, 8.0])
        ax.set_aspect('equal')
        ax.set_title(f"Seed {seeds[ax_i]}",fontsize=fontsize)
        lava_1 = Rectangle((0.0, 5.0), context[0]+4.0, 2.0, facecolor="red", alpha=0.5)
        lava_2 = Rectangle((context[1]+4.0, 1.0), 8.0, 2.0, facecolor="red", alpha=0.5)
        ax.add_patch(lava_1)
        ax.add_patch(lava_2)
        ax.scatter(0.5, 7.5, marker="o", color="black", s=500)
        ax.scatter(7.5, 0.5, marker="X", color="green", s=500)

    for algo_i, algo in enumerate(algorithms):
        algorithm = algorithms[algo]["algorithm"]
        label = algorithms[algo]["label"]
        model = algorithms[algo]["model"]
        color = algorithms[algo]["color"]
        print(algorithm)

        for seed_i, seed in enumerate(seeds):
            policy_path = os.path.join(base_log_dir, experiment_name, algorithm, model, f"seed-{seed}", 
                                       f"iteration-{policy_from_iteration}", "model.zip")
            print(policy_path)
            policy = load_policy(env, rl_algorithm, policy_path)
            env.set_context(context)
            obs, acts, rews, costs, succs = rollout_policy(policy, env)
            disc_return = np.sum(rews * np.power(discount_factor, np.arange(rews.shape[0])))
            disc_cost = np.sum(costs * np.power(discount_factor, np.arange(costs.shape[0])))
            x, v_x, y, v_y = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            axes[seed_i].plot(x+4.0, y+4.0, color=color, alpha=0.5, linewidth=3.0)
            axes[seed_i].quiver(x+4.0, y+4.0, v_x/(10*np.sqrt(v_x**2+v_y**2)), v_y/(10*np.sqrt(v_x**2+v_y**2)),
                                 color=color, alpha=0.5, width=0.01)
            axes[seed_i].text(0.1, 0.1+1.0/len(algorithms)*algo_i,
                              f"Return: {disc_return:.2f} || Cost: {disc_cost:.2f} || Succ: {np.any(succs)}",
                              color=color)

    colors = []
    labels = []
    num_alg = len(algorithms)
    for algo in algorithms:
        colors.append(algorithms[algo]["color"])
        labels.append(algorithms[algo]["label"])

    markers = ["" for i in range(num_alg)]
    linestyles = ["-" for i in range(num_alg)]
    labels.reverse()
    lines = [Line2D([0], [0], color=colors[-i-1], linestyle=linestyles[i], marker=markers[i], linewidth=2.0)
             for i in range(num_alg)]
    lgd = fig.legend(lines, labels, ncol=num_alg, loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, algo in enumerate(algorithms):
        figname += algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))

    figpath = os.path.join(Path(os.getcwd()).parent, "figures", 
                           f"{experiment_name}_{figname}{figname_extra}_c={context}_iter={policy_from_iteration}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500, bbox_inches='tight', 
                # bbox_extra_artists=(lgd,),
                )

def main():
    base_log_dir = os.path.join(Path(os.getcwd()).parent, "logs")
    policy_from_iteration = 120
    seeds = [str(i) for i in range(1, 4)]
    rl_algorithm = PPO
    experiment_name = "safety_point_mass_2d_narrow"
    env_name = "ContextualSafetyPointMass2D-v1"
    figname_extra = "_KL_EPS=1.0_D30"
    discount_factor = 0.99
    
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
            "fontsize": 10,
            "figsize": (13, 5),
            "bbox_to_anchor": (.1, 1.01),
        },
    }

    plot_trajectories(
        base_log_dir=base_log_dir,
        policy_from_iteration=policy_from_iteration,
        seeds=seeds,
        env_name=env_name,
        experiment_name=experiment_name,
        rl_algorithm=rl_algorithm,
        discount_factor=discount_factor,
        setting=settings[experiment_name],
        algorithms=algorithms[experiment_name],
        figname_extra=figname_extra,
        )

if __name__ == "__main__":
    main()
