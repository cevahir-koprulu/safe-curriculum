import os
import sys
import math
import torch
import numpy as np
import gymnasium
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.patches import Rectangle
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import deep_sprl.environments
from omnisafe.models.actor import ActorBuilder
from omnisafe.common import Normalizer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def rollout_policy(policy, env):
    obs, info = env.reset()
    terminated = False
    truncated = False
    observations = []
    actions = []
    rewards = []
    costs = []
    success = []
    while not terminated and not truncated:
        with torch.no_grad():
            action = policy(obs)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        # print(f"obs: {obs}, action: {action}, reward: {reward}, cost: {cost}, terminated: {terminated}, truncated: {truncated}")
        observations.append(obs.detach().numpy())
        actions.append(action.detach().numpy())
        rewards.append(reward.detach().numpy())
        costs.append(cost.detach().numpy())
        success.append(info["success"].detach().numpy()*1)
    return np.array(observations), np.array(actions), np.array(rewards), np.array(costs), np.array(success)

def load_policy(model_path, exp, device='cpu'):
    if os.path.exists(model_path):
        custom_cfgs = exp.create_learner_params()
        actor_builder = ActorBuilder(
            obs_space=exp.eval_env._observation_space,
            act_space=exp.eval_env._action_space,
            hidden_sizes=custom_cfgs['model_cfgs']['actor']['hidden_sizes'],
            activation=custom_cfgs['model_cfgs']['actor']['activation'],
            weight_initialization_mode=custom_cfgs['model_cfgs']['weight_initialization_mode'],
        )
        actor = actor_builder.build_actor(custom_cfgs['model_cfgs']['actor_type'])
        model_params = torch.load(model_path, map_location=device)
        actor.load_state_dict(model_params['pi'])
        old_min_action = torch.tensor(
            exp.eval_env._action_space.low,
            dtype=torch.float32,
            device=device,
        )
        old_max_action = torch.tensor(
            exp.eval_env._action_space.high,
            dtype=torch.float32,
            device=device,
        )
        min_action = torch.zeros_like(old_min_action) - 1
        max_action = torch.zeros_like(old_min_action) + 1
        def descale_action(scaled_act):
            return old_min_action + (old_max_action - old_min_action) * (
                scaled_act - min_action) / (max_action - min_action)
        
        if "obs_normalizer" in model_params:
            normalizer = Normalizer(exp.eval_env._observation_space.shape).to(device)
            normalizer.load_state_dict(model_params["obs_normalizer"])
            return lambda obs: descale_action(actor.predict(normalizer.normalize(obs), deterministic=False))
        else:
            return lambda obs: descale_action(actor.predict(obs, deterministic=False))
    else:
        return None
        # raise ValueError(f"No policy found at path: {policy_path}")
    
def load_eval_contexts(experiment_name):
    return np.load(os.path.join(Path(os.getcwd()).parent, "eval_contexts", f"{experiment_name}_eval_contexts.npy"))

def plot_trajectories(base_log_dir, policy_from_iteration, seeds, exp, env_name, experiment_name,
                      discount_factor, setting, algorithms, figname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]

    context = np.array([2.4, -2.4])
    # context = np.array([-3.5, 3.5])
    # context = load_eval_contexts(experiment_name)[0]
    # context = np.array([-4.0, 4.0])

    fig, axes = plt.subplots(1, len(seeds), figsize=figsize, constrained_layout=True)
    plt.suptitle(f"Context: ({context[0]},{context[1]}) || Iteration: {policy_from_iteration}")
    if len(seeds) == 1:
        axes = [axes]
    else:
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
            omnisafe_dir_file = os.path.join(base_log_dir, "logs", experiment_name, algorithm, model, f"seed-{seed}", "omnisafe_log_dir.txt")
            with open(omnisafe_dir_file, "r") as f:
                omnisafe_dir = f.read()
            policy_path = os.path.join(base_log_dir, omnisafe_dir, "torch_save", f"epoch-{policy_from_iteration}.pt")
            print(policy_path)
            policy = load_policy(model_path=policy_path, exp=exp)
            if policy is None:
                continue
            exp.eval_env.set_context(context)
            obs, acts, rews, costs, succs = rollout_policy(policy, exp.eval_env)
            disc_return = np.sum(rews * np.power(discount_factor, np.arange(rews.shape[0])))
            disc_cost = np.sum(costs * np.power(discount_factor, np.arange(costs.shape[0])))
            x, v_x, y, v_y = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
            axes[seed_i].plot(x+4.0, y+4.0, color=color, alpha=0.5, linewidth=3.0)
            axes[seed_i].quiver(x+4.0, y+4.0, v_x/(10*np.sqrt(v_x**2+v_y**2)), v_y/(10*np.sqrt(v_x**2+v_y**2)),
                                 color=color, alpha=0.5, width=0.01)
            axes[seed_i].text(0.1, 0.1+0.5*algo_i,
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
    lgd = fig.legend(lines, labels, ncol=min(num_alg,3), loc="upper center", bbox_to_anchor=bbox_to_anchor,
                     fontsize=fontsize, handlelength=1.0, labelspacing=0., handletextpad=0.5, columnspacing=1.0)

    figname = ""
    for cur_algo_i, algo in enumerate(algorithms):
        figname += algo
        if cur_algo_i < len(algorithms)-1:
            figname += "_vs_"

    if not os.path.exists(os.path.join(Path(os.getcwd()).parent, "figures")):
        os.makedirs(os.path.join(Path(os.getcwd()).parent, "figures"))

    figpath = os.path.join(Path(os.getcwd()).parent, "figures", 
                           f"{experiment_name}_{figname}{figname_extra}_c=({context[0]},{context[1]})"+\
                            f"_iter={policy_from_iteration}.pdf")
    print(figpath)
    plt.savefig(figpath, dpi=500, bbox_inches='tight', 
                # bbox_extra_artists=(lgd,),
                )

def main():
    base_log_dir = Path(os.getcwd()).parent
    policy_from_iteration = 150
    seeds = [str(i) for i in range(1, 4)]
    rl_algorithm = "PPOLag"
    experiment_name = "safety_point_mass_2d_narrow"
    env_name = "ContextualSafetyPointMass2D-v0"
    figname_extra = "_KL_EPS=1.0"
    discount_factor = 0.99
    
    if experiment_name[:experiment_name.rfind('_')] == "safety_point_mass_2d":
        from deep_sprl.experiments import SafetyPointMass2DExperiment
        exp = SafetyPointMass2DExperiment(base_log_dir="logs", curriculum_name="default", 
                                          learner_name=rl_algorithm, 
                                          parameters={"TARGET_TYPE": experiment_name[experiment_name.rfind('_')+1:]},
                                          seed=1, device="cpu")
    else:
        raise ValueError("Invalid environment")


    algorithms = {
        "safety_point_mass_2d_narrow": {
            "CSPDL": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=30",
                "model": "PPO_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "gray",
            },
            "CSPDL2": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=30",
                "model": "PPOLag_DELTA=30.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "tan",
            },
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
            "DEF_Lag": {
                "algorithm": "default",
                "label": "DEF_Lag",
                "model": "PPOLag",
                "color": "red",
            },
            "DEF": {
                "algorithm": "default",
                "label": "Default",
                "model": "PPO",
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
        exp=exp,
        env_name=env_name,
        experiment_name=experiment_name,
        discount_factor=discount_factor,
        setting=settings[experiment_name],
        algorithms=algorithms[experiment_name],
        figname_extra=figname_extra,
        )

if __name__ == "__main__":
    main()
