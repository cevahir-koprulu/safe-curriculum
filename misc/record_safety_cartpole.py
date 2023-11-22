import os
import sys
import torch
import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import deep_sprl.environments
from deep_sprl.environments.safety_cartpole import ContextualSafetyCartpole2D
from omnisafe.models.actor import ActorBuilder
from omnisafe.common import Normalizer
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def add_red_bar(frame, context):
    left_bar = int((ContextualSafetyCartpole2D.X_THRESHOLD + context[0])/(
        2*ContextualSafetyCartpole2D.X_THRESHOLD)*frame.shape[1])
    right_bar = int((ContextualSafetyCartpole2D.X_THRESHOLD + context[1])/(
        2*ContextualSafetyCartpole2D.X_THRESHOLD)*frame.shape[1])
    frame[:,left_bar-5:left_bar+5,0] = frame[:,right_bar-5:right_bar+5,0] = 255
    frame[:,left_bar-5:left_bar+5,1] = frame[:,left_bar-5:left_bar+5,2] = 0
    frame[:,right_bar-5:right_bar+5,1] = frame[:,right_bar-5:right_bar+5,2] = 0
    return frame

def rollout_policy(policy, env):
    terminated = False
    truncated = False
    rewards = [0.0]
    costs = [0.0]
    success = [0]
    rendereds = []
    obs, info = env.reset()
    rendered_obs = env.render()
    rendereds.append(add_red_bar(rendered_obs, env.context))
    while not terminated and not truncated:
        with torch.no_grad():
            action = policy(obs)
        obs, reward, cost, terminated, truncated, info = env.step(action)
        rendered_obs = env.render()
        rendereds.append(add_red_bar(rendered_obs, env.context))
        rewards.append(reward.detach().numpy())
        costs.append(cost.detach().numpy())
        success.append(info["success"].detach().numpy()*1)
    return np.array(rendereds), np.array(rewards), np.array(costs), np.array(success)

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
                      discount_factor, setting, algorithms, vidname_extra):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.size'] = setting["fontsize"]

    fontsize = setting["fontsize"]
    figsize = setting["figsize"]
    bbox_to_anchor = setting["bbox_to_anchor"]

    if "2D" in env_name:
        context = np.array([-0.6, 0.6])
        # context = np.array([-1.2, 1.2])
        # context = np.array([-3.5, 3.5])
        # context = load_eval_contexts(experiment_name)[0]
        # context = np.array([-4.0, 4.0])
    elif "1D" in env_name:
        context = np.array([-1.2])
        # context = np.array([3.5])
        # context = load_eval_contexts(experiment_name)[0]
        # context = np.array([-4.0])

    # Collect rollout data
    fig = plt.figure(constrained_layout=True)
    frames_dict = {}
    for algo_i, algo in enumerate(algorithms):
        algorithm = algorithms[algo]["algorithm"]
        label = algorithms[algo]["label"]
        model = algorithms[algo]["model"]
        color = algorithms[algo]["color"]
        print(algorithm)
        frames_dict[algo] = {}
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
            frames, rews, costs, succs = rollout_policy(policy, exp.eval_env)
            disc_return = np.cumsum(rews * np.power(discount_factor, np.arange(rews.shape[0])))
            disc_cost = np.cumsum(costs * np.power(discount_factor, np.arange(costs.shape[0])))
            success = (np.cumsum(succs)>0)*1.0
            frames_dict[algo][seed] = {
                'frames': frames,
                'disc_return': disc_return,
                'disc_cost': disc_cost,
                'success': success,
            }

    # Merge frames
    max_len = 0
    for algo in frames_dict:
        for seed in frames_dict[algo]:
            max_len = max(max_len, frames_dict[algo][seed]['frames'].shape[0])
            frame_shape = frames_dict[algo][seed]['frames'][0].shape
    
    merge_frame_shape = (frame_shape[0]*len(algorithms)+10*(len(algorithms)+1), 
                         frame_shape[1]*len(seeds)+10*(len(seeds)+1), 
                         frame_shape[2]) # (H, W, C)
    merged_frames = []
    for t in range(max_len):
        merged_frame = np.zeros(merge_frame_shape, dtype=np.uint8)
        for algo_i, algo in enumerate(algorithms):
            for seed_i, seed in enumerate(seeds):
                if t < frames_dict[algo][seed]['frames'].shape[0]:
                    merged_frame[10*(algo_i+1)+frame_shape[0]*algo_i:frame_shape[0]*(algo_i+1)+10*(algo_i+1),
                                 10*(seed_i+1)+frame_shape[1]*seed_i:frame_shape[1]*(seed_i+1)+10*(seed_i+1),:] = \
                        frames_dict[algo][seed]['frames'][t]
                else:
                    merged_frame[10*(algo_i+1)+frame_shape[0]*algo_i:frame_shape[0]*(algo_i+1)+10*(algo_i+1),
                                 10*(seed_i+1)+frame_shape[1]*seed_i:frame_shape[1]*(seed_i+1)+10*(seed_i+1),:] = \
                        frames_dict[algo][seed]['frames'][-1]
                    
        img = Image.fromarray(merged_frame)
        draw = ImageDraw.Draw(img)    
        for algo_i, algo in enumerate(algorithms):
            for seed_i, seed in enumerate(seeds):
                if t < frames_dict[algo][seed]['frames'].shape[0]:
                    draw.text((10*(seed_i+1)+frame_shape[1]*seed_i+10,
                               10*(algo_i+1)+frame_shape[0]*algo_i+50),
                               f"t={t} || R={frames_dict[algo][seed]['disc_return'][t]:.2f} ||"+\
                                f" C={frames_dict[algo][seed]['disc_cost'][t]:.2f} ||"+\
                                f" S={frames_dict[algo][seed]['success'][t]:.2f}",                                
                                font=ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24),
                                fill=(0,0,0))
                else:
                    draw.text((10*(seed_i+1)+frame_shape[1]*seed_i+10,
                               10*(algo_i+1)+frame_shape[0]*algo_i+50),
                               f"t={frames_dict[algo][seed]['frames'].shape[0]-1} ||"+\
                                f" R={frames_dict[algo][seed]['disc_return'][-1]:.2f} ||"+\
                                f" C={frames_dict[algo][seed]['disc_cost'][-1]:.2f} ||"+\
                                f" S={frames_dict[algo][seed]['success'][-1]:.2f}",
                                font=ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24),
                                fill=(0,0,0))
                draw.text((10*(seed_i+1)+frame_shape[1]*seed_i+10,
                            10*(algo_i+1)+frame_shape[0]*algo_i+10),
                            f"{algorithms[algo]['label']} || seed={seed}",                                
                            font=ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24),
                            fill=(0,0,0))
        plt.axis('off')
        merged_frames.append([plt.imshow(np.asarray(img), animated=True)])

    vidname = ""
    for cur_algo_i, algo in enumerate(algorithms):
        vidname += algo
        if cur_algo_i < len(algorithms)-1:
            vidname += "_vs_"

    vid_dir = os.path.join(Path(os.getcwd()).parent, "videos")
    if not os.path.exists(vid_dir):
        os.makedirs(vid_dir)

    if "2D" in env_name:
        vid_path = os.path.join(vid_dir, 
                           f"{experiment_name}_{vidname}{vidname_extra}_c=({context[0]},{context[1]})"+\
                            f"_iter={policy_from_iteration}.mp4")
    elif "1D" in env_name:
        vid_path = os.path.join(vid_dir,
                            f"{experiment_name}_{vidname}{vidname_extra}_c=({context[0]})"+\
                            f"_iter={policy_from_iteration}.mp4")
    print(vid_path)

    ani = animation.ArtistAnimation(fig, merged_frames, interval=100, blit=True,
                                    repeat_delay=1000)
    ani.save(filename=vid_path, dpi=500)

def main():
    base_log_dir = Path(os.getcwd()).parent
    policy_from_iteration = 200
    seeds = [str(i) for i in range(1, 4)]
    # seeds = [1]
    rl_algorithm = "PPOLag"
    experiment_name = "safety_cartpole_2d_narrow"
    env_name = "ContextualSafetyCartpole2D-v0"
    vidname_extra = "_KL_EPS=1.0_D=40.0"
    # vidname_extra = ""
    discount_factor = 0.99
    
    if experiment_name[:experiment_name.rfind('_')] == "safety_cartpole_2d":
        from deep_sprl.experiments import SafetyCartpole2DExperiment
        exp = SafetyCartpole2DExperiment(base_log_dir="logs", curriculum_name="default", 
                                          learner_name=rl_algorithm, 
                                          parameters={"TARGET_TYPE": experiment_name[experiment_name.rfind('_')+1:]},
                                          seed=1, device="cpu")
    else:
        raise ValueError("Invalid environment")


    algorithms = {
        "safety_cartpole_2d_narrow": {
            "CSPDL": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL_D=40",
                "model": "PPO_DELTA=40.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "gray",
            },
            "CSPDL2": {
                "algorithm": "constrained_self_paced",
                "label": "CSPDL2_D=40",
                "model": "PPOLag_DELTA=40.0_DELTA_C=0.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "tan",
            },
            "SPDL": {
                "algorithm": "self_paced",
                "label": "SPDL_D=40",
                "model": "PPO_DELTA=40.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
                "color": "blue",
            },
            "SPDL2": {
                "algorithm": "self_paced",
                "label": "SPDL2_D=40",
                "model": "PPOLag_DELTA=40.0_DIST_TYPE=gaussian_INIT_VAR=0.1_KL_EPS=1.0",
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
        "safety_cartpole_2d_narrow":{
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
        vidname_extra=vidname_extra,
        )

if __name__ == "__main__":
    main()
