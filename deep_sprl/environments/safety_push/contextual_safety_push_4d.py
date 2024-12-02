from __future__ import annotations

import random
from typing import Any, ClassVar

import os
from typing import Tuple, Union
import pickle

import gymnasium as gym
import numpy as np
import robosuite
import torch
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from robosuite import load_controller_config
from transforms3d import euler
import deep_sprl.environments.safety_push.push_puck

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU

os.environ["MUJOCO_GL"] = "egl"

@env_register
class ContextualSafetyPush4D(CMDP):
    _support_envs: ClassVar[list[str]] = ['ContextualSafetyPush4D-v0']
    metadata: ClassVar[dict[str, int]] = {'render_fps': 10}
    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    MAX_TIME_STEPS = 500

    def __init__(self, 
                 env_id: str,
                 num_envs: int = 1,
                 device: torch.device = DEVICE_CPU,
                 **kwargs,
                 ):
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = device
        self._state = None
        self._timestep = 0
        self._hazard_passes = 0
        self.sparse_reward = False
        # gx: [0,1] -> [-0.4, 0.4] (left, right)
        # gy: [0,1] -> [-0.4, 0.4] (down, up)
        # hy: [0,1] -> [0, 0.15] (full hazard, no hazard)
        # goal_tolerance: [0,1] -> [0.05, 0.2] (clinical, relaxed)
        self._context = np.array([0.875, 0.5, 1.0, 0.]) # gx, gy, hy, goal_tolerance
        self._single_pass_cost = 0.25

        # set up position and rotation scales
        self.position_scale = 20
        self.rotation_scale = 2
        self.min_rotation = -np.pi / 2
        self.max_rotation = np.pi / 2
        self.pos_threshold = 0.05 # 0.02
        self.vel_threshold = 0.05 # 0.02
        self.arm_height = 0.825
        # self.arm_xy_range = np.array([-0.1, -0.1, 0.1, 0.1])
        # self.arm_xy_range = np.array([-0.3, -0.1, -0.2, 0.1]) 
        # self.arm_xy_range = np.array([0.25, -0.25, 0.25, -0.25])
        # self.arm_xy_range = np.array([-0.275, -0.5, -0.225, 0.5]) 
        # self.arm_xy_range = np.array([-0.27, -0.1, -0.26, 0.1]) 
        self.arm_xy_range = np.array([-0.2, -0.2, -0.16, 0.2]) 
        self.hazard_x_range = np.array([-0.1, 0.1]) # np.array([-0.175, 0.075])
        self.hazard_y_range = np.array([-0.075, 0.075]) # np.array([-0.125, 0.125]) # np.array([-0.075, 0.075])
        self.reset_step_limit = 100
        self.reset_distance_th = 0.05 # 0.01 # 0.002
        self.safety_radius = 0.03 # cylinder radius is 0.03

        self._env = robosuite.make(
            "PushPuck",
            robots="Panda",
            controller_configs=load_controller_config(default_controller="OSC_POSE"),
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=False,
            use_camera_obs=True,
            control_freq=10,
            horizon=self.MAX_TIME_STEPS+500,
            cylinder_xrange=(0.1,-0.1),#(-0.25,-0.25),# (-0.275,-0.275), #(-0.3, -0.3),
            cylinder_yrange=(0.05,0.05),#(-0.05, 0.05),
        )
        self._observation_space = spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        self._action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)  # delta x, delta y, rot z

    @staticmethod
    def sample_initial_state(n=None):
        return None
    
    @staticmethod
    def transform_context(context):
        gx = context[0]*0.4-0.2 # context[0]*0.6-0.25
        gy = context[1]*0.4-0.2
        hly = context[2]*0.05
        g_tol = context[3]*0.075+0.025 # context[3]*0.1+0.025
        return np.array([gx, gy, hly, g_tol])

    @staticmethod
    def _is_feasible(context):
        return True
    
    def reset(
        self,
        seed: int=None,
        options: dict[str, Any] = None):
        """
        Extends env reset method to return flattened observation instead of normal
        OrderedDict and optionally resets seed
        Inputs:
            seed (int): Optional seed to reset the environment to
        Returns:
            np.array: Flattened environment observation space after reset occurs
            info (dict): Dictionary containing additional information about the env
        """
        info = {}
        info["is_success"] = False

        # set random seed
        if seed is not None:
            self.set_seed(seed)

        self._real_context = self.transform_context(self._context)
        self.goal_pos_xy = self._real_context[:2]
        self.pos_threshold = self._real_context[3]
        self.goal_vel_xy = np.zeros(2)
        reset_done = False
        time_to_reset = 0
        print("\nReset!")
        while not reset_done:
            time_to_reset += 1
            overlap = False
            time_to_not_overlap = 0
            while not overlap:
                time_to_not_overlap += 1
                obs = self._env.reset()
                obj_pos = obs["object_pos"][:2]
                self.arm_init = np.random.uniform(
                    low=[self.arm_xy_range[0], self.arm_xy_range[1]],
                    high=[self.arm_xy_range[2], self.arm_xy_range[3]],
                )
                if np.linalg.norm(self.arm_init - obj_pos) > 0.05:
                    overlap = True

            obs = self._env.set_goal(
                goal_pos_xy=self.goal_pos_xy, goal_vel_xy=self.goal_vel_xy
            )

            # move the arm to the desired height , np.array([self.arm_height]
            # move xy first
            failed = False
            time_to_move_xy = 0
            while (
                not reset_done
                and np.linalg.norm(obs["robot0_eef_pos"][:2] - self.arm_init) > self.reset_distance_th
            ):
                time_to_move_xy += 1
                obs, reward, done, info = self._env.step(
                    np.array(
                        [
                            (self.arm_init[0] - obs["robot0_eef_pos"][0])
                            * self.position_scale,
                            (self.arm_init[1] - obs["robot0_eef_pos"][1])
                            * self.position_scale,
                            0,
                            0,  # x rotation
                            0,  # y rotation
                            0,  # z rotation
                            1,  # gripper
                        ]
                    )
                )
                if done or self.reset_step_limit < time_to_move_xy:
                    print(f"Failed to move xy")
                    failed = True
                    break

            # move xyz
            time_to_move_xyz = 0
            while (
                not failed
                and np.linalg.norm(
                    obs["robot0_eef_pos"]
                    - np.concatenate([self.arm_init, np.array([self.arm_height])])
                )
                > self.reset_distance_th
            ):
                time_to_move_xyz += 1
                obs, reward, done, info = self._env.step(
                    np.array(
                        [
                            (self.arm_init[0] - obs["robot0_eef_pos"][0])
                            * self.position_scale,
                            (self.arm_init[1] - obs["robot0_eef_pos"][1])
                            * self.position_scale,
                            (self.arm_height - obs["robot0_eef_pos"][2])
                            * self.position_scale,  # z translation
                            0,  # x rotation
                            0,  # y rotation
                            0,  # z rotation
                            1,  # gripper
                        ]
                    )
                )
                if done or self.reset_step_limit < time_to_move_xyz:
                    print(f"Failed to move xyz")
                    failed = True
                    break
            print(f"not overlap: {time_to_not_overlap} || move xy: {time_to_move_xy} || move xyz: {time_to_move_xyz}")
            if not failed:
                reset_done = True

        self.obs_robotsuite = obs
        self._state = torch.as_tensor(
            np.concatenate((
                obs["object_pos"][:2],
                obs["object_vel"][:2],
                obs["robot0_eef_pos"][:2],
                )
            )
        )
        self._timestep = 0
        self._hazard_passes = 0
        self.dist_goal_puck = np.linalg.norm(obs["object_pos"][:2] - self._real_context[:2])
        self.dist_puck_eff = np.linalg.norm(obs["object_pos"][:2] - obs["robot0_eef_pos"][:2])
        print(f"\nInitial state: {self._state} || Context: {self._context} || Real context: {self._real_context}")
        # print(f"Obs robosuite: {self.obs_robotsuite}")
        return torch.clone(self._state), {}

    def step(self, action):
        self._timestep += 1

        # extract action
        action = action.detach().cpu().numpy()

        # extract rot z
        current_quat = self.obs_robotsuite["robot0_eef_quat"]
        correction_angles = euler.quat2euler(current_quat, axes="szyx")
        correction_actions = np.array(correction_angles) * self.rotation_scale
        # step the environment
        action_7 = np.array(
            [
                action[0], # delta_x
                action[1], # delta_y
                (self.arm_height - self.obs_robotsuite["robot0_eef_pos"][2])
                * self.position_scale,
                correction_actions[0],
                correction_actions[1],
                0,  # delta_rot_zs
                1,
            ]
        )
        obs, _, done, info = self._env.step(action_7)
        new_dist_goal_puck = np.linalg.norm(obs["object_pos"][:2] - self._real_context[:2])
        new_dist_puck_eff = np.linalg.norm(obs["object_pos"][:2] - obs["robot0_eef_pos"][:2])
        self.obs_robotsuite = obs
        self._state = torch.as_tensor(
            np.concatenate((
                obs["object_pos"][:2],
                obs["object_vel"][:2],
                obs["robot0_eef_pos"][:2],
                )
            )
        )
        success = torch.as_tensor(self._env._check_success(self.pos_threshold, self.vel_threshold))
        if self.sparse_reward:
            # reward = torch.as_tensor(-1.0 * (not success))
            reward = torch.as_tensor(1.0 * success)
        else:
            # DENSE-0: distance bt goal & puck and puck & eef + relative velocity terms
            # obs_eef_xy = np.linalg.norm(obs["object_pos"][:2] - obs["robot0_eef_pos"][:2])
            # obs_eef_vel = np.linalg.norm(obs["object_vel"][:2])
            # obs_goal_xy = np.linalg.norm(obs["object_pos"][:2] - obs["goal_xy_pos"])
            # obs_goal_vel = np.linalg.norm(obs["object_vel"][:2] - obs["goal_xy_vel"])
            # reward = torch.as_tensor(np.clip(
            #     -(obs_goal_xy + obs_goal_vel) - (obs_eef_xy + obs_eef_vel) + 1.0 * success,
            #     -1, 1)
            #     )

            # DENSE-1: distance bt goal & puck and puck & eef
            # obs_eef_xy = np.linalg.norm(obs["object_pos"][:2] - obs["robot0_eef_pos"][:2])
            # obs_goal_xy = np.linalg.norm(obs["object_pos"][:2] - obs["goal_xy_pos"])
            # reward = torch.as_tensor(-obs_goal_xy - obs_eef_xy + 1.0 * success)

            # DENSE-2: change of distance bt goal & puck and puck & eef
            # reward = torch.as_tensor((self.dist_goal_puck-new_dist_goal_puck)+(self.dist_puck_eff-new_dist_puck_eff) + 1.0 * success)

            # DENSE-3: change of distance bt goal & puck
            reward = torch.as_tensor((self.dist_goal_puck-new_dist_goal_puck) + 1.0 * success)

        self.dist_goal_puck = new_dist_goal_puck
        self.dist_puck_eff = new_dist_puck_eff
        # in_hazard_left = (self.hazard_x_range[0] <= self._state[0] <= self.hazard_x_range[1]) and (
        #     self.hazard_y_range[0] <= self._state[1] <= -self._real_context[2])
        # in_hazard_right = (self.hazard_x_range[0] <= self._state[0] <= self.hazard_x_range[1]) and (
        #     self.hazard_y_range[1] >= self._state[1] >= self._real_context[2])
        in_hazard_left = (self.hazard_x_range[0]-self.safety_radius <= self._state[0] <= self.hazard_x_range[1]+self.safety_radius) and (
            self.hazard_y_range[0]-self.safety_radius <= self._state[1] <= -self._real_context[2]+self.safety_radius)
        in_hazard_right = (self.hazard_x_range[0]-self.safety_radius <= self._state[0] <= self.hazard_x_range[1]+self.safety_radius) and (
            self.hazard_y_range[1]+self.safety_radius >= self._state[1] >= self._real_context[2]-self.safety_radius)
        
        if in_hazard_left or in_hazard_right:
            print(f"state: {self._state}")
            input(f" in_hazard_left: {in_hazard_left} || in_hazard_right: {in_hazard_right}")

        cost = torch.as_tensor(self._single_pass_cost * (in_hazard_left or in_hazard_right))
        self._hazard_passes += 1 if cost > 0 else 0
        info = {"success": success,
                "reward": torch.as_tensor(1.0) if success else torch.as_tensor(0.), 
                "cost": cost}
        terminated = truncated = torch.as_tensor((self._timestep >= self.MAX_TIME_STEPS) or info["success"] or done)
        if terminated:
            info["final_observation"] = torch.clone(self._state)
            print(f"Terminated at step {self._timestep} || state: {self._state} || success: {success} || hazard passes: {self._hazard_passes}")
        return torch.clone(self._state), reward, cost, terminated, truncated, info

    def render(self, mode="rgb_array") -> None:
        """
        Render the environment.
        """
        if mode == "human":
            self._env.render()
        elif mode == "rgb_array":
            return self.obs_robotsuite["agentview_image"]

    def close(self) -> None:
        """
        Close the environment.
        """
        self._env.close()

    def set_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def sample_action(self):
        return torch.as_tensor(self._action_space.sample())

    @property
    def context(self):
        return self._context
    
    @context.setter
    def context(self, context):
        self._context = context