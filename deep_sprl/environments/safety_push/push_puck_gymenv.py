"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like
interface.
"""
import os
from typing import Tuple, Union
import pickle

import gymnasium as gym
import numpy as np
import robosuite
from gymnasium import spaces
from gymnasium.utils.ezpickle import EzPickle
from robosuite import load_controller_config
from transforms3d import euler


import deep_sprl.environments.safety_push.push_puck  # noqa: F401

os.environ["MUJOCO_GL"] = "egl"


class PushPuckGymEnv(gym.Env):
    def __init__(
        self,
        pos_threshold=0.02,
        vel_threshold=0.02,
        reward_shaping=True,
        arm_height=0.825,
        goal_env=False,
        cylinder_xrange=(-0.05, 0.05),  # 0.03
        cylinder_yrange=(-0.05, 0.05),  # 0.03
        **kwargs,
    ) -> None:
        """
        Initializes the PushPuckGymEnv environment.
        Args:
            pos_threshold: threshold for position error
            vel_threshold: threshold for velocity error
            reward_shaping: whether to use reward shaping
            arm_height: height of the arm above the ground in meters (the table is 0.8m)
            goal_env: whether to use goal environment
        """

        # set up position and rotation scales
        self.position_scale = 20
        self.rotation_scale = 2
        self.min_rotation = -np.pi / 2
        self.max_rotation = np.pi / 2
        self.pos_threshold = pos_threshold
        self.vel_threshold = vel_threshold
        self.reward_shaping = reward_shaping
        self.arm_height = arm_height
        self.arm_xy_range = np.array([0, 0, 0, 0])
        self.goal_env = goal_env

        # import from robosuite environment
        config = load_controller_config(default_controller="OSC_POSE")
        self.env = robosuite.make(
            "PushPuck",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=False,
            use_camera_obs=False,
            control_freq=10,
            horizon=500,
            cylinder_xrange=cylinder_xrange,  # 0.03
            cylinder_yrange=cylinder_yrange,  # 0.03
        )

        # set up observation and action spaces
        if self.goal_env:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
                    "achieved_goal": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
                    "desired_goal": spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self.action_space = spaces.Box(
            -1, 1, (2,), dtype=np.float32
        )  # delta x, delta y, rot z

    def reset(
        self,
        seed=None,
    ) -> Tuple[dict, dict]:
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
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")

        self.goal_pos_xy = self.unwrapped.env.sim.model.body("walls").pos[:2]
        self.goal_vel_xy = np.zeros(2)
        reset_done = False
        while not reset_done:
            overlap = False
            while not overlap:
                obs = self.env.reset()
                obj_pos = obs["object_pos"][:2]
                self.arm_init = np.random.uniform(
                    low=[self.arm_xy_range[0], self.arm_xy_range[1]],
                    high=[self.arm_xy_range[2], self.arm_xy_range[3]],
                )
                if np.linalg.norm(self.arm_init - obj_pos) > 0.05:
                    overlap = True

            obs = self.env.set_goal(
                goal_pos_xy=self.goal_pos_xy, goal_vel_xy=self.goal_vel_xy
            )

            # move the arm to the desired height , np.array([self.arm_height]
            # move xy first
            failed = False
            while (
                not reset_done
                and np.linalg.norm(obs["robot0_eef_pos"][:2] - self.arm_init) > 0.002
            ):
                obs, reward, done, info = self.env.step(
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
                if done:
                    failed = True
                    break
            # move xyz
            while (
                not failed
                and np.linalg.norm(
                    obs["robot0_eef_pos"]
                    - np.concatenate([self.arm_init, np.array([self.arm_height])])
                )
                > 0.002
            ):
                obs, reward, done, info = self.env.step(
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
                if done:
                    failed = True
                    break

            if not failed:
                reset_done = True

        self.obs_robotsuite = obs
        self.desired_goal = np.concatenate(
            (self.goal_pos_xy, self.goal_vel_xy), axis=0, dtype=np.float32
        )
        obs_dict = {}
        obj_goal_pos = obs["object_pos"][:2] - obs["goal_xy_pos"]
        obj_goal_vel = obs["object_vel"][:2] - obs["goal_xy_vel"]
        obs_dict["observation"] = np.concatenate(
            (
                obs["object_pos"][:2] - obs["robot0_eef_pos"][:2],
                obs["object_vel"][:2],
            ),
            axis=0,
            dtype=np.float32,
        )  # obj - eef
        obs_dict["achieved_goal"] = np.zeros(4, dtype=np.float32)  # 0
        obs_dict["desired_goal"] = np.concatenate(
            (obj_goal_pos, obj_goal_vel), axis=0, dtype=np.float32
        )  # obj - goal

        obs_gym = np.concatenate(
            (obs_dict["observation"], obs_dict["desired_goal"]),
            axis=0,
            dtype=np.float32,
        )

        self.cnt = 0
        self.obs = obs_gym
        self.next_obs = obs_gym

        if self.goal_env:
            self.next_obs = obs_dict
            return (obs_dict, info)
        else:
            self.next_obs = obs_gym
            return (obs_gym, info)

    def step(self, action) -> Tuple[dict, float, bool, bool, dict]:
        """
        Extends vanilla step() function call to return flattened observation instead of
        normal OrderedDict.
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        info = {}
        obs_dict = {}
        # extract action
        delta_x, delta_y = action[0], action[1]

        # extract rot z
        current_quat = self.obs_robotsuite["robot0_eef_quat"]
        correction_angles = euler.quat2euler(current_quat, axes="szyx")
        correction_actions = np.array(correction_angles) * self.rotation_scale
        # step the environment
        action_7 = np.array(
            [
                delta_x,
                delta_y,
                (self.arm_height - self.obs_robotsuite["robot0_eef_pos"][2])
                * self.position_scale,
                correction_actions[0],
                correction_actions[1],
                0,  # delta_rot_zs
                0,
            ]
        )
        obs, _, done, info = self.env.step(action_7)
        self.obs_robotsuite = obs
        obj_goal_pos = obs["object_pos"][:2] - obs["goal_xy_pos"]
        obj_goal_vel = obs["object_vel"][:2] - obs["goal_xy_vel"]
        obs_dict["observation"] = np.concatenate(
            (
                obs["object_pos"][:2] - obs["robot0_eef_pos"][:2],
                obs["object_vel"][:2],
            ),
            axis=0,
            dtype=np.float32,
        )  # obj - eef
        obs_dict["achieved_goal"] = np.zeros(4, dtype=np.float32)  # 0
        obs_dict["desired_goal"] = np.concatenate(
            (obj_goal_pos, obj_goal_vel), axis=0, dtype=np.float32
        )  # obj - goal

        info["is_success"] = self.env._check_success(
            self.pos_threshold, self.vel_threshold
        )
        terminated = info["is_success"]
        truncated = done

        obs_gym = np.concatenate(
            (obs_dict["observation"], obs_dict["desired_goal"]),
            axis=0,
            dtype=np.float32,
        )
        # calculate reward
        reward = self.compute_reward(obs_gym, info)
        self.cnt += 1
        self.obs = self.next_obs
        self.next_obs = obs_gym

        if self.goal_env:
            return (obs_dict, reward, terminated, truncated, info)
        else:
            return (obs_gym, reward, terminated, truncated, info)

    def compute_reward(
        self, obs, info, scale=1.0, success_scale=1.0
    ) -> Union[float, np.ndarray]:
        """
        Reward function for the environment.
        Inputs: (must be compatible with batched data)
            achieved_goal: achieved goal
            desired_goal: desired goal
            scale: scale of the reward
        Returns:
            reward: reward for the environment
        """
        if len(obs.shape) == 1:  # unbatched data
            obs_eef_xy = np.linalg.norm(obs[0:2])
            obs_eef_vel = np.linalg.norm(obs[2:4])
            obs_goal_xy = np.linalg.norm(obs[4:6])
            obs_goal_vel = np.linalg.norm(obs[6:8])
        elif len(obs.shape) == 2:  # batched data
            obs_eef_xy = np.linalg.norm(obs[:, 0:2], axis=1)
            obs_eef_vel = np.linalg.norm(obs[:, 2:4], axis=1)
            obs_goal_xy = np.linalg.norm(obs[:, 4:6], axis=1)
            obs_goal_vel = np.linalg.norm(obs[:, 6:8], axis=1)
            assert obs_goal_xy.shape == obs_goal_vel.shape
            assert len(obs_goal_xy.shape) == 1
        else:
            raise ValueError("Invalid goal shape!")
        # check if the object is at the goal location
        is_success = np.logical_and(
            obs_goal_xy < self.pos_threshold, obs_goal_vel < self.vel_threshold
        )
        reward = (
            -(obs_goal_xy + obs_goal_vel) - (obs_eef_xy + obs_eef_vel)
            if self.reward_shaping
            else 0
        )
        reward += is_success.astype(np.float32) * success_scale
        reward = np.clip(reward, -1, 1)
        reward *= scale
        return reward

    def render(self, mode="human") -> None:
        """
        Render the environment.
        """
        self.env.render()

    def close(self) -> None:
        """
        Close the environment.
        """
        self.env.close()