import typing as T
import gymnasium as gym
import numpy as np
from src.env.self_balancing_robot_env.self_balancing_robot_env import SelfBalancingRobotEnv

class RewardWrapper(gym.Wrapper):
    """
    Wrapper for the SelfBalancingRobotEnv to modify the reward structure.
    """
    def __init__(self, env: SelfBalancingRobotEnv):
        super().__init__(env)
        self.reward_calculator = RewardCalculator()

    def step(self, action):
        """
        Executes one step in the environment with the given action.
        
        Args:
            action: The action to take in the environment.
        Returns:
            obs: The observation after taking the action.
            reward: The modified reward after taking the action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode was truncated.
            info: Additional information from the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        torque_penalty = 0.01 * np.sum(np.square(action))
        reward -= torque_penalty

        if not terminated and not truncated:
            reward += 1.0
        
        reward += self.reward_calculator.compute_reward(self.env)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial observation and additional info.
        """
        # reset dello stato interno della reward (anchor, contatori, ecc.)
        self.reward_calculator.reset()
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    """
    def __init__(self, 
                 alpha_yaw_displacement_penalty=0.3, 
                 alpha_pos_displacement_penalty=0.005,
                 alpha_linear_velocity_penalty=0.0001, 
                 alpha_torque_penalty=0.025, 
                 weight_fall_penalty=100.0,
                 
                 alpha_yaw_magnitude_penalty=0.3,
                 anchor_lock_steps=10,
                 yaw_settle_thresh=0.08,
                 lin_settle_thresh=0.03,
                 yaw_disp_settle_thresh=0.01,
                 torque_persist_penalty_scale=0.02,
                 ang_vel_settle_thresh=0.12
                 ):
        self.alpha_yaw_displacement_penalty = alpha_yaw_displacement_penalty
        self.alpha_pos_displacement_penalty = alpha_pos_displacement_penalty
        self.alpha_linear_velocity_penalty = alpha_linear_velocity_penalty
        self.alpha_torque_penalty = alpha_torque_penalty
        self.weight_fall_penalty = weight_fall_penalty

        self.alpha_yaw_magnitude_penalty = alpha_yaw_magnitude_penalty
        self.anchor_lock_steps = anchor_lock_steps
        self.yaw_settle_thresh = yaw_settle_thresh
        self.lin_settle_thresh = lin_settle_thresh
        self.yaw_disp_settle_thresh = yaw_disp_settle_thresh
        self.ang_vel_settle_thresh = ang_vel_settle_thresh
        self.torque_persist_penalty_scale = torque_persist_penalty_scale
        self.alpha_pitch_penalty = 5.0
        self.alpha_velocity_penalty = 2.0
        self.alpha_position_penalty = 1.0
        self.alpha_yaw_penalty = 0.5

        # internal state for the reward calculator
        self.negative_sign = 0
        self.positive_sign = 0
        self.anchor_position = None
        self.stable_counter = 0

    def compute_reward(self, env) -> float:
        """
        Compute the reward for the current step, focused on self-balancing and staying still.

        Reward scenarios:
            1) In equilibrium (yaw < threshold) but moving too fast: negative reward based on velocity intensity
            2) In equilibrium (yaw < threshold) with acceptable velocity: positive reward based on yaw and velocity
            3) Out of equilibrium: negative reward to encourage balancing

        Args:
            env: environment instance of SelfBalancingRobotEnv.

        Returns:
            float: computed reward value.
        """
        # Initialize tracking variables if needed
        if not hasattr(self, 'bad_motion_counter'):
            self.bad_motion_counter = 0
        if not hasattr(self, 'good_behavior_counter'):
            self.good_behavior_counter = 0
        pitch = abs(env.pitch)
        lin_vel_norm = np.linalg.norm(env.linear_vel[:2])  # solo x,y
        pos_deviation = np.linalg.norm([env.x, env.y])

        # Inclination penalties
        pitch_penalty = self.alpha_pitch_penalty * pitch**2

        # Movement penalties
        velocity_penalty = self.alpha_velocity_penalty * lin_vel_norm**2
        position_penalty = self.alpha_position_penalty * pos_deviation**2

        # Stability bonus
        stability_bonus = np.tanh(self.good_behavior_counter / 20.0)

        # Yaw penalty
        yaw_rate = abs(env.body_ang_vel[2])
        yaw_penalty = self.alpha_yaw_penalty * yaw_rate**2

        # Final reward
        reward = (
            + 1.0 * stability_bonus
            - pitch_penalty
            - velocity_penalty
            - position_penalty
            - yaw_penalty
        )
        
        if pitch < 0.05 and lin_vel_norm < 0.05 and np.linalg.norm(env.body_ang_vel) < 0.05:
            reward += 2.0  # type: ignore

        return reward

    def reset(self):
        """
        Reset of the internal state of the reward calculator.
        """
        self.negative_sign = 0
        self.positive_sign = 0
        self.anchor_position = None
        self.stable_counter = 0
        self.bad_motion_counter = 0
        self.good_behavior_counter = 0

    def _check_episode_end(self, env, yaw_displacement, pos_penalty, linear_penalty, yaw_magnitude_penalty, reward): 
        if env._is_truncated():
            reward -= (self.weight_fall_penalty + 10 * yaw_displacement)
        elif env._is_terminated():
            terminal_bonus = 500.0 # * pos_penalty * linear_penalty * yaw_magnitude_penalty
            reward += terminal_bonus # - 3 * (self.positive_sign + self.negative_sign)
        return reward

    def _check_torque_persistence(self, env, yaw_displacement_penalty, pos_penalty, linear_penalty, yaw_magnitude_penalty):
        if env.torque_l * env.torque_r > 0:
            if np.sign(env.torque_l) > 0:
                self.positive_sign += 1.0
                self.negative_sign = 0
            else:
                self.negative_sign += 1.0
                self.positive_sign = 0
        else:
            self.positive_sign = 0
            self.negative_sign = 0
        persistence = min(self.negative_sign + self.positive_sign, 50)
        persistence_penalty = self.torque_persist_penalty_scale * persistence

        reward = (yaw_displacement_penalty * yaw_magnitude_penalty) * linear_penalty * pos_penalty
        reward -= persistence_penalty
        
        return reward

    def _kernel(self, x: float, alpha: float) -> float:
        """
        Gaussian kernel function for reward computation.
        Args:
            x (float): The input value.
            alpha (float): The bandwidth parameter for the Gaussian kernel. 
        Returns:
            float: The value of the Gaussian kernel at x.
        """
        return np.exp(-(x**2)/alpha)

    def _compute_velocity_penalty(self, velocity: float) -> float:
        """
        Compute velocity penalty that increases non-linearly with velocity.
        
        Args:
            velocity (float): Linear velocity magnitude
            
        Returns:
            float: Penalty value (always positive)
        """
        # Exponential penalty that grows rapidly for high velocities
        return np.exp(velocity / self.lin_settle_thresh) - 1.0
    
    def _compute_imbalance_penalty(self, yaw_angle: float) -> float:
        """
        Compute penalty for being out of equilibrium.
        
        Args:
            yaw_angle (float): Absolute yaw angle
            
        Returns:
            float: Penalty value (always positive)
        """
        # Quadratic penalty that grows with angle deviation from equilibrium
        normalized_angle = yaw_angle / self.yaw_settle_thresh
        return 5.0 * (normalized_angle ** 2)
