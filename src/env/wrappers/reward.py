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
        
        reward += self.reward_calculator.compute_reward(self.env, action)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Args:
            **kwargs: Optional arguments for the reset.
        Returns:
            Initial observation and additional info.
        """
        return self.env.reset(**kwargs)

class RewardCalculator:
    """
    Class to compute the reward for the SelfBalancingRobotEnv.
    """
    def __init__(self, alpha_yaw_displacement_penalty=0.3, alpha_pos_displacement_penalty=0.1,
                 alpha_linear_velocity_penalty=0.001, alpha_torque_penalty=0.5, weight_fall_penalty=100.0):
        self.alpha_yaw_displacement_penalty = alpha_yaw_displacement_penalty
        self.alpha_pos_displacement_penalty = alpha_pos_displacement_penalty
        self.alpha_linear_velocity_penalty = alpha_linear_velocity_penalty
        self.alpha_torque_penalty = alpha_torque_penalty
        self.weight_fall_penalty = weight_fall_penalty

    def compute_reward(self, env, action: np.ndarray) -> float:
        """
        Compute the reward for the current step, focused on self-balancing and staying still.

        Args:
            env: L'istanza dell'ambiente.
            action (np.ndarray): L'azione presa nell'ambiente (es. comandi motori).

        Returns:
            float: La reward calcolata.
        """
        yaw_displacement = abs(env.yaw - env.last_yaw)
        yaw_displacement_penalty = self._kernel(yaw_displacement, alpha=self.alpha_yaw_displacement_penalty)
        if env.count_yaw == 10:
            env.last_yaw = env.yaw
            env.count_yaw = 0
        env.count_yaw += 1

        if abs(env.yaw) < 0.08:
            env.last_position = env.data.qpos[:2].copy()

        position = np.array([env.x, env.y])
        pos_displacement = np.linalg.norm(position - env.last_position)
        pos_displacement_penalty = self._kernel(float(pos_displacement), alpha=self.alpha_pos_displacement_penalty)

        linear_norm = np.linalg.norm([env.linear_vel[0], env.linear_vel[1]])
        linear_penalty = self._kernel(float(linear_norm), alpha=self.alpha_linear_velocity_penalty)

        torques = env.data.ctrl
        torque_norm = np.linalg.norm(torques)
        torque_penalty = self._kernel(float(torque_norm), alpha=self.alpha_torque_penalty)

        if pos_displacement <= 0.1:
            reward = yaw_displacement_penalty * torque_penalty * linear_penalty * pos_displacement_penalty
        else:
            reward = yaw_displacement_penalty * torque_penalty * pos_displacement_penalty
        if env._is_truncated():
            reward -= (self.weight_fall_penalty + 10 * yaw_displacement + 10 * pos_displacement)
        elif env._is_terminated() and (pos_displacement < 0.1):
            reward += 500  # Bonus for staying still at the end of the episode
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
