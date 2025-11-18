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

        if not terminated and not truncated:
            reward += 1.0 # type: ignore
        
        reward += self.reward_calculator.compute_reward(self.env) # type: ignore

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
                 weight_fall_penalty=100.0,
                 
                 alpha_yaw_magnitude_penalty=0.3,
                 anchor_lock_steps=10,
                 yaw_settle_thresh=0.08,
                 lin_settle_thresh=0.03,
                 yaw_disp_settle_thresh=0.01,
                 ang_vel_settle_thresh=0.12
                 ):
        self.alpha_yaw_displacement_penalty = alpha_yaw_displacement_penalty
        self.alpha_pos_displacement_penalty = alpha_pos_displacement_penalty
        self.alpha_linear_velocity_penalty = alpha_linear_velocity_penalty
        self.weight_fall_penalty = weight_fall_penalty

        self.alpha_yaw_magnitude_penalty = alpha_yaw_magnitude_penalty
        self.anchor_lock_steps = anchor_lock_steps
        self.yaw_settle_thresh = yaw_settle_thresh
        self.lin_settle_thresh = lin_settle_thresh
        self.yaw_disp_settle_thresh = yaw_disp_settle_thresh
        self.ang_vel_settle_thresh = ang_vel_settle_thresh
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
        Compute the reward for the current step using the current env API.

        Usa:
          - env.pitch
          - env.angular_velocity (array)
          - env.right_wheel_velocity, env.left_wheel_velocity
          - env.data.qpos[:2] per la posizione (x,y)
        """
        # Ensure counters exist
        if not hasattr(self, 'bad_motion_counter'):
            self.bad_motion_counter = 0
        if not hasattr(self, 'good_behavior_counter'):
            self.good_behavior_counter = 0

        # State from environment (fallback a 0 se non presente)
        pitch = abs(getattr(env, "pitch", 0.0))

        # Use wheel angular velocities as proxy della velocità lineare
        right_w = getattr(env, "right_wheel_velocity", 0.0)
        left_w = getattr(env, "left_wheel_velocity", 0.0)
        wheel_vels = np.array([right_w, left_w], dtype=float)
        lin_vel_norm = float(np.mean(np.abs(wheel_vels)))  # rad/s proxy per velocità

        # Position deviation from origin using qpos if disponibile
        if hasattr(env, "data") and hasattr(env.data, "qpos"):
            pos = np.array(env.data.qpos[:2], dtype=float)
        else:
            pos = np.zeros(2, dtype=float)
        pos_deviation = np.linalg.norm(pos)

        # Penalità per inclinazione (pitch)
        pitch_penalty = self.alpha_pitch_penalty * (pitch ** 2)

        # Penalità movimento (velocità e posizione)
        velocity_penalty = self.alpha_velocity_penalty * (lin_vel_norm ** 2)
        position_penalty = self.alpha_position_penalty * (pos_deviation ** 2)

        # Yaw rate penalty (usa angular_velocity z)
        ang_vel = getattr(env, "angular_velocity", np.zeros(3, dtype=float))
        yaw_rate = abs(ang_vel[2]) if len(ang_vel) > 2 else 0.0
        yaw_penalty = self.alpha_yaw_penalty * (yaw_rate ** 2)

        # Stability bonus basato sul contatore di buon comportamento
        stability_bonus = np.tanh(self.good_behavior_counter / 20.0)

        # Incremento o decremento dei contatori di comportamento
        stable_condition = (
            (pitch < self.yaw_settle_thresh) and
            (lin_vel_norm < self.lin_settle_thresh) and
            (np.linalg.norm(ang_vel) < self.ang_vel_settle_thresh)
        )
        if stable_condition:
            self.good_behavior_counter += 1
        else:
            self.bad_motion_counter += 1

        # Composizione reward
        reward = (
            + 1.0 * stability_bonus
            - pitch_penalty
            - velocity_penalty
            - position_penalty
            - yaw_penalty
        )

        # Bonus addizionale se molto stabile e quasi fermo
        if (pitch < 0.05) and (lin_vel_norm < 0.05) and (np.linalg.norm(ang_vel) < 0.05):
            reward += 2.0

        return float(reward)

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
