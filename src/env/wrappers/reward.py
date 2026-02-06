import numpy as np
import typing as T
import gymnasium as gym
from scipy.spatial.transform import Rotation as R
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

        reward += self.reward_calculator.compute_reward(self.env)

        if terminated and not truncated:
            reward += 10 
        elif truncated:
            reward -= 100 * (1 - (self.env.env.data.time / self.env.env.max_time))

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
    HEADING LOCK VERSION: Keeps the robot facing a specific direction (Yaw=0).
    """
    def __init__(self, 
                 alpha_pitch_penalty = 0.05,       
                 alpha_yaw_speed_penalty = 0.5,    # Damping: frena la rotazione
                 alpha_heading_penalty = 2.0,      # NUOVO: "Bussola". Punisce se non guarda avanti.
                 alpha_x_vel_penalty = 0.01,       
                 alpha_equilibrium = 0.1,          
                 alpha_ctrl_variation_penalty = 0.05, 
                 alpha_ctrl_intensity_penalty = 0.01, 
                 alpha_linear_equilibrium = 0.1):   
        
        self.alpha_pitch_penalty = alpha_pitch_penalty
        self.alpha_yaw_speed_penalty = alpha_yaw_speed_penalty
        self.alpha_heading_penalty = alpha_heading_penalty # Peso della direzione
        self.alpha_ctrl_variation_penalty = alpha_ctrl_variation_penalty
        self.alpha_x_vel_penalty = alpha_x_vel_penalty
        self.alpha_equilibrium = alpha_equilibrium
        self.alpha_ctrl_intensity_penalty = alpha_ctrl_intensity_penalty
        self.alpha_linear_equilibrium = alpha_linear_equilibrium

        self.past_ctrl = np.array([0.0, 0.0])

    def compute_reward(self, env) -> float:
        # --- INPUT DATA ---
        w_roll, w_pitch, w_yaw = env.real_angular_velocity
        x_vel = env.x_vel 
        x_pos = env.env.data.qpos[0]
        pitch = env.pitch 
        
        # IMPORTANTE: Assicurati che env.yaw sia disponibile e sia 0.0 quando guarda avanti
        current_yaw_angle = env.yaw  
        
        current_ctrl = env.env.data.ctrl.copy()

        # --- CALCOLO PENALITÀ ---

        # 1. Heading Penalty (Direzione)
        # Punisce la deviazione dall'angolo 0. 
        # Questo costringerà il robot a correggere attivamente la rotta se ruota.
        heading_penalty = self.alpha_heading_penalty * abs(current_yaw_angle)

        # 2. Yaw Speed (Damping)
        # Serve ancora per evitare che corregga la direzione troppo velocemente (oscillazioni)
        yaw_speed_penalty = self.alpha_yaw_speed_penalty * abs(w_yaw)

        # 3. Smoothness
        # Tenuto basso per permettere le correzioni di rotta e equilibrio
        ctrl_variation = np.sum(np.square(current_ctrl - self.past_ctrl))
        ctrl_penalty = self.alpha_ctrl_variation_penalty * ctrl_variation

        # 4. Damping Pitch
        pitch_velocity_penalty = self.alpha_pitch_penalty * abs(w_pitch)

        # 5. Fattore Equilibrio (Kernel)
        equilibrium_factor = self._kernel(pitch, self.alpha_equilibrium)

        # 6. Penalità fini
        # Lasciamo un po' di libertà lineare per bilanciarsi
        fine_tuning_penalty = (
            self.alpha_linear_equilibrium * abs(x_vel)
        ) * equilibrium_factor

        # 7. Energia
        energy_penalty = self.alpha_ctrl_intensity_penalty * np.linalg.norm(current_ctrl)

        # --- REWARD TOTALE ---
        
        reward = 5.0 
        
        # Somma penalità
        total_penalty = (ctrl_penalty + 
                         heading_penalty +     # <--- Il pilastro della direzione
                         yaw_speed_penalty +
                         pitch_velocity_penalty + 
                         fine_tuning_penalty + 
                         energy_penalty + 
                         (self.alpha_x_vel_penalty * abs(x_vel)) +
                         (0.2 * abs(x_pos))) 
        
        reward -= total_penalty
        
        self.past_ctrl = current_ctrl

        return float(reward)

    def _kernel(self, x: float, alpha: float) -> float:
        return np.exp(-(x**2)/alpha)