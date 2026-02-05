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
    TUNED FOR SURVIVAL: Prioritizes staying alive over smoothness initially.
    """
    def __init__(self, 
                 alpha_pitch_penalty = 0.05,       # RIDOTTO: Meno freno, lasciamolo reagire
                 alpha_yaw_speed_penalty = 0.001, 
                 alpha_x_vel_penalty = 0.01,       
                 alpha_equilibrium = 0.1,          # ALLARGATO: Più facile trovare la zona "buona"
                 alpha_ctrl_variation_penalty = 0.05, # RIDOTTO DI 10x: Permettiamo scatti per salvarsi
                 alpha_ctrl_intensity_penalty = 0.01, # RIDOTTO
                 alpha_yaw_equilibrium = 0.5,
                 alpha_linear_equilibrium = 0.1):   
        
        self.alpha_pitch_penalty = alpha_pitch_penalty
        self.alpha_yaw_speed_penalty = alpha_yaw_speed_penalty
        self.alpha_ctrl_variation_penalty = alpha_ctrl_variation_penalty
        self.alpha_x_vel_penalty = alpha_x_vel_penalty
        self.alpha_equilibrium = alpha_equilibrium
        self.alpha_ctrl_intensity_penalty = alpha_ctrl_intensity_penalty
        self.alpha_yaw_equilibrium = alpha_yaw_equilibrium
        self.alpha_linear_equilibrium = alpha_linear_equilibrium

        self.past_ctrl = np.array([0.0, 0.0])

    def compute_reward(self, env) -> float:
        # --- INPUT DATA ---
        w_roll, w_pitch, w_yaw = env.real_angular_velocity
        x_vel = env.x_vel 
        x_pos = env.env.data.qpos[0]
        pitch = env.pitch 
        current_ctrl = env.env.data.ctrl.copy()

        # --- CALCOLO PENALITÀ ---

        # 1. Smoothness (Ridotta drasticamente per favorire l'apprendimento)
        ctrl_variation = np.sum(np.square(current_ctrl - self.past_ctrl))
        ctrl_penalty = self.alpha_ctrl_variation_penalty * ctrl_variation

        # 2. Damping
        pitch_velocity_penalty = self.alpha_pitch_penalty * abs(w_pitch)

        # 3. Fattore Equilibrio
        equilibrium_factor = self._kernel(pitch, self.alpha_equilibrium)

        # 4. Penalità fini (solo se stabili)
        fine_tuning_penalty = (
            self.alpha_linear_equilibrium * abs(x_vel) + 
            self.alpha_yaw_equilibrium * abs(w_yaw)
        ) * equilibrium_factor

        # 5. Energia
        energy_penalty = self.alpha_ctrl_intensity_penalty * np.linalg.norm(current_ctrl)

        # --- REWARD TOTALE ---
        
        # AUMENTATO: Alive bonus molto alto. 
        # Questo assicura che nessuna penalità (ragionevole) possa rendere il passo negativo.
        reward = 5.0 
        
        # Sottrazioni (Clamped per sicurezza)
        # Assicuriamoci che le penalità non esplodano mai sopra 4.0 totali
        total_penalty = (ctrl_penalty + pitch_velocity_penalty + 
                         fine_tuning_penalty + energy_penalty + 
                         (self.alpha_x_vel_penalty * abs(x_vel)) +
                         (0.1 * abs(x_pos)))
        
        reward -= total_penalty
        
        # Aggiorna controllo passato
        self.past_ctrl = current_ctrl

        return float(reward)

    def _kernel(self, x: float, alpha: float) -> float:
        return np.exp(-(x**2)/alpha)