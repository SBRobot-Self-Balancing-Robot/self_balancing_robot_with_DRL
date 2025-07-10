from typing import Optional
import numpy as np
import gymnasium as gym
import mujoco
from mujoco import MjModel, MjData
import os

class SelfBalancingRobotEnv(gym.Env):

    def __init__(self, environment_path: str = "models/scene.xml"):

        # Initialize the environment
        super().__init__()
        full_path = os.path.abspath(environment_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        self.model = MjModel.from_xml_path(full_path)
        self.data = MjData(self.model)
        print("qpos names:", self.model.names(self.model.qposadr, self.model.nq))

        # Action and observation spaces
        # Observation space: inclination angle, angular velocity, linear position and velocity (could be added last action)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Action space: torque applied to the wheels
        self.action_limit = 10.0
        self.action_space = gym.spaces.Box(low=np.array([-self.action_limit, -self.action_limit]), high=np.array([self.action_limit, self.action_limit]), dtype=np.float32)

        # If last action is needed, uncomment the following line
        # self.last_tau = 0.0

    def _get_obs(self):
        pass  # Placeholder for observation retrieval logic

    def _get_info(self):
        pass # Placeholder for additional information retrieval logic

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # Seed the random number generator
        super().reset(seed=seed)

        # Place the agent in the origin with a small random perturbation    

def main():
    env = SelfBalancingRobotEnv()

if __name__ == "__main__":
    main()