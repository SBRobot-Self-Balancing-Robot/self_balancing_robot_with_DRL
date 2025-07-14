import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import time
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env.self_balancing_robot_env import SelfBalancingRobotEnv

def make_env():
    """
    Crea un'istanza dell'ambiente SelfBalancingRobotEnv.
    """
    def _init():
        env = gym.make("SelfBalancingRobot-v0")
        env = Monitor(env)
        check_env(env, warn=True)
        return env
    return _init

if __name__ == "__main__":
    MODEL = PPO
    MODEL_FILE = "a"
    # Wrappa per l'algoritmo
    vec_env = SubprocVecEnv([make_env() for _ in range(50)])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    # Load model if it exists
    try:
        model = MODEL.load(f"./recordings/{MODEL_FILE}", env=vec_env)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found, starting training from scratch.")
        model = None

    model = MODEL("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=2_000_000, progress_bar=True)
    model.save(f"./recordings/new_reward_{timestamp}")

    # Test
    env = make_env()()
    env.reset()
    obs, _ = env.reset()
    for _ in range(10000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
