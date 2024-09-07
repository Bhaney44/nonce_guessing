from stable_baselines3 import PPO  
import numpy as np  
from env import CustomEnv

class Agent:
    def __init__(self, env):
        self.env = env  # Store the environment
        self.model = PPO("MlpPolicy", self.env, verbose=1)  # Initialize the PPO model with the environment

    def train(self):
        # Train the model
        self.model.learn(total_timesteps=10000)

    def test(self):
        obs = self.env.reset()
        for _ in range(len(self.env.target_list)):
            action, _states = self.model.predict(obs)
            obs, reward, done, info = self.env.step(action)
            print(f"Target: {info['current_target']}, Action: {action}, Reward: {reward}")
            self.env.render()
            if done:
                break
