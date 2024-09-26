import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self, target_list=None):
        super(CustomEnv, self).__init__()
        # Define action and observation space
        # Example: 2 possible actions
        self.action_space = spaces.Discrete(2)  
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        
        # Target list to track specific goals for the environment
        self.target_list = target_list if target_list is not None else [np.random.rand(4) for _ in range(10)]
        self.current_target_idx = 0

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_target_idx = 0
        return np.zeros(self.observation_space.shape)

    def step(self, action):
        # Implement how the environment reacts to actions
        obs = np.random.random(self.observation_space.shape)
        reward = 1.0
        done = self.current_target_idx >= len(self.target_list) - 1

        # Move to the next target after a step
        if not done:
            self.current_target_idx += 1

        info = {"current_target": self.target_list[self.current_target_idx]}
        return obs, reward, done, info

    def render(self):
        # Render the environment to the console
        print(f"Current Target Index: {self.current_target_idx}")
        print(f"Current Target: {self.target_list[self.current_target_idx]}")
