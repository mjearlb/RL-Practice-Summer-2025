import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

# Initialize an environment. Render mode determines how the environment is visualized. 
env = gym.make("Blackjack-v1", render_mode="human")

# Resetting will get the 1st observation of the environment as well as additional info. 
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated

env.close()
