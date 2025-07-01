from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
import gymnasium as gym
import torch
import time

# This example trains an RL-Lib PPO 
# algo on CartPole-v1. It then displays 
# an example run of CartPole-v1 using 
# Gymnasium's human renderer. 
# 
# This uses the new API stack from RL-Lib. 

# Set the config settings
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .env_runners(num_env_runners=4)
    .training(
        gamma=0.95,
        lr=0.05,
        train_batch_size_per_learner=2000,
        num_epochs=10,
    )
    # `FlattenObservations` converts int observations to one-hot.
    .env_runners(env_to_module_connector=lambda env: FlattenObservations())
)

# Create the algo instance
ppo = config.build_algo()

for i in range(20): 
    result = ppo.train()

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset()

episode_over = False

while not episode_over: 
    module = ppo.get_module("default_policy") # Get the module
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0) # Prepare the observation
    out = module.forward_inference({"obs": obs_tensor}) # Forward pass

    #action = torch.argmax(out["action_dist_inputs"], dim=1).item()
    action = torch.argmax(out["action_dist_inputs"][0]).item()
    obs, reward, terminated, truncated, info = env.step(action)
    episode_over = terminated or truncated
    env.render()
    time.sleep(0.02)

env.close() 