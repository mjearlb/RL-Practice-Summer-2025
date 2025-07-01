import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

gym.pprint_registry() 

config = (
    PPOConfig()
    .training( # Set the training parameters
        gamma=0.9, # Discount factor. 
        lr=0.01, # alpha: how much the learner's knowledge is updated
        train_batch_size_per_learner=2000,
        num_epochs=10,
    )
    .environment("Pendulum-v1")
    .env_runners(
        num_env_runners=4
    )
)

# Build the algorithm
ppo = config.build_algo()

from pprint import pprint

for i in range(5):
    result = ppo.train()
    #pprint(ppo.train())

# Checkpoint (save) the trained ppo algorithm
checkpoint_path = ppo.save_to_path()

# Evaluate the trained model
config.evaluation(
    # Run one evaluation round every iteration.
    evaluation_interval=1,

    # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
    evaluation_num_env_runners=2,

    # Run evaluation for exactly 10 episodes. Note that because you have
    # 2 EnvRunners, each one runs through 5 episodes.
    evaluation_duration_unit="episodes",
    evaluation_duration=10,
)

# Rebuild the PPO, but with the extra evaluation EnvRunnerGroup
ppo_with_evaluation = config.build_algo()

for _ in range(5): 
    ppo_with_evaluation.train()
    #pprint(ppo_with_evaluation.train())


import numpy as np
import torch
import time

# Initialize an environment. Render mode determines how the environment is visualized. 
env = gym.make("Pendulum-v1", render_mode="human")

# Resetting will get the 1st observation of the environment as well as additional info. 
obs, info = env.reset()
episode_over = False
episode_return = 0.0

while not episode_over:
    env.render()

    action = ppo_with_evaluation.compute_single_action(obs) # get the trained PPO's next mvoe given current obs

    # Send the action to the environment for the next step.
    obs, reward, terminated, truncated, info = env.step(action)
    episode_return += reward

    # Perform env-loop bookkeeping.
    episode_return += reward
    episode_over = terminated or truncated
    time.sleep(0.02)

print(f"Reached episode return of {episode_return}.")
env.close()