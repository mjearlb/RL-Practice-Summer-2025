from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
import gymnasium as gym
gym.pprint_registry() 
PYTHONWARNINGS="ignore::DeprecationWarning"
env_type = "Blackjack-v1"
config = (
    PPOConfig()
    .environment(env=env_type)
    .env_runners(
        num_env_runners=2, 
        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
)
config.api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)

from pprint import pprint
# Build the algorithm
algo = config.build_algo()

# Train it
for _ in range(10): 
    algo.train()
   #pprint(algo.train())

# ... and evaluate it.
pprint(algo.evaluate())

#
# Display a single run
import time

print("\n\nAttempting to display\n\n")
# Create the same environment
env = gym.make(env_type, render_mode="human")
obs, _ = env.reset()
done = False

# Run one episode using the trained agent
while not done: 
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    time.sleep(0.02)

# Close env
env.close()

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()