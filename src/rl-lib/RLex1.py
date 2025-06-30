from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations

# Configure the algorithm.
# The config defines the RL env and any other needed settings/param's
config = (
    PPOConfig()
    .environment("Taxi-v3") # This can be any Gymnasiu registered env
    .env_runners(
        num_env_runners=2,
        # Observations are discrete (ints) -> We need to flatten (one-hot) them.
        env_to_module_connector=lambda env: FlattenObservations(),
    )
    .evaluation(evaluation_num_env_runners=1)
)

# Next, we need to build and train the algorithm
from pprint import pprint

# Build the algorithm.
algo = config.build_algo()

# Train it for 5 iterations ...
for _ in range(5):
    #pprint(algo.train())
    algo.train()

# Done training, so now we can evaluate it and release its resources:
pprint(algo.evaluate())

# Release the algo's resources (remote actors, like EnvRunners and Learners).
algo.stop()