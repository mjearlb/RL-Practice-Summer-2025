from ray.rllib.algorithms.ppo import PPOConfig

# Configure.
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .training(
        train_batch_size_per_learner=2000,
        lr=0.0004,
    )
)

# Build the Algorithm.
algo = config.build()

# Train for one iteration, which is 2000 timesteps (1 train batch).
print(algo.train())