from ray.rllib.algorithms.ppo import PPOConfig
import gymnasium as gym

config = (
    PPOConfig()
    .environment(env="CartPoleV1")
    .rollouts(num_rollour_workers=0)
)
trainer = config.build()

# Training loop
for i in range(10): 
    result = trainer.train()
    print(f"Iteration {i}: reward = {result['episode_reward_mean']:.2f}")

env = gym.make("CartPole-v1", render_mode="human")
obs, _ = env.reset()
done = False

while not done:
    action = trainer.compute_single_action(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()