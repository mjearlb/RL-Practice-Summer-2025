import gymnasium as gym

gym.pprint_registry() # Show all available env's

# Initialize an environment. Render mode determines how the environment is visualized. 
env = gym.make("CartPole-v1", render_mode="human")

# Resetting will get the 1st observation of the environment as well as additional info. 
observation, info = env.reset()

for _ in range(500):
    env.render() # Render the environment
    action = env.action_space.sample() # take a random action
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
