# https://www.youtube.com/watch?v=ZhoIgo3qqLU

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False): 
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if (is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # inits a 64 x 4 array
    else: 
        f = open('frozen_lake8x8.pk1', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha
    discount_factor_g = 0.9 # gamma

    epsilon = 1 # 1 = 100% random actions taken
    epsilon_decay = 0.0001 # Rate at which epsilon decays (stops taking random actions)
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0] 
        terminated = False
        truncated = False

        while (not terminated and not truncated): 
            if is_training and rng.random() < epsilon: 
                action = env.action_space.sample() 
            else: 
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)

            # Q-Learning algorithm
            if is_training: 
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action]
                ) 

            state = new_state

        epsilon = max(epsilon - epsilon_decay, 0)

        # Reduce the learning rate when exploration has finished
        if (epsilon == 0): 
            learning_rate_a = 0.0001

        if (reward == 1): 
            rewards_per_episode[i] = 1
    
    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes): 
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen-lake-8x8.png')

    if (is_training): 
        f = open("frozen_lake8x8.pk1","wb")
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__': 
    # To train
    run(150000)

    # To see how the agent does after training 
    #run(1, is_training=False, render=True)