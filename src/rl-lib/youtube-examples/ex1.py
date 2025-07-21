# https://www.youtube.com/watch?v=HteW2lfwLXM

# Step 1: init ray
import ray

ray.init()

# Step 2: run exp to solve RL probs
# need 4 things
# RL env (like cartpole
# RL algo (like PPO)
# configuration (algo config, env configs, etc.)
# exp runner (like tune)
from ray import tune

# as 1st arg, provide algo. 
#tune.run("PPO", 
#         config={"env": "CartPole-v1", 
#                 # other configs. default if none given
#                 # default configs work well for most env's
#                }
#        )

# step 3: running an exp
# Train -> Evaluation -> Train -> Evaluation... 
# Change training time by changing evaluation_interval in config
# Change evaluation time by changing evaluation_duration in configs

tune.run("PPO", 
         config={"env": "CartPole-v1", 
                 "evaluation_interval": 2, 
                 "evaluation_duration": 20,
                 "num_gpus": 2,
                }
        )