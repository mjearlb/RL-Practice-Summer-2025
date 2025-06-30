from ray import tune

tune.run("PPO", 
         config = {"env": "CartPole-v1", 
                   # other configurations go here
                   # if none supplied, default will be used
                   }
            )