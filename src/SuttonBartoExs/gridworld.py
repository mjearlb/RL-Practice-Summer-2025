from typing import Optional
import numpy as np
import gymnasium as gym

# Following this tutorial: https://gymnasium.farama.org/introduction/create_custom_env/

# Step 1: Environmental __init__
class GridWorldEnv(gym.Env): 
    def __init__(self, size: int = 5): 
        # The size of the grid world (default == 5)
        self.size = size
        
        # Init the positions. Will be randomly set in reset() 
        # Let -1, -1 be the "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Define what the agent can observe
        # Dict space gives us structured & human-readable observation
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape-(2,), dtype=int), # [x, y] coordinates
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int), # [x, y] coordinates
            }
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Discrete(4)

        # Map action numbers to actual movements on the grid
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: np.array([1,0]), # Move Right (positive x)
            1: np.array([0,1]), # Move Up (positive y)
            2: np.array([-1,0]), # Move Left (negative x)
            3: np.array([0,-1]), # Move Down (negative y)
        }
# Step 2: Constructing Observations
def _get_obs(self): 
    """"Convert internal state to observation format
    
    Returns: 
        dict: Observation with agent and target positions
    """
    return {"agent": self._agent_location, "target": self._target_location}

def _get_info(self): 
    """Compute auxiliary information for debugging. 
    
    Returns: 
        dict: Info with distance between agent and target
    """
    return {
        "distance": np.linalg.norm(
            self._agent_location - self._target_location, ord=1
        )
    }