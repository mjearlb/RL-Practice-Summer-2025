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
# These helper functions will be used by Env.reset() and Env.step()
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

# Step 3: Reset function
# Starts a new episode. Has 2 optional param's: seed for 
# reproducible seed generation and options for additional
# configs. 
# Our reset() will randomly place the agent and target 
# in the gridworld. We will return initial obs and info as
# a tuple. 
def reset(self, seed: Optional[int] = None, options: Optional[dict] = None): 
    """Start a new episode
    
    Args: 
        seed: Random seed for reproducible episodes
        options: Additional configuration (unused in this example)
        
    Returns: 
        tuple: (observation, info) for the initial state
    """
    # IMPORTANT: Must call this first to seed the random number generator
    super().reset(seed=seed)

    # Randomly place the agent anywhere in the grid
    self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

    # Randomly place the target, ensuring it's different from the agent position
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
    
    observation = self._get_obs()
    info = self._get_info()

    return observation, info