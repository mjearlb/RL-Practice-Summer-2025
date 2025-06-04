import gymnasium as gym

# natural is whether to give extra reward for natural blackjack
# sab is whether to follow Sutton and Barto rules. 
gym.make('Blackjack-v1', natural=False, sab=False)