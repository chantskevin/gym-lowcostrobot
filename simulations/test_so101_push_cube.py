import gymnasium
import numpy as np
import gym_lowcostrobot

# Create the environment
env = gymnasium.make('SO101PushCube-v0', render_mode="human", action_mode="joint")
observation, info = env.reset()

# Run the simulation for 1000 steps
for _ in range(1000):
    # Sample a random action
    action = env.action_space.sample()

    # Step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # Reset if the episode is done
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment
env.close()
