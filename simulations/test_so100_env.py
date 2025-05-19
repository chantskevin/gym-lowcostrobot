import gymnasium
import numpy as np
import gym_lowcostrobot
env = gymnasium.make('SO101ReachFixedTarget-v0', render_mode="human", action_mode="joint")
env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    env.step(action)
    env.render()


env.close()
