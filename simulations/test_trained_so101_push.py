import os
import time
import argparse
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test a trained SO101 push cube model')
parser.add_argument('--delay', type=float, default=0.05,
                    help='Delay between steps in seconds (default: 0.05)')
parser.add_argument('--steps', type=int, default=5000,
                    help='Maximum number of steps to run (default: 5000)')
args = parser.parse_args()

# Path to the trained model
models_dir = "logs/so101_push_cube/models"
model_path = os.path.join(models_dir, "so101_push_cube_final")
vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")

# Load the trained model
model = PPO.load(model_path)

# Create a test environment with rendering
env = gym.make('SO101PushCube-v0',
              observation_mode="state",
              action_mode="joint",
              reward_type="dense",
              render_mode="human")

# Run the trained agent
obs, info = env.reset()
total_reward = 0

print(f"Running test with {args.delay} second delay between steps")
print("Press Ctrl+C to exit")

try:
    for i in range(args.steps):
        # Get the model's action
        action, _states = model.predict(obs, deterministic=True)

        # Execute the action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Add delay to slow down visualization
        time.sleep(args.delay)

        # Print some information
        if i % 50 == 0:
            print(f"Step: {i}, Reward: {reward:.4f}, Total Reward: {total_reward:.4f}")
            if 'is_success' in info:
                print(f"Success: {info['is_success']}")

        # Reset if episode is done
        if terminated or truncated:
            print(f"Episode finished after {i+1} steps with total reward: {total_reward:.4f}")
            obs, info = env.reset()
            total_reward = 0
except KeyboardInterrupt:
    print("Test interrupted by user")
finally:
    env.close()
    print("Test completed")
