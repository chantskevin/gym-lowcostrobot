import os
import time
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Set device to MPS if available (for Apple Silicon Macs)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create log directory
log_dir = "logs/so101_lift_cube/"
os.makedirs(log_dir, exist_ok=True)
models_dir = os.path.join(log_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Create the environment
def make_env():
    env = gym.make('SO101LiftCube-v0',
                  observation_mode="state",  # Use state observations for faster training
                  action_mode="joint",       # Control the joints directly
                  reward_type="dense",       # Dense rewards help with learning
                  render_mode=None)          # No rendering during training for speed
    return env

# Create a vectorized environment (can be used for parallel training)
env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Set up callbacks for saving models and evaluating performance
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=models_dir,
    name_prefix="so101_lift_cube_model"
)

# Create and train the agent
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    device=device,
)

# Train the agent
total_timesteps = 500000
model.learn(
    total_timesteps=total_timesteps,
    callback=checkpoint_callback,
    tb_log_name="PPO"
)

# Save the final model
model.save(os.path.join(models_dir, "so101_lift_cube_final"))
env.save(os.path.join(models_dir, "vec_normalize.pkl"))

print("Training complete!")

# Create a function to test the trained model
def test_model():
    # Load the trained model
    model = PPO.load(os.path.join(models_dir, "so101_lift_cube_final"))

    # Create a test environment with rendering
    test_env = gym.make('SO101LiftCube-v0',
                       observation_mode="state",
                       action_mode="joint",
                       reward_type="dense",
                       render_mode="human")

    # Test the model
    obs, info = test_env.reset()

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        if terminated or truncated:
            obs, info = test_env.reset()

    test_env.close()

# Uncomment to test the model after training
# test_model()
