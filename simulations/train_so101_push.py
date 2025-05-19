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
from stable_baselines3.common.utils import set_random_seed

# Set device to MPS if available (for Apple Silicon Macs)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Configure PyTorch for maximum performance
if torch.backends.mps.is_available():
    # Set PyTorch to use the largest possible memory allocation
    torch.mps.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
    print("Configured MPS for maximum memory usage")

def main():
    # Create log directory
    log_dir = "logs/so101_push_cube/"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Create the environment with seed for reproducibility
    def make_env(rank=0):
        def _init():
            env = gym.make('SO101PushCube-v0',
                          observation_mode="state",  # Use state observations for faster training
                          action_mode="joint",       # Control the joints directly
                          reward_type="dense",       # Dense rewards help with learning
                          render_mode=None)          # No rendering during training for speed
            env.reset(seed=42 + rank)
            return env
        return _init

    # Use DummyVecEnv instead of SubprocVecEnv to avoid multiprocessing issues
    # Number of parallel environments
    n_envs = 40  # For Mac, often 1 is best with MPS

    # Create a vectorized environment
    env = make_vec_env(
        make_env(),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Set up callbacks for saving models and evaluating performance
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=models_dir,
        name_prefix="so101_push_cube_model"
    )

    # Create and train the agent with optimized hyperparameters for GPU utilization
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=512,            # Reduced to allow more frequent updates
        batch_size=128,          # Increased for better GPU utilization
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,
        clip_range_vf=None,      # no clipping on value-fn
        ent_coef=0.001,           # Slightly higher entropy coefficient for exploration
        device=device,
        policy_kwargs={
            "net_arch": [128, 128],  # Larger network to utilize GPU better
        }
    )

    # Train the agent
    total_timesteps = 5000000
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="PPO"
    )

    # Save the final model
    model.save(os.path.join(models_dir, "so101_push_cube_final"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))

    print("Training complete!")

# Create a function to test the trained model
def test_model():
    # Load the trained model
    models_dir = "logs/so101_push_cube/models"
    model = PPO.load(os.path.join(models_dir, "so101_push_cube_final"))

    # Create a test environment with rendering
    test_env = gym.make('SO101PushCube-v0',
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
            print(f"Episode finished with reward: {reward}")
            obs, info = test_env.reset()

    test_env.close()

if __name__ == "__main__":
    main()
    # Uncomment to test the model after training
    # test_model()
