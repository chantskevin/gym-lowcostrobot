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
import argparse
import datetime
import shutil

# Set device to MPS if available (for Apple Silicon Macs)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Configure PyTorch for maximum performance
if torch.backends.mps.is_available():
    # Set PyTorch to use the largest possible memory allocation
    torch.mps.set_per_process_memory_fraction(0.95)  # Use 95% of available GPU memory
    print("Configured MPS for maximum memory usage")

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Train or continue training a PPO agent for the SO101 reach fixed target task')
    parser.add_argument('--continue_training', action='store_true',
                        help='Continue training from a saved model')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Number of timesteps to train for')
    parser.add_argument('--distance_threshold', type=float, default=0.1,
                        help='Distance threshold for success (default: 0.1)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Create log directory
    log_dir = "logs/so101_reach_fixed/"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Create the environment with seed for reproducibility
    def make_env(rank=0):
        def _init():
            env = gym.make('SO101ReachFixedTarget-v0',
                         distance_threshold=args.distance_threshold,
                         n_substeps=20,
                         reward_type="dense",  # Change to dense reward for better learning
                         render_mode=None)
            env.reset(seed=42 + rank)
            return env
        return _init

    # Use DummyVecEnv for simplicity
    n_envs = 16  # Reduced number of environments for more stable training

    # Create a vectorized environment
    env = make_vec_env(
        make_env(),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    # Set up callbacks for saving models and evaluating performance
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save more frequently since training will be faster
        save_path=models_dir,
        name_prefix="so101_reach_fixed_model"
    )

    # Create and train the agent with simpler hyperparameters for this easy task
    if args.continue_training:
        model_path = os.path.join(models_dir, "so101_reach_fixed_final.zip")
        print(f"Loading model from {model_path} and continuing training...")

        # First, create a checkpoint of the current model before continuing training
        checkpoint_dir = "model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"before_continue_{timestamp}_thresh{args.distance_threshold}"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        os.makedirs(checkpoint_path, exist_ok=True)

        # Copy the model file
        shutil.copy2(model_path, os.path.join(checkpoint_path, "model.zip"))

        # Copy the VecNormalize stats
        vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            shutil.copy2(vec_normalize_path, os.path.join(checkpoint_path, "vec_normalize.pkl"))
        print(f"✅ Created checkpoint at {checkpoint_path}")

        # Define the environment creation function with explicit parameters
        # This is critical - must match the test environment exactly
        def make_env_with_params():
            env = gym.make('SO101ReachFixedTarget-v0',
                          distance_threshold=args.distance_threshold,
                          reward_type="dense",
                          n_substeps=20)
            return env

        # Create the vectorized environment with the explicit parameters
        env = make_vec_env(
            make_env_with_params,
            n_envs=n_envs,
            vec_env_cls=DummyVecEnv
        )

        # Load the saved normalization stats if provided
        if os.path.exists(vec_normalize_path):
            print(f"Loading normalization stats from {vec_normalize_path}")
            try:
                # First create the VecNormalize wrapper with the same parameters as in testing
                env = VecNormalize(
                    env,
                    norm_obs=False,
                    norm_reward=False,
                    clip_obs=10.0,
                    clip_reward=10.0
                )

                # Then load the stats
                env = VecNormalize.load(vec_normalize_path, env)

                env.training = False
                env.norm_reward = False
                print("✅ Normalization stats loaded successfully")
            except Exception as e:
                print(f"⚠️ Error loading normalization stats: {e}")
                print("Creating new normalization stats instead")

                # If loading fails, create a fresh VecNormalize wrapper
                env = VecNormalize(
                    env,
                    norm_obs=True,
                    norm_reward=True,
                    clip_obs=10.0,
                    clip_reward=10.0
                )
        else:
            # If no stats file exists, create a fresh VecNormalize wrapper
            env = VecNormalize(
                env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                clip_reward=10.0
            )

        # Load the model with the fresh environment
        print("Loading model with fresh environment...")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
        print("✅ Model loaded successfully with fresh environment")

        # Disable training temporarily to check performance before modifying
        env.training = False
        env.norm_reward = False

        # Check initial performance
        print("\n=== Checking model performance before continuing training ===")

        # Using Gymnasium 1.1.1 API where reset() returns (obs, info)
        obs = env.reset()
        episodes = 0
        success_count = 0

        for _ in range(5 * 150):  # 5 episodes, 150 steps max per episode
            action, _ = model.predict(obs, deterministic=True)

            # VecNormalize wrapper returns 4 values, not 5
            next_obs, reward, done, info = env.step(action)

            # Extract the first element if these are arrays/lists (for vectorized environments)
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = reward[0]
            if isinstance(done, (list, tuple, np.ndarray)):
                done = done[0]
            if isinstance(info, (list, tuple)):
                info = info[0]

            if done:
                episodes += 1
                if isinstance(info, dict) and 'is_success' in info:
                    if info['is_success']:
                        success_count += 1
                        print(f"Episode {episodes} finished successfully")
                    else:
                        print(f"Episode {episodes} finished unsuccessfully")

                if episodes >= 5:
                    break

                # Reset for next episode - Gymnasium 1.1.1 API
                obs = env.reset()
            else:
                obs = next_obs

        print(f"Initial performance: {success_count}/{episodes} successful episodes ({success_count/episodes*100:.1f}%)")

        # Re-enable training
        env.training = True
        env.norm_reward = True

        # Use a VERY conservative learning rate for fine-tuning
        from stable_baselines3.common.utils import constant_fn
        model.learning_rate = 5e-6  # 20x smaller than default
        model.clip_range = constant_fn(0.05)  # Very small clip range

        print(f"Fine-tuning with:")
        print(f"- Learning rate: {model.learning_rate}")
        print(f"- Clip range: 0.05 (constant)")
        print(f"- Distance threshold: {args.distance_threshold}")
    else:
        print("Creating new model for training...")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=1e-4,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            clip_range=0.2,
            n_steps=512,
            batch_size=128,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            device=device,
            policy_kwargs={
                "net_arch": {
                    "pi": [256, 256],
                    "vf": [256, 256]
                },
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True,
                "log_std_init": 0.0,  # Start with higher action noise
                "optimizer_kwargs": {
                    "weight_decay": 0.0,  # No weight decay for more aggressive updates
                },
            },
        )

    # Train the agent
    total_timesteps = args.timesteps
    tb_log_name = "PPO_continued" if args.continue_training else "PPO"

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        tb_log_name=tb_log_name,
        reset_num_timesteps=not args.continue_training  # Don't reset if continuing training
    )

    # Save the final model
    model.save(os.path.join(models_dir, "so101_reach_fixed_final"))
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))

    print("Training complete!")

    # Automatically test the model after training
    test_model()

# Create a function to test the trained model
def test_model():
    print("Testing the trained model...")

    # Use the same args as in main
    args = parse_args()

    # Load the trained model
    models_dir = "logs/so101_reach_fixed/models"
    model_path = os.path.join(models_dir, "so101_reach_fixed_final")
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create a test environment with the same parameters as training
    test_env = gym.make('SO101ReachFixedTarget-v0',
                       distance_threshold=args.distance_threshold,
                       reward_type="dense",
                       n_substeps=20,
                       render_mode="human")

    # Wrap the environment in a DummyVecEnv to make it compatible with VecNormalize
    test_env = DummyVecEnv([lambda: test_env])

    # Load the normalization stats
    vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        try:
            test_env = VecNormalize(
                test_env,
                norm_obs=False,
                norm_reward=False,
                clip_obs=10.0,
                clip_reward=10.0
            )
            test_env = VecNormalize.load(vec_normalize_path, test_env)
            test_env.training = False
            test_env.norm_reward = False
            print(f"Loaded normalization stats from {vec_normalize_path}")
        except Exception as e:
            print(f"Error loading normalization stats: {e}")
            # Create a new VecNormalize wrapper if loading fails
            test_env = VecNormalize(
                test_env,
                norm_obs=False,
                norm_reward=False,
                clip_obs=10.0,
                clip_reward=10.0
            )
            test_env.training = False
            test_env.norm_reward = False
    else:
        print("No normalization stats found, creating new VecNormalize wrapper")
        test_env = VecNormalize(
            test_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
        test_env.training = False
        test_env.norm_reward = False

    # Test the model
    obs = test_env.reset()
    # Handle different gym versions with different reset() return types
    if isinstance(obs, tuple):
        obs = obs[0]  # First element is observation

    max_steps = 150  # Increase from default to give more time to reach target
    max_episodes = 25
    episodes = 0
    success_count = 0
    timeout_count = 0
    total_reward = 0

    for i in range(max_steps * max_episodes):  # Allow more steps per episode
        action, _ = model.predict(obs, deterministic=False)

        # Handle different gym versions with different step() return types
        step_result = test_env.step(action)
        if len(step_result) == 4:  # Old gym: obs, reward, done, info
            next_obs, rewards, dones, infos = step_result
        elif len(step_result) == 5:  # New gym: obs, reward, terminated, truncated, info
            next_obs, rewards, terminated, truncated, infos = step_result
            dones = [t or tr for t, tr in zip(terminated, truncated)]
        else:
            raise ValueError(f"Unexpected step() return length: {len(step_result)}")

        # Extract the first element if these are arrays/lists
        reward = float(rewards[0]) if isinstance(rewards, (list, tuple, np.ndarray)) else float(rewards)
        done = bool(dones[0]) if isinstance(dones, (list, tuple, np.ndarray)) else bool(dones)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos

        total_reward += reward
        terminated = done
        truncated = False

        # Print distance to target every 10 steps
        if i % 10 == 0:
            if isinstance(info, dict) and 'distance' in info:
                if isinstance(info['distance'], (np.ndarray, list)):
                    distance_value = float(info['distance'].item()) if hasattr(info['distance'], 'item') else info['distance'][0]
                    print(f"Distance to target: {distance_value:.4f}, Reward: {reward:.4f}")
                else:
                    print(f"Distance to target: {info['distance']:.4f}, Reward: {reward:.4f}")
            else:
                print(f"Reward: {reward:.4f}, Info: {info}")

        # Check if the episode is done
        if terminated or truncated:
            episodes += 1

            # Check for success
            if isinstance(info, dict) and 'is_success' in info:
                success = info['is_success']
                if success:
                    success_count += 1
                    print(f"Success! Episode {episodes} finished with total reward: {float(total_reward):.4f}")
                else:
                    timeout_count += 1
                    print(f"Timeout! Episode {episodes} finished with total reward: {float(total_reward):.4f}")
            else:
                print(f"Episode {episodes} finished with total reward: {float(total_reward):.4f}, info: {info}")

            if episodes >= max_episodes:
                break

            # Reset for next episode
            reset_result = test_env.reset()
            if isinstance(reset_result, tuple):
                obs = reset_result[0]  # First element is observation
            else:
                obs = reset_result

            total_reward = 0
        else:
            obs = next_obs

    print(f"Testing complete. Success rate: {success_count}/{episodes} ({success_count/episodes*100:.1f}%), Timeouts: {timeout_count}")
    test_env.close()

if __name__ == "__main__":
    # main()
    test_model()
