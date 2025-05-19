import os
import argparse
import shutil
from datetime import datetime
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Create a checkpoint of a trained model')
    parser.add_argument('--source-model', type=str, default="logs/so101_reach_fixed/models/so101_reach_fixed_final",
                        help='Path to the source model (without .zip extension)')
    parser.add_argument('--checkpoint-dir', type=str, default="model_checkpoints",
                        help='Directory to save the checkpoint')
    parser.add_argument('--checkpoint-name', type=str, default=None,
                        help='Name for the checkpoint (default: auto-generated with timestamp and threshold)')
    parser.add_argument('--distance-threshold', type=float, default=None,
                        help='Distance threshold used for training (for naming purposes)')
    parser.add_argument('--include-stats', action='store_true',
                        help='Include VecNormalize statistics in the checkpoint')
    parser.add_argument('--include-env-config', action='store_true',
                        help='Include environment configuration in the checkpoint')
    return parser.parse_args()

def create_checkpoint(args):
    # Check if source model exists
    source_model_path = args.source_model
    if not os.path.exists(source_model_path + ".zip"):
        print(f"❌ Source model not found at {source_model_path}.zip")
        return False

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Generate checkpoint name if not provided
    if args.checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        threshold_str = f"_thresh{args.distance_threshold}" if args.distance_threshold is not None else ""
        checkpoint_name = f"model_checkpoint_{timestamp}{threshold_str}"
    else:
        checkpoint_name = args.checkpoint_name

    # Create a subdirectory for this checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Copy the model file
    source_model_file = source_model_path + ".zip"
    target_model_file = os.path.join(checkpoint_path, "model.zip")
    shutil.copy2(source_model_file, target_model_file)
    print(f"✅ Model copied to {target_model_file}")

    # Copy VecNormalize statistics if requested
    if args.include_stats:
        source_stats_file = os.path.join(os.path.dirname(source_model_path), "vec_normalize.pkl")
        if os.path.exists(source_stats_file):
            target_stats_file = os.path.join(checkpoint_path, "vec_normalize.pkl")
            shutil.copy2(source_stats_file, target_stats_file)
            print(f"✅ VecNormalize statistics copied to {target_stats_file}")
        else:
            print(f"⚠️ VecNormalize statistics not found at {source_stats_file}")

    # Save environment configuration if requested
    if args.include_env_config:
        try:
            # Load the model to get its parameters
            model = PPO.load(source_model_path)

            # Create a config file
            config_file = os.path.join(checkpoint_path, "config.txt")
            with open(config_file, 'w') as f:
                f.write("=== Model Checkpoint Configuration ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source Model: {source_model_path}\n")
                f.write(f"Distance Threshold: {args.distance_threshold}\n\n")

                f.write("=== Model Hyperparameters ===\n")
                f.write(f"Learning Rate: {model.learning_rate}\n")
                f.write(f"Entropy Coefficient: {model.ent_coef}\n")
                f.write(f"Value Function Coefficient: {model.vf_coef}\n")
                f.write(f"Max Gradient Norm: {model.max_grad_norm}\n")
                f.write(f"Gamma (Discount Factor): {model.gamma}\n")
                f.write(f"GAE Lambda: {model.gae_lambda}\n")
                f.write(f"Number of Steps: {model.n_steps}\n")
                f.write(f"Batch Size: {model.batch_size}\n")
                f.write(f"Number of Epochs: {model.n_epochs}\n")
                f.write(f"Target KL Divergence: {model.target_kl}\n")

                # Try to get clip range (might be a function)
                try:
                    if callable(model.clip_range):
                        f.write(f"Clip Range: Function (likely constant or schedule)\n")
                    else:
                        f.write(f"Clip Range: {model.clip_range}\n")
                except:
                    f.write(f"Clip Range: Unable to determine\n")

            print(f"✅ Environment configuration saved to {config_file}")
        except Exception as e:
            print(f"⚠️ Failed to save environment configuration: {e}")

    print(f"\n✅ Checkpoint created successfully at {checkpoint_path}")
    return True

def main():
    args = parse_args()
    create_checkpoint(args)

if __name__ == "__main__":
    main()
