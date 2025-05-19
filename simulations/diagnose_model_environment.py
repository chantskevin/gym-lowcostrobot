import os
import gymnasium as gym
import numpy as np
import torch
import gym_lowcostrobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import argparse
import pickle
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose model and environment issues')
    parser.add_argument('--model-path', type=str, default="logs/so101_reach_fixed/models/so101_reach_fixed_final",
                        help='Path to the model (without .zip extension)')
    parser.add_argument('--distance-threshold', type=float, default=0.05,
                        help='Distance threshold for environment')
    parser.add_argument('--run-episodes', type=int, default=5,
                        help='Number of episodes to run for evaluation')
    parser.add_argument('--save-diagnostics', action='store_true',
                        help='Save diagnostic information to a file')
    return parser.parse_args()

def create_env_with_params(distance_threshold, reward_type="dense", n_substeps=20, render_mode=None):
    """Create an environment with explicit parameters"""
    env = gym.make('SO101ReachFixedTarget-v0',
                  distance_threshold=distance_threshold,
                  reward_type=reward_type,
                  n_substeps=n_substeps,
                  render_mode=render_mode)
    return env

def inspect_vec_normalize(vec_normalize_env, prefix=""):
    """Inspect the VecNormalize wrapper and its statistics"""
    print(f"{prefix}VecNormalize Configuration:")
    print(f"{prefix}- norm_obs: {vec_normalize_env.norm_obs}")
    print(f"{prefix}- norm_reward: {vec_normalize_env.norm_reward}")
    print(f"{prefix}- clip_obs: {vec_normalize_env.clip_obs}")
    print(f"{prefix}- clip_reward: {vec_normalize_env.clip_reward}")
    print(f"{prefix}- gamma: {vec_normalize_env.gamma}")
    print(f"{prefix}- epsilon: {vec_normalize_env.epsilon}")

    if hasattr(vec_normalize_env, 'obs_rms') and vec_normalize_env.obs_rms is not None:
        print(f"{prefix}Observation Normalization Statistics:")
        if isinstance(vec_normalize_env.obs_rms, dict):
            print(f"{prefix}- Observation RMS is a dictionary with keys: {list(vec_normalize_env.obs_rms.keys())}")

            # Print detailed stats for each key
            for key, rms in vec_normalize_env.obs_rms.items():
                print(f"{prefix}  Stats for '{key}':")
                if hasattr(rms, 'mean'):
                    print(f"{prefix}  - Mean shape: {rms.mean.shape}")
                    print(f"{prefix}  - Mean: {rms.mean}")
                    print(f"{prefix}  - Variance: {rms.var}")
                    print(f"{prefix}  - Count: {rms.count}")
                else:
                    print(f"{prefix}  - Unable to extract detailed stats for {key}")
        else:
            print(f"{prefix}- Observation RMS type: {type(vec_normalize_env.obs_rms)}")

    if hasattr(vec_normalize_env, 'ret_rms') and vec_normalize_env.ret_rms is not None:
        print(f"{prefix}Reward Normalization Statistics:")
        if hasattr(vec_normalize_env.ret_rms, 'mean'):
            print(f"{prefix}- Mean: {vec_normalize_env.ret_rms.mean}")
            print(f"{prefix}- Variance: {vec_normalize_env.ret_rms.var}")
            print(f"{prefix}- Count: {vec_normalize_env.ret_rms.count}")
        else:
            print(f"{prefix}- Reward RMS type: {type(vec_normalize_env.ret_rms)}")

def compare_environments(env1, env2):
    """Compare two environments to check for differences"""
    print("\n=== Environment Comparison ===")

    # Compare basic properties
    env1_props = vars(env1.envs[0].unwrapped)
    env2_props = vars(env2.envs[0].unwrapped)

    print("Key environment parameters:")
    print(f"- distance_threshold: {env1_props.get('distance_threshold')} vs {env2_props.get('distance_threshold')}")
    print(f"- reward_type: {env1_props.get('reward_type')} vs {env2_props.get('reward_type')}")
    print(f"- n_substeps: {env1_props.get('control_decimation')} vs {env2_props.get('control_decimation')}")

    # Compare observation and action spaces
    print("\nObservation spaces:")
    print(f"- Env1: {env1.observation_space}")
    print(f"- Env2: {env2.observation_space}")

    print("\nAction spaces:")
    print(f"- Env1: {env1.action_space}")
    print(f"- Env2: {env2.action_space}")

    # If both are VecNormalize, compare normalization stats
    if isinstance(env1, VecNormalize) and isinstance(env2, VecNormalize):
        print("\nVecNormalize comparison:")
        print(f"- norm_obs: {env1.norm_obs} vs {env2.norm_obs}")
        print(f"- norm_reward: {env1.norm_reward} vs {env2.norm_reward}")
        print(f"- clip_obs: {env1.clip_obs} vs {env2.clip_obs}")
        print(f"- clip_reward: {env1.clip_reward} vs {env2.clip_reward}")

        # Compare observation RMS if available
        if hasattr(env1, 'obs_rms') and hasattr(env2, 'obs_rms'):
            if isinstance(env1.obs_rms, dict) and isinstance(env2.obs_rms, dict):
                print("\nObservation RMS keys:")
                print(f"- Env1: {list(env1.obs_rms.keys())}")
                print(f"- Env2: {list(env2.obs_rms.keys())}")

                # Compare stats for common keys
                common_keys = set(env1.obs_rms.keys()).intersection(set(env2.obs_rms.keys()))
                for key in common_keys:
                    if hasattr(env1.obs_rms[key], 'mean') and hasattr(env2.obs_rms[key], 'mean'):
                        mean_diff = np.abs(env1.obs_rms[key].mean - env2.obs_rms[key].mean).max()
                        var_diff = np.abs(env1.obs_rms[key].var - env2.obs_rms[key].var).max()
                        print(f"\nKey '{key}' differences:")
                        print(f"- Max mean difference: {mean_diff}")
                        print(f"- Max variance difference: {var_diff}")

                        if mean_diff > 0.1 or var_diff > 0.1:
                            print(f" SIGNIFICANT DIFFERENCE DETECTED in '{key}' normalization!")
                            print(f"Env1 mean: {env1.obs_rms[key].mean}")
                            print(f"Env2 mean: {env2.obs_rms[key].mean}")
                            print(f"Env1 var: {env1.obs_rms[key].var}")
                            print(f"Env2 var: {env2.obs_rms[key].var}")

def evaluate_model(model, env, num_episodes=5):
    """Evaluate a model in an environment for a number of episodes"""
    print("\n=== Model Evaluation ===")

    # Handle different gym versions with different reset() return types
    try:
        reset_result = env.reset()
        # Check if reset returns a tuple (newer gym versions)
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # First element is observation
        else:
            # Older gym versions just return the observation
            obs = reset_result
    except Exception as e:
        print(f"Error during reset: {e}")
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_episode_length": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
            "error": str(e)
        }

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    current_episode_reward = 0
    current_episode_length = 0

    for _ in range(num_episodes * 150):  # 150 steps per episode max
        try:
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)

            # Handle different gym versions with different step() return types
            if len(step_result) == 4:  # Old gym: obs, reward, done, info
                next_obs, reward, done, info = step_result
                truncated = False
            elif len(step_result) == 5:  # New gym: obs, reward, terminated, truncated, info
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step() return length: {len(step_result)}")

            # Extract the first element if these are arrays/lists
            if isinstance(reward, (list, tuple, np.ndarray)):
                reward = reward[0]
            if isinstance(done, (list, tuple, np.ndarray)):
                done = done[0]
            if isinstance(info, (list, tuple)):
                info = info[0]

            current_episode_reward += reward
            current_episode_length += 1

            if done:
                # Check for success
                if isinstance(info, dict) and 'is_success' in info:
                    if info['is_success']:
                        success_count += 1
                        print(f"Episode finished successfully with reward {current_episode_reward}")
                    else:
                        print(f"Episode finished unsuccessfully with reward {current_episode_reward}")
                else:
                    print(f"Episode finished with reward {current_episode_reward}")

                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)

                if len(episode_rewards) >= num_episodes:
                    break

                # Reset for next episode
                try:
                    reset_result = env.reset()
                    # Check if reset returns a tuple (newer gym versions)
                    if isinstance(reset_result, tuple):
                        obs = reset_result[0]  # First element is observation
                    else:
                        # Older gym versions just return the observation
                        obs = reset_result
                except Exception as e:
                    print(f"Error during reset: {e}")
                    break

                current_episode_reward = 0
                current_episode_length = 0
            else:
                obs = next_obs

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            break

    if len(episode_rewards) == 0:
        print("No episodes completed successfully")
        return {
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_episode_length": 0.0,
            "episode_rewards": [],
            "episode_lengths": []
        }

    print(f"\nEvaluation results after {len(episode_rewards)} episodes:")
    print(f"- Success rate: {success_count}/{len(episode_rewards)} ({success_count/len(episode_rewards)*100:.1f}%)")
    print(f"- Average reward: {np.mean(episode_rewards):.4f}")
    print(f"- Average episode length: {np.mean(episode_lengths):.1f}")

    return {
        "success_rate": success_count/len(episode_rewards),
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "episode_rewards": [float(r) for r in episode_rewards],
        "episode_lengths": [int(l) for l in episode_lengths]
    }

def save_diagnostics(model_path, diagnostics, filename=None):
    """Save diagnostic information to a file"""
    if filename is None:
        model_name = os.path.basename(model_path)
        filename = f"diagnostics_{model_name}.json"

    with open(filename, 'w') as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Diagnostics saved to {filename}")

def main():
    args = parse_args()
    model_path = args.model_path
    distance_threshold = args.distance_threshold

    print("=" * 50)
    print(f"DIAGNOSING MODEL: {model_path}")
    print(f"DISTANCE THRESHOLD: {distance_threshold}")
    print("=" * 50)

    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f" Model not found at {model_path}.zip")
        return

    # Check for VecNormalize stats
    vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
    has_vec_normalize = os.path.exists(vec_normalize_path)
    print(f"VecNormalize stats: {' Found' if has_vec_normalize else ' Not found'} at {vec_normalize_path}")

    # Create two environments: one for training (with training=True) and one for testing (with training=False)
    print("\n=== Creating Training Environment ===")
    train_env = DummyVecEnv([lambda: create_env_with_params(distance_threshold)])
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0
    )

    print("\n=== Creating Testing Environment ===")
    test_env = DummyVecEnv([lambda: create_env_with_params(distance_threshold)])
    test_env = VecNormalize(
        test_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0
    )
    test_env.training = False  # This is the key difference
    test_env.norm_reward = False

    # Load VecNormalize stats if available
    if has_vec_normalize:
        print("\n=== Loading VecNormalize Stats ===")
        try:
            # First, examine the VecNormalize pickle file
            with open(vec_normalize_path, 'rb') as f:
                vec_data = pickle.load(f)
                print(f"VecNormalize pickle keys: {list(vec_data.keys()) if isinstance(vec_data, dict) else 'Not a dict'}")

            # Load stats into both environments
            train_env = VecNormalize.load(vec_normalize_path, train_env)
            train_env.training = False
            train_env.norm_reward = False

            test_env = VecNormalize.load(vec_normalize_path, test_env)
            test_env.training = False
            test_env.norm_reward = False

            print(" VecNormalize stats loaded successfully")
        except Exception as e:
            print(f" Error loading VecNormalize stats: {e}")

    # Inspect the environments
    print("\n=== Training Environment ===")
    inspect_vec_normalize(train_env, prefix="  ")

    print("\n=== Testing Environment ===")
    inspect_vec_normalize(test_env, prefix="  ")

    # Compare the environments
    compare_environments(train_env, test_env)

    # Load the model
    print("\n=== Loading Model ===")
    try:
        # First with the training environment
        model_train = PPO.load(model_path, env=train_env)
        print(" Model loaded successfully with training environment")

        # Then with the testing environment
        model_test = PPO.load(model_path, env=test_env)
        print(" Model loaded successfully with testing environment")

        # Print model hyperparameters
        print("\n=== Model Hyperparameters ===")
        print(f"- Learning rate: {model_test.learning_rate}")
        print(f"- Entropy coefficient: {model_test.ent_coef}")
        print(f"- Value function coefficient: {model_test.vf_coef}")
        print(f"- Max gradient norm: {model_test.max_grad_norm}")
        print(f"- Gamma (discount factor): {model_test.gamma}")
        print(f"- GAE Lambda: {model_test.gae_lambda}")
        print(f"- Number of steps: {model_test.n_steps}")
        print(f"- Batch size: {model_test.batch_size}")
        print(f"- Number of epochs: {model_test.n_epochs}")
        print(f"- Target KL divergence: {model_test.target_kl}")

        # Try to get clip range (might be a function)
        try:
            if callable(model_test.clip_range):
                print(f"- Clip range: Function (likely constant or schedule)")
            else:
                print(f"- Clip range: {model_test.clip_range}")
        except:
            print(f"- Clip range: Unable to determine")

        # Evaluate the model in both environments
        print("\n=== Evaluating Model in Training Environment ===")
        train_results = evaluate_model(model_train, train_env, num_episodes=args.run_episodes)

        print("\n=== Evaluating Model in Testing Environment ===")
        test_results = evaluate_model(model_test, test_env, num_episodes=args.run_episodes)

        # Save diagnostics if requested
        if args.save_diagnostics:
            diagnostics = {
                "model_path": model_path,
                "distance_threshold": distance_threshold,
                "has_vec_normalize": has_vec_normalize,
                "training_environment_results": train_results,
                "testing_environment_results": test_results,
                "hyperparameters": {
                    "learning_rate": float(model_test.learning_rate),
                    "entropy_coef": float(model_test.ent_coef),
                    "value_function_coef": float(model_test.vf_coef),
                    "max_gradient_norm": float(model_test.max_grad_norm),
                    "gamma": float(model_test.gamma),
                    "gae_lambda": float(model_test.gae_lambda),
                    "n_steps": int(model_test.n_steps),
                    "batch_size": int(model_test.batch_size),
                    "n_epochs": int(model_test.n_epochs),
                    "target_kl": float(model_test.target_kl) if model_test.target_kl is not None else None
                }
            }
            save_diagnostics(model_path, diagnostics)

    except Exception as e:
        print(f" Error loading or evaluating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
