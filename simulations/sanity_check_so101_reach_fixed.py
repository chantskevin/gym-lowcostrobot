import os
import time
import argparse
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Sanity check for SO101ReachFixedTarget-v0 environment')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run')
parser.add_argument('--steps', type=int, default=100, help='Maximum steps per episode')
parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps (seconds)')
parser.add_argument('--distance-threshold', type=float, default=0.05, help='Distance threshold for success')
parser.add_argument('--reward-type', type=str, default='dense', choices=['sparse', 'dense'], help='Reward type')
parser.add_argument('--action-mode', type=str, default='joint', choices=['joint', 'ee'], help='Action mode')
parser.add_argument('--test-expert', action='store_true', help='Test with a trained expert model')
parser.add_argument('--save-plots', action='store_true', help='Save plots of the results')
args = parser.parse_args()

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)

def print_subheader(title):
    """Print a formatted subheader"""
    print("\n" + "-"*80)
    print(f" {title} ".center(80, "-"))
    print("-"*80)

def test_env_creation():
    """Test environment creation"""
    print_subheader("Environment Creation")

    try:
        env = gym.make('SO101ReachFixedTarget-v0',
                      distance_threshold=args.distance_threshold,
                      reward_type=args.reward_type,
                      action_mode=args.action_mode,
                      n_substeps=20,
                      render_mode="human")
        print("✅ Environment created successfully")

        # Print environment information
        print(f"\nAction Space: {env.action_space}")
        print(f"Observation Space: {env.observation_space}")

        # Close the environment
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return False

def test_reset():
    """Test environment reset"""
    print_subheader("Environment Reset")

    try:
        env = gym.make('SO101ReachFixedTarget-v0',
                      distance_threshold=args.distance_threshold,
                      reward_type=args.reward_type,
                      action_mode=args.action_mode,
                      n_substeps=20,
                      render_mode="human")

        # Reset the environment
        obs, info = env.reset()

        print("✅ Environment reset successful")
        print(f"\nObservation keys: {obs.keys()}")
        print(f"Arm qpos shape: {obs['arm_qpos'].shape}, values: {obs['arm_qpos']}")
        print(f"Arm qvel shape: {obs['arm_qvel'].shape}, values: {obs['arm_qvel']}")
        print(f"Target position: {obs['target_pos']}")
        print(f"Info: {info}")

        # Close the environment
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed to reset environment: {e}")
        return False

def test_step():
    """Test environment step"""
    print_subheader("Environment Step")

    try:
        env = gym.make('SO101ReachFixedTarget-v0',
                      distance_threshold=args.distance_threshold,
                      reward_type=args.reward_type,
                      action_mode=args.action_mode,
                      n_substeps=20,
                      render_mode="human")

        # Reset the environment
        obs, _ = env.reset()

        # Take a random action
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        print("✅ Environment step successful")
        print(f"\nAction taken: {action}")
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print(f"Info: {info}")
        print(f"Distance to target: {info.get('distance', 'Not available')}")
        print(f"Success: {info.get('is_success', 'Not available')}")

        # Close the environment
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed to step environment: {e}")
        return False

def test_random_episodes():
    """Test random episodes"""
    print_subheader("Random Episodes")

    try:
        env = gym.make('SO101ReachFixedTarget-v0',
                      distance_threshold=args.distance_threshold,
                      reward_type=args.reward_type,
                      action_mode=args.action_mode,
                      n_substeps=20,
                      render_mode="human")

        # Statistics to track
        episode_rewards = []
        episode_lengths = []
        episode_distances = []
        episode_successes = []

        print(f"Running {args.episodes} random episodes with {args.steps} steps each...")

        for episode in range(args.episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_success = False
            final_distance = None

            for step in range(args.steps):
                # Take a random action
                action = env.action_space.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)

                # Update statistics
                episode_reward += reward
                final_distance = info.get('distance', None)
                episode_success = info.get('is_success', False)

                # Print progress every 10 steps
                if step % 10 == 0:
                    print(f"Episode {episode+1}, Step {step}: Reward={reward:.4f}, Distance={info.get('distance', 'N/A')}")

                # Add delay for visualization
                time.sleep(args.delay)

                # Check if episode is done
                if terminated or truncated:
                    break

                # Update observation
                obs = next_obs

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            episode_distances.append(final_distance)
            episode_successes.append(episode_success)

            print(f"Episode {episode+1} finished: Reward={episode_reward:.4f}, Steps={step+1}, Success={episode_success}, Final Distance={final_distance:.4f}")

        # Calculate and print summary statistics
        success_rate = sum(episode_successes) / len(episode_successes) * 100
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_steps = sum(episode_lengths) / len(episode_lengths)
        avg_distance = sum(episode_distances) / len(episode_distances)

        print("\nRandom Episodes Summary:")
        print(f"Success Rate: {success_rate:.2f}% ({sum(episode_successes)}/{len(episode_successes)})")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Average Final Distance: {avg_distance:.4f}")

        # Save plots if requested
        if args.save_plots:
            save_plots(episode_rewards, episode_lengths, episode_distances, episode_successes, "random")

        # Close the environment
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed during random episodes: {e}")
        return False

def test_expert_episodes():
    """Test episodes with a trained expert model"""
    print_subheader("Expert Model Episodes")

    try:
        # Path to the trained model
        models_dir = "logs/so101_reach_fixed/models"
        model_path = os.path.join(models_dir, "so101_reach_fixed_final")
        vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")

        # Check if model exists
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Expert model not found at {model_path}.zip")
            return False

        # Load the trained model
        model = PPO.load(model_path)
        print("✅ Expert model loaded successfully")

        # Create the environment
        env = gym.make('SO101ReachFixedTarget-v0',
                      distance_threshold=args.distance_threshold,
                      reward_type=args.reward_type,
                      action_mode=args.action_mode,
                      n_substeps=20,
                      render_mode="human")

        # Wrap the environment in a DummyVecEnv to make it compatible with VecNormalize
        env = DummyVecEnv([lambda: env])

        # Load normalization stats if they exist
        if os.path.exists(vec_normalize_path):
            env = VecNormalize.load(vec_normalize_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ VecNormalize loaded successfully")

        # Statistics to track
        episode_rewards = []
        episode_lengths = []
        episode_distances = []
        episode_successes = []

        print(f"Running {args.episodes} expert episodes with {args.steps} steps each...")

        for episode in range(args.episodes):
            obs = env.reset()
            episode_reward = 0
            episode_success = False
            final_distance = None

            for step in range(args.steps):
                # Get action from the expert model
                action, _ = model.predict(obs, deterministic=True)

                # Apply the action to the environment
                obs, rewards, dones, infos = env.step(action)

                # Unwrap the vectorized environment returns
                reward = float(rewards[0])
                done = bool(dones[0])
                info = infos[0]

                # Update statistics
                episode_reward += reward
                final_distance = info.get('distance', None)
                episode_success = info.get('is_success', False)

                # Print progress every 10 steps
                if step % 10 == 0:
                    print(f"Episode {episode+1}, Step {step}: Reward={reward:.4f}, Distance={info.get('distance', 'N/A')}")

                # Add delay for visualization
                time.sleep(args.delay)

                # Check if episode is done
                if done:
                    break

            # Record episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(step + 1)
            episode_distances.append(final_distance)
            episode_successes.append(episode_success)

            print(f"Episode {episode+1} finished: Reward={episode_reward:.4f}, Steps={step+1}, Success={episode_success}, Final Distance={final_distance:.4f}")

        # Calculate and print summary statistics
        success_rate = sum(episode_successes) / len(episode_successes) * 100
        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_steps = sum(episode_lengths) / len(episode_lengths)
        avg_distance = sum(episode_distances) / len(episode_distances)

        print("\nExpert Episodes Summary:")
        print(f"Success Rate: {success_rate:.2f}% ({sum(episode_successes)}/{len(episode_successes)})")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Average Steps: {avg_steps:.2f}")
        print(f"Average Final Distance: {avg_distance:.4f}")

        # Save plots if requested
        if args.save_plots:
            save_plots(episode_rewards, episode_lengths, episode_distances, episode_successes, "expert")

        # Close the environment
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed during expert episodes: {e}")
        return False

def print_model_info(model_path):
    """Print detailed information about a trained model"""
    print_subheader("Model Information")

    try:
        # Check if model exists
        if not os.path.exists(model_path + ".zip"):
            print(f"❌ Model not found at {model_path}.zip")
            return False

        # Load the model without an environment (just to inspect parameters)
        model = PPO.load(model_path)

        # Get model attributes
        print("✅ Model loaded successfully for inspection")
        print("\nModel Hyperparameters:")
        print(f"- Learning rate: {model.learning_rate}")
        print(f"- Entropy coefficient: {model.ent_coef}")
        print(f"- Value function coefficient: {model.vf_coef}")
        print(f"- Max gradient norm: {model.max_grad_norm}")
        print(f"- Gamma (discount factor): {model.gamma}")
        print(f"- GAE Lambda: {model.gae_lambda}")
        print(f"- Number of steps: {model.n_steps}")
        print(f"- Batch size: {model.batch_size}")
        print(f"- Number of epochs: {model.n_epochs}")
        print(f"- Target KL divergence: {model.target_kl}")

        # Try to get clip range (might be a function)
        try:
            if callable(model.clip_range):
                print(f"- Clip range: Function (likely constant or schedule)")
            else:
                print(f"- Clip range: {model.clip_range}")
        except:
            print(f"- Clip range: Unable to determine")

        # Print policy network architecture
        if hasattr(model.policy, 'net_arch'):
            print("\nPolicy Network Architecture:")
            print(f"{model.policy.net_arch}")

        # Print observation and action space info
        print("\nAction Space Info:")
        if hasattr(model, 'action_space'):
            print(f"{model.action_space}")
        else:
            print("Action space information not available")

        print("\nObservation Space Info:")
        if hasattr(model, 'observation_space'):
            print(f"{model.observation_space}")
        else:
            print("Observation space information not available")

        # Try to get normalization statistics if VecNormalize was used
        vec_normalize_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")
        if os.path.exists(vec_normalize_path):
            print("\nVecNormalize Information:")
            print(f"- VecNormalize file exists at: {vec_normalize_path}")

            # Create a dummy environment to load the stats
            dummy_env = gym.make('SO101ReachFixedTarget-v0',
                              distance_threshold=args.distance_threshold,
                              reward_type=args.reward_type,
                              action_mode=args.action_mode)
            dummy_env = DummyVecEnv([lambda: dummy_env])

            try:
                dummy_env = VecNormalize.load(vec_normalize_path, dummy_env)
                print(f"- Observation normalization: {'Enabled' if dummy_env.norm_obs else 'Disabled'}")
                print(f"- Reward normalization: {'Enabled' if dummy_env.norm_reward else 'Disabled'}")
                print(f"- Observation clipping: {dummy_env.clip_obs}")
                print(f"- Reward clipping: {dummy_env.clip_reward}")

                # Try to print some normalization statistics
                if hasattr(dummy_env, 'obs_rms') and dummy_env.obs_rms is not None:
                    print("\nObservation Normalization Statistics:")
                    try:
                        if hasattr(dummy_env.obs_rms, 'mean'):
                            print(f"- Mean: {dummy_env.obs_rms.mean}")
                            print(f"- Variance: {dummy_env.obs_rms.var}")
                            print(f"- Count: {dummy_env.obs_rms.count}")
                        elif isinstance(dummy_env.obs_rms, dict):
                            print("- Observation RMS is a dictionary with keys:")
                            for key in dummy_env.obs_rms:
                                print(f"  - {key}")

                            # Try to print stats for each key in the dictionary
                            for key, rms in dummy_env.obs_rms.items():
                                print(f"\n  Stats for '{key}':")
                                if hasattr(rms, 'mean'):
                                    print(f"  - Mean: {rms.mean}")
                                    print(f"  - Variance: {rms.var}")
                                    print(f"  - Count: {rms.count}")
                                else:
                                    print(f"  - Unable to extract detailed stats for {key}")
                        else:
                            print(f"- Observation RMS type: {type(dummy_env.obs_rms)}")
                    except Exception as e:
                        print(f"- Error extracting observation stats: {e}")

                if hasattr(dummy_env, 'ret_rms') and dummy_env.ret_rms is not None:
                    print("\nReward Normalization Statistics:")
                    try:
                        if hasattr(dummy_env.ret_rms, 'mean'):
                            print(f"- Mean: {dummy_env.ret_rms.mean}")
                            print(f"- Variance: {dummy_env.ret_rms.var}")
                            print(f"- Count: {dummy_env.ret_rms.count}")
                        elif isinstance(dummy_env.ret_rms, dict):
                            print("- Reward RMS is a dictionary")
                            for key in dummy_env.ret_rms:
                                print(f"  - {key}")
                        else:
                            print(f"- Reward RMS type: {type(dummy_env.ret_rms)}")
                    except Exception as e:
                        print(f"- Error extracting reward stats: {e}")

                dummy_env.close()
            except Exception as e:
                print(f"- Unable to load VecNormalize stats: {e}")
        else:
            print("\nNo VecNormalize file found.")

        return True
    except Exception as e:
        print(f"❌ Failed to inspect model: {e}")
        return False

def save_plots(rewards, lengths, distances, successes, prefix):
    """Save plots of the results"""
    results_dir = "sanity_check_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(15, 10))

    # Plot success/failure
    plt.subplot(2, 2, 1)
    plt.bar(['Success', 'Failure'], [sum(successes), len(successes) - sum(successes)])
    plt.title(f'Success Rate: {sum(successes)/len(successes)*100:.2f}%')
    plt.ylabel('Count')

    # Plot rewards
    plt.subplot(2, 2, 2)
    plt.plot(rewards, 'b-')
    plt.axhline(y=sum(rewards)/len(rewards), color='r', linestyle='--', label=f'Avg: {sum(rewards)/len(rewards):.2f}')
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    # Plot episode lengths
    plt.subplot(2, 2, 3)
    plt.plot(lengths, 'g-')
    plt.axhline(y=sum(lengths)/len(lengths), color='r', linestyle='--', label=f'Avg: {sum(lengths)/len(lengths):.2f}')
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()

    # Plot final distances
    plt.subplot(2, 2, 4)
    plt.plot(distances, 'r-')
    plt.axhline(y=sum(distances)/len(distances), color='b', linestyle='--',
                label=f'Avg: {sum(distances)/len(distances):.4f}')
    plt.axhline(y=args.distance_threshold, color='g', linestyle=':', label=f'Threshold: {args.distance_threshold}')
    plt.title('Final Distances to Target')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    plt.savefig(os.path.join(results_dir, f"{prefix}_results_{timestamp}.png"))
    print(f"Plots saved to {os.path.join(results_dir, f'{prefix}_results_{timestamp}.png')}")

def main():
    print_header("SO101ReachFixedTarget-v0 Environment Sanity Check")

    print(f"Configuration:")
    print(f"- Episodes: {args.episodes}")
    print(f"- Steps per episode: {args.steps}")
    print(f"- Delay: {args.delay} seconds")
    print(f"- Distance threshold: {args.distance_threshold}")
    print(f"- Reward type: {args.reward_type}")
    print(f"- Action mode: {args.action_mode}")

    # Run the tests
    env_created = test_env_creation()
    if not env_created:
        print("❌ Environment creation failed. Aborting further tests.")
        return

    reset_ok = test_reset()
    if not reset_ok:
        print("❌ Environment reset failed. Aborting further tests.")
        return

    step_ok = test_step()
    if not step_ok:
        print("❌ Environment step failed. Aborting further tests.")
        return

    # Run random episodes
    random_ok = test_random_episodes()
    if not random_ok:
        print("❌ Random episodes test failed.")

    # Run expert episodes if requested
    if args.test_expert:
        expert_ok = test_expert_episodes()
        if not expert_ok:
            print("❌ Expert episodes test failed.")

        # Print model information
        model_path = os.path.join("logs/so101_reach_fixed/models", "so101_reach_fixed_final")
        print_model_info(model_path)

    print_header("Sanity Check Complete")

if __name__ == "__main__":
    main()
