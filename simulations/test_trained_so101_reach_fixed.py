import os
import time
import argparse
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid thread issues
import matplotlib.pyplot as plt
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description='Test the trained SO101 reach fixed target model')
parser.add_argument('--delay', type=float, default=0.01, help='Delay between steps (seconds)')
parser.add_argument('--episodes', type=int, default=50, help='Number of episodes to test')
parser.add_argument('--max-steps', type=int, default=150, help='Maximum steps per episode')
parser.add_argument('--distance-threshold', type=float, default=0.05, help='Distance threshold for success')
parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
parser.add_argument('--save-results', action='store_true', help='Save test results to file')
parser.add_argument('--verbose', action='store_true', help='Print detailed information during testing')
args = parser.parse_args()

def run_test():
    # Path to the trained model
    models_dir = "logs/so101_reach_fixed/models"
    model_path = os.path.join(models_dir, "so101_reach_fixed_final")
    vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")

    # Load the trained model
    model = PPO.load(model_path)
    print(f"Loaded model from {model_path}")

    # Create a test environment with rendering
    env = gym.make('SO101ReachFixedTarget-v0',
                  distance_threshold=args.distance_threshold,
                  reward_type="dense",  # Use the same reward type as training
                  n_substeps=20,
                  render_mode="human")

    # Wrap the environment in a DummyVecEnv to make it compatible with VecNormalize
    env = DummyVecEnv([lambda: env])

    # After loading your model:
    env = VecNormalize.load(vec_normalize_path, env)
    env.training = False    # turn off further updates
    env.norm_reward = False # only if you want raw rewards reported

    # Initialize metrics tracking
    episode_rewards = []
    episode_lengths = []
    episode_distances = []
    episode_successes = []
    episode_final_distances = []

    # Run the test for the specified number of episodes
    print(f"Running test for {args.episodes} episodes with {args.delay} second delay between steps")

    for episode in tqdm(range(args.episodes), desc="Testing Episodes"):
        # Reset the environment
        obs = env.reset()
        total_reward = 0
        step_count = 0
        distances = []
        success = False
        final_distance = None

        if args.verbose:
            print(f"\nStarting episode {episode+1}/{args.episodes}")

        # Run the episode
        for step in range(args.max_steps):
            # Get the action from the model
            action, _states = model.predict(obs, deterministic=True)

            # Apply the action to the environment
            obs, rewards, dones, infos = env.step(action)
            reward = float(rewards[0])
            done   = bool(dones[0])
            info   = infos[0]

            # In newer versions of Gym, 'done' is split into 'terminated' and 'truncated'
            # For compatibility with vectorized environments, we need to handle this
            terminated = done
            truncated = False

            # No need to unwrap observations - they're already in the right format
            # Just unwrap the other return values
            terminated = terminated[0] if isinstance(terminated, (list, np.ndarray)) else terminated

            # Accumulate reward
            total_reward += reward
            step_count += 1

            # Track distance to target
            if isinstance(info, dict) and 'distance' in info:
                distance = info['distance']
                if isinstance(distance, (np.ndarray, list)):
                    distance = float(distance.item()) if hasattr(distance, 'item') else distance[0]
                distances.append(distance)
                final_distance = distance

                if args.verbose and step % 10 == 0:
                    print(f"Step {step}: Distance to target: {distance:.4f}, Reward: {reward:.4f}")

            # Add delay for visualization
            time.sleep(args.delay)

            # Check if the episode is done
            if terminated or truncated:
                # Check for success
                if isinstance(info, dict) and 'is_success' in info:
                    success = bool(info['is_success'])

                if args.verbose:
                    print(f"Episode {episode+1} finished after {step_count} steps")
                    print(f"Total reward: {total_reward:.4f}")
                    print(f"Success: {success}")
                    print(f"Final distance: {final_distance:.4f}")

                break

        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        episode_successes.append(success)
        episode_final_distances.append(final_distance)
        episode_distances.append(distances)

    # Close the environment
    env.close()

    # Calculate and display summary statistics
    success_rate = sum(episode_successes) / len(episode_successes) * 100
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_lengths) / len(episode_lengths)
    avg_final_distance = sum(episode_final_distances) / len(episode_final_distances)

    print("\n===== Test Results =====")
    print(f"Success Rate: {success_rate:.2f}% ({sum(episode_successes)}/{len(episode_successes)})")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Final Distance: {avg_final_distance:.4f}")

    # Save results if requested
    if args.save_results:
        results_dir = "test_results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"test_results_{timestamp}.txt")

        with open(results_file, 'w') as f:
            f.write("===== Test Parameters =====\n")
            f.write(f"Episodes: {args.episodes}\n")
            f.write(f"Max Steps: {args.max_steps}\n")
            f.write(f"Distance Threshold: {args.distance_threshold}\n")
            f.write("\n===== Test Results =====\n")
            f.write(f"Success Rate: {success_rate:.2f}% ({sum(episode_successes)}/{len(episode_successes)})\n")
            f.write(f"Average Reward: {avg_reward:.4f}\n")
            f.write(f"Average Steps: {avg_steps:.2f}\n")
            f.write(f"Average Final Distance: {avg_final_distance:.4f}\n")

            f.write("\n===== Episode Details =====\n")
            for i in range(len(episode_rewards)):
                f.write(f"Episode {i+1}: Reward={episode_rewards[i]:.4f}, Steps={episode_lengths[i]}, ")
                f.write(f"Success={episode_successes[i]}, Final Distance={episode_final_distances[i]:.4f}\n")

        print(f"Results saved to {results_file}")

    # Visualize results if requested
    if args.visualize:
        visualize_results(episode_rewards, episode_lengths, episode_successes, episode_final_distances, episode_distances)

    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_final_distance': avg_final_distance,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_successes': episode_successes,
        'episode_final_distances': episode_final_distances,
        'episode_distances': episode_distances
    }

def visualize_results(rewards, lengths, successes, final_distances, distances):
    """Visualize test results with matplotlib"""
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
    plt.plot(final_distances, 'r-')
    plt.axhline(y=sum(final_distances)/len(final_distances), color='b', linestyle='--',
                label=f'Avg: {sum(final_distances)/len(final_distances):.4f}')
    plt.axhline(y=args.distance_threshold, color='g', linestyle=':', label=f'Threshold: {args.distance_threshold}')
    plt.title('Final Distances to Target')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()

    # Save the figure
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, f"test_visualization_{timestamp}.png"))

    # Show distance progression for successful episodes
    plt.figure(figsize=(12, 6))
    for i, (dist_list, success) in enumerate(zip(distances, successes)):
        if success:
            plt.plot(dist_list, alpha=0.5, label=f'Episode {i+1}' if i < 5 else None)

    plt.axhline(y=args.distance_threshold, color='r', linestyle='--', label=f'Success Threshold: {args.distance_threshold}')
    plt.title('Distance to Target Progression (Successful Episodes)')
    plt.xlabel('Step')
    plt.ylabel('Distance')
    if sum(successes) > 0:
        if sum(successes) <= 5:
            plt.legend()
        else:
            plt.legend(['First 5 episodes shown'] + [None]*(min(5, sum(successes))-1) + [f'Success Threshold: {args.distance_threshold}'])
    plt.savefig(os.path.join(results_dir, f"distance_progression_{timestamp}.png"))
    print(f"Distance progression saved to {os.path.join(results_dir, f'distance_progression_{timestamp}.png')}")

if __name__ == "__main__":
    try:
        run_test()
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        print("Testing complete")
