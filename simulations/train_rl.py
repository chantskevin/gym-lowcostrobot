import os
import time
import gymnasium as gym
import numpy as np
import gym_lowcostrobot
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
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

# Dictionary of available environments and their configurations
AVAILABLE_ENVS = {
    "reach_fixed": {
        "id": "SO101ReachFixedTarget-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Reach a fixed target position"
    },
    "reach_cube": {
        "id": "SO101ReachCube-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Reach a cube on the table"
    },
    "push_cube": {
        "id": "SO101PushCube-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Push a cube to a target position"
    },
    "lift_cube": {
        "id": "SO101LiftCube-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Lift a cube from the table"
    },
    "pick_place": {
        "id": "SO101PickPlaceCube-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Pick and place a cube at a target position"
    },
    "stack_cubes": {
        "id": "SO101StackTwoCubes-v0",
        "default_distance_threshold": 0.1,
        "default_n_substeps": 20,
        "default_reward_type": "dense",
        "description": "Stack two cubes"
    }
}

def parse_args():
    parser = argparse.ArgumentParser(description='Train or test a PPO agent for various SO101 robot tasks')

    # Environment selection
    parser.add_argument('--env', type=str, default='reach_fixed',
                        choices=list(AVAILABLE_ENVS.keys()),
                        help=f'Environment to use. Available: {", ".join(AVAILABLE_ENVS.keys())}')

    # Training/testing mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'continue'],
                        help='Mode: train (new model), continue (training), or test (existing model)')

    # Training parameters
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Number of timesteps to train for')
    parser.add_argument('--n_envs', type=int, default=16,
                        help='Number of parallel environments for training')

    # Environment parameters
    parser.add_argument('--distance_threshold', type=float, default=0.1,
                        help='Distance threshold for success (default depends on env)')
    parser.add_argument('--n_substeps', type=int, default=20,
                        help='Number of substeps for physics simulation (default depends on env)')
    parser.add_argument('--reward_type', type=str, default='dense',
                        choices=['sparse', 'dense'],
                        help='Reward type: sparse or dense (default depends on env)')

    # Testing parameters
    parser.add_argument('--test_episodes', type=int, default=25,
                        help='Number of episodes to test for')
    parser.add_argument('--render', action='store_true',
                        help='Enable rendering during testing')

    args = parser.parse_args()

    # Set default values based on selected environment
    env_config = AVAILABLE_ENVS[args.env]
    if args.distance_threshold is None:
        args.distance_threshold = env_config['default_distance_threshold']
    if args.n_substeps is None:
        args.n_substeps = env_config['default_n_substeps']
    if args.reward_type is None:
        args.reward_type = env_config['default_reward_type']

    return args

def get_env_id(env_name):
    """Get the gym environment ID from the environment name"""
    if env_name not in AVAILABLE_ENVS:
        raise ValueError(f"Unknown environment: {env_name}. Available: {', '.join(AVAILABLE_ENVS.keys())}")
    return AVAILABLE_ENVS[env_name]['id']

def create_env_factory(env_id, distance_threshold, n_substeps, reward_type, render_mode=None):
    """Create a factory function for environment creation"""
    def make_env(rank=0):
        def _init():
            env = gym.make(env_id,
                         distance_threshold=distance_threshold,
                         n_substeps=n_substeps,
                         reward_type=reward_type,
                         render_mode=render_mode)
            env.reset(seed=42 + rank)
            return env
        return _init
    return make_env

def setup_directories(env_name):
    """Set up log and model directories for the environment"""
    log_dir = f"logs/{env_name}/"
    os.makedirs(log_dir, exist_ok=True)
    models_dir = os.path.join(log_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return log_dir, models_dir

def create_vectorized_env(make_env, n_envs):
    """Create a vectorized environment with normalization"""
    env = make_vec_env(
        make_env(),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    return env

def setup_callbacks(models_dir, env_name):
    """Set up training callbacks"""
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=models_dir,
        name_prefix=f"{env_name}_model"
    )
    return checkpoint_callback

def create_checkpoint(models_dir, model_path, distance_threshold):
    """Create a checkpoint of the current model"""
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"before_continue_{timestamp}_thresh{distance_threshold}"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Copy the model file
    shutil.copy2(model_path, os.path.join(checkpoint_path, "model.zip"))

    # Copy the VecNormalize stats
    vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        shutil.copy2(vec_normalize_path, os.path.join(checkpoint_path, "vec_normalize.pkl"))
    print(f"✅ Created checkpoint at {checkpoint_path}")

def train(args):
    """Train a new model or continue training an existing one"""
    env_name = args.env
    env_id = get_env_id(env_name)
    log_dir, models_dir = setup_directories(env_name)

    # Create environment factory
    make_env = create_env_factory(
        env_id,
        args.distance_threshold,
        args.n_substeps,
        args.reward_type
    )

    # Create vectorized environment
    env = create_vectorized_env(make_env, args.n_envs)

    # Set up callbacks
    checkpoint_callback = setup_callbacks(models_dir, env_name)

    # Model path for saving/loading
    model_path = os.path.join(models_dir, f"{env_name}_final.zip")

    # Continue training or create new model
    if args.mode == 'continue':
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Training new model instead...")
            args.mode = 'train'
        else:
            print(f"Loading model from {model_path} and continuing training...")
            create_checkpoint(models_dir, model_path, args.distance_threshold)

            # Load the model and VecNormalize stats
            vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
            if os.path.exists(vec_normalize_path):
                env = VecNormalize.load(vec_normalize_path, env)
                env.norm_obs=False
                print("Loaded normalization stats")

            model = PPO.load(model_path, env=env, device=device)
            print("Model loaded successfully")

    if args.mode == 'train':
        print(f"Creating new model for {env_name}...")
        # Create a new model with appropriate hyperparameters
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
            },
        )

    # Train the model
    print(f"Starting training for {args.timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"{env_name}_training"
    )

    # Save the final model and normalization stats
    model.save(model_path)
    env.save(os.path.join(models_dir, "vec_normalize.pkl"))

    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    print(f"Model saved to {model_path}")

    env.close()
    return model_path

def test(args):
    """Test a trained model"""
    env_name = args.env
    env_id = get_env_id(env_name)
    _, models_dir = setup_directories(env_name)

    # Model path for loading
    model_path = os.path.join(models_dir, f"{env_name}_final.zip")

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Please train a model first using --mode train")
        return

    print(f"Loading model from {model_path} for testing...")

    # Create test environment (single env)
    render_mode = "human" if args.render else None
    make_env = create_env_factory(
        env_id,
        args.distance_threshold,
        args.n_substeps,
        args.reward_type,
        render_mode=render_mode
    )

    test_env = DummyVecEnv([make_env()])

    # Try to load normalization stats
    vec_normalize_path = os.path.join(models_dir, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        print(f"Loading normalization stats from {vec_normalize_path}")
        test_env = VecNormalize.load(vec_normalize_path, test_env)
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

    # Load the model
    model = PPO.load(model_path, device=device)

    # Test the model
    obs = test_env.reset()
    # Handle different gym versions with different reset() return types
    if isinstance(obs, tuple):
        obs = obs[0]  # First element is observation

    max_steps = 150
    max_episodes = args.test_episodes
    episodes = 0
    success_count = 0
    timeout_count = 0
    total_reward = 0

    print(f"Testing model for {max_episodes} episodes...")

    for i in range(max_steps * max_episodes):
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

def main():
    args = parse_args()

    print(f"Selected environment: {args.env} ({AVAILABLE_ENVS[args.env]['description']})")
    print(f"Mode: {args.mode}")
    print(f"Parameters: distance_threshold={args.distance_threshold}, n_substeps={args.n_substeps}, reward_type={args.reward_type}")

    if args.mode in ['train', 'continue']:
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__ == "__main__":
    main()
