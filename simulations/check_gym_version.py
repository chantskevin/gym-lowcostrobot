import sys
import importlib.util

def check_package_version(package_name):
    """Check if a package is installed and get its version."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return f"{package_name} is not installed"

        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            return f"{package_name} version: {module.__version__}"
        else:
            return f"{package_name} is installed but version information not found"
    except ImportError:
        return f"{package_name} is not installed"

# Check for both gymnasium and gym
print(check_package_version("gymnasium"))
print(check_package_version("gym"))

# Test the reset method to see which API version is being used
try:
    import gymnasium as gym
    env = gym.make('CartPole-v1')
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        print("Gymnasium API: reset() returns a tuple (observation, info)")
        print(f"Return type: {type(reset_result)}, Length: {len(reset_result)}")
    else:
        print("Gymnasium API: reset() returns just the observation")
        print(f"Return type: {type(reset_result)}")
    env.close()
except ImportError:
    print("Gymnasium not available, trying gym")
    try:
        import gym
        env = gym.make('CartPole-v1')
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            print("Gym API: reset() returns a tuple (observation, info)")
            print(f"Return type: {type(reset_result)}, Length: {len(reset_result)}")
        else:
            print("Gym API: reset() returns just the observation")
            print(f"Return type: {type(reset_result)}")
        env.close()
    except ImportError:
        print("Neither gymnasium nor gym is available")

# Check for stable-baselines3
print(check_package_version("stable_baselines3"))

# Print Python version for reference
print(f"Python version: {sys.version}")
