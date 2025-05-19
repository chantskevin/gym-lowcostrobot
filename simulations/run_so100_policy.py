# run_so100_policy.py
import os
import mujoco
import mujoco.viewer
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import glob
import threading
import pickle

def load_latest_model():
    """Load the latest model from the checkpoints directory."""
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None

    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not checkpoint_files:
        return None

    # Sort by modification time (newest first)
    latest_file = max(checkpoint_files, key=os.path.getmtime)
    print(f"Loading model from {latest_file}")

    try:
        model = SAC.load(latest_file)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def run_policy_visualization():
    """Run a visualization of the current policy."""
    # Set GLFW as the rendering backend
    os.environ["MUJOCO_GL"] = "glfw"

    # Load the MuJoCo model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "so100_6dof/so100.xml")

    # Create the model and data
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_data = mujoco.MjData(mj_model)

    # Initialize simulation
    mujoco.mj_resetData(mj_model, mj_data)

    # Create a simple environment for observation handling
    from so100_env import SO100Env
    env = SO100Env(render_mode=None)

    # Launch the viewer
    print("Launching viewer...")
    viewer = mujoco.viewer.launch(mj_model, mj_data)
    print("Viewer launched. You should see the SO100 robot now.")

    # Variables for policy execution
    current_model = None
    last_check_time = 0
    check_interval = 5  # Check for new model every 5 seconds

    # Create a shared state for communication
    state = {"running": True}

    def policy_execution_loop():
        """Run the policy execution loop in a separate thread."""
        nonlocal current_model, last_check_time

        while state["running"]:
            current_time = time.time()

            # Check for a new model periodically
            if current_time - last_check_time > check_interval:
                new_model = load_latest_model()
                if new_model is not None:
                    current_model = new_model
                    print("Updated to latest model")
                last_check_time = current_time

            # If we have a model, use it to generate actions
            if current_model is not None:
                # Get the current state from the MuJoCo simulation
                qpos = mj_data.qpos.copy()
                qvel = mj_data.qvel.copy()

                # Find the site or body ID for the end effector
                ee_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw")
                ee_pos = mj_data.xpos[ee_id].copy()

                # Combine into observation
                obs = np.concatenate([qpos, qvel, ee_pos])

                # Get action from policy
                action, _ = current_model.predict(obs, deterministic=True)

                # Apply action to simulation
                mj_data.ctrl[:] = action
            else:
                # If no model is available, just apply random actions
                mj_data.ctrl[:] = np.random.uniform(-0.1, 0.1, size=mj_model.nu)

            # Step the simulation
            mujoco.mj_step(mj_model, mj_data)

            # Sleep to control simulation speed
            time.sleep(0.01)

    # Start the policy execution in a separate thread
    policy_thread = threading.Thread(target=policy_execution_loop)
    policy_thread.daemon = True
    policy_thread.start()

    # The viewer.launch call is blocking, so the script will exit when the viewer is closed
    # When that happens, set running to False to stop the policy thread
    state["running"] = False

if __name__ == "__main__":
    run_policy_visualization()
