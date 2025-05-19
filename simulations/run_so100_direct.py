# run_so100_direct.py
import os
import mujoco
import mujoco.viewer
import time
import numpy as np

def main():
    # Load the MuJoCo model directly
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "so100_6dof/so100.xml")
    print(f"Loading model from: {model_path}")

    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Initialize simulation
    mujoco.mj_resetData(model, data)

    # Print model information
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Number of degrees of freedom: {model.nv}")
    print(f"Number of actuators: {model.nu}")

    # Set GLFW as the rendering backend
    os.environ["MUJOCO_GL"] = "glfw"

    # Launch the viewer directly - this is the key part
    print("Launching viewer...")
    viewer = mujoco.viewer.launch(model, data)
    print("Viewer launched. You should see the SO100 robot now.")

    # The launch function is blocking, so we don't need to do anything else
    # The viewer will stay open until the user closes it

if __name__ == "__main__":
    main()
