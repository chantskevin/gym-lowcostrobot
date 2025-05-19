import mujoco
import mujoco.viewer

# 1. Load your model and data
model = mujoco.MjModel.from_xml_path("./gym_so101_env/gym_lowcostrobot/assets/so101_arm_6dof/reach_fixed_target.xml")
data  = mujoco.MjData(model)

# 2. Launch the built-in viewer
mujoco.viewer.launch(model, data)