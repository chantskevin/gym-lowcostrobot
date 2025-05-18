import os

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_lowcostrobot import ASSETS_PATH, SO101_ASSETS_PATH


class ReachFixedTargetEnv(Env):
    """
    ## Description

    The robot has to reach a fixed target position with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 3     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"target_pos"`: the position of the target, as (x, y, z)

    ## Reward

    For a dense reward type, it is the negative distance between the end effector and the target.
    For a sparse reward type, it is -1 when the distance between the end effector and the target exceeds a specified distance_threshold.

    ## Arguments

    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `reward_type (str)`: the reward type, can be "sparse" or "dense", default is "sparse".
    - `block_gripper (bool)`: whether to block the gripper, default is True.
    - `distance_threshold (float)`: the threshold for the distance between the end effector and the target, default is 0.05.
    - `n_substeps (int)`: the number of substeps for each step, default is 20.
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    - `robot_model (str)`: the robot model, can be "so101", default is "so101".
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        action_mode="joint",
        reward_type="sparse",
        block_gripper=True,
        distance_threshold=0.05,
        n_substeps=20,
        render_mode=None,
        robot_model="so101",
    ):
        # Select the appropriate assets path based on the robot model
        if robot_model == "so101":
            assets_path = SO101_ASSETS_PATH
        else:
            assets_path = ASSETS_PATH

        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(assets_path, "reach_fixed_target.xml"))
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        self.block_gripper = block_gripper
        action_shape = {"joint": 5, "ee": 3}[self.action_mode]
        action_shape += 0 if self.block_gripper else 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        # Set other parameters
        self.num_dof = 6
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.control_decimation = n_substeps

        # Set the observation space
        observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dof,), dtype=np.float32),
            "arm_qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_dof,), dtype=np.float32),
            "target_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        }
        self.observation_space = spaces.Dict(observation_subspaces)

        # Set the fixed target position
        self.target_pos = np.array([0.0, 0.3, 0.3])  # Fixed target position

        # Try to find a target site if it exists
        self.target_site_id = None
        try:
            self.target_site_id = self.model.site("target").id
            # Note: We can't directly modify site positions in this version of MuJoCo
            # We'll visualize the target in a different way
        except Exception:
            # Target site doesn't exist, that's okay
            pass

        # Set up rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=480, width=640)

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, gripper]
        """
        # Scale the action
        action = np.clip(action, -1.0, 1.0)

        if self.action_mode == "joint":
            # Scale the action to the joint limits
            joint_range = self.model.jnt_range[:self.num_dof]
            joint_center = 0.5 * (joint_range[:, 1] + joint_range[:, 0])
            joint_extent = 0.5 * (joint_range[:, 1] - joint_range[:, 0])

            # Scale the action to the joint range
            if self.block_gripper:
                target_arm_qpos = joint_center[:self.num_dof-1] + action * joint_extent[:self.num_dof-1]
                target_gripper_pos = np.array([0.0])  # Closed gripper
            else:
                target_arm_qpos = joint_center[:self.num_dof-1] + action[:-1] * joint_extent[:self.num_dof-1]
                target_gripper_pos = np.array([action[-1]])  # Gripper position from action

            # Combine arm and gripper targets
            target_qpos = np.append(target_arm_qpos, target_gripper_pos)

        elif self.action_mode == "ee":
            # Get the current end effector position
            ee_site_id = self.model.site("end_effector_site").id
            mujoco.mj_forward(self.model, self.data)
            current_ee_pos = self.data.site(ee_site_id).xpos.copy()

            # Scale the action to a reasonable displacement
            ee_displacement = action[:3] * 0.05  # 5cm maximum displacement per step

            # Compute the target end effector position
            target_ee_pos = current_ee_pos + ee_displacement

            # Use inverse kinematics to compute the target joint positions
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, None, ee_site_id)
            jac = jacp[:, :self.num_dof-1]  # Exclude gripper

            # Compute the update using damped least squares
            JTJ = jac.T @ jac
            JTJ_reg = JTJ + 0.1 * np.eye(self.num_dof-1)
            JTe = jac.T @ (target_ee_pos - current_ee_pos)

            # Solve the system
            dq = np.linalg.solve(JTJ_reg, JTe)

            # Update the joint positions
            target_arm_qpos = self.data.qpos[:self.num_dof-1] + dq

            # Set the gripper position
            if self.block_gripper:
                target_gripper_pos = np.array([0.0])  # Closed gripper
            else:
                target_gripper_pos = np.array([action[-1]])  # Gripper position from action

            # Combine arm and gripper targets
            target_qpos = np.append(target_arm_qpos, target_gripper_pos)
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()

    def get_observation(self):
        # Get the joint positions and velocities
        observation = {
            "arm_qpos": self.data.qpos[:self.num_dof].astype(np.float32),
            "arm_qvel": self.data.qvel[:self.num_dof].astype(np.float32),
            "target_pos": self.target_pos.astype(np.float32),
        }
        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[:self.num_dof] = robot_qpos

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        # Note: We don't need to update the target site position as it's fixed in the XML

        # Initialize previous distance for reward shaping
        ee_id = self.model.site("end_effector_site").id
        ee_pos = self.data.site(ee_id).xpos.copy()
        self.prev_distance = np.linalg.norm(ee_pos - self.target_pos)

        return self.get_observation(), {}

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation()

        # Get the position of the end effector
        ee_id = self.model.site("end_effector_site").id
        ee_pos = self.data.site(ee_id).xpos.copy()

        # Calculate the distance to the target
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # Determine if the task is successful
        is_success = distance < self.distance_threshold

        info = {
            "is_success": is_success,
            "distance": distance,
        }

        # Determine termination and reward
        terminated = is_success
        truncated = False

        # Compute the reward - using a shaped reward to help learning
        if self.reward_type == "sparse":
            reward = 0.0 if is_success else -1.0
        else:  # dense reward with shaping
            # Base reward is negative distance
            reward = -distance

            # Add a bonus for getting closer to the target
            if hasattr(self, 'prev_distance'):
                # Reward improvement in distance
                distance_improvement = self.prev_distance - distance
                reward += 5.0 * distance_improvement  # Scale the improvement reward

            # Add a large bonus for success
            if is_success:
                reward += 10.0

        # Store the current distance for the next step
        self.prev_distance = distance

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()