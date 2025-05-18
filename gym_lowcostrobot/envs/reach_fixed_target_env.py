import os
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
from gymnasium import Env, spaces
from gym_lowcostrobot import ASSETS_PATH, SO101_ASSETS_PATH

class ReachFixedTargetEnv(Env):
    """
    Environment where the robot has to reach a target position with its end-effector.
    Features:
    - Action scaling for larger movements
    - Randomized target positions
    - Shaped rewards for better learning
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        distance_threshold=0.05,
        n_substeps=20,
        render_mode=None,
        robot_model="so101",
        action_scale=2.0,  # Scale factor for actions
        randomize_target=True,  # Whether to randomize target position
        target_range=None,  # Custom target range [x_range, y_range, z_range]
    ):
        # Load the MuJoCo model and data
        if robot_model == "so101":
            assets_path = SO101_ASSETS_PATH
        else:
            assets_path = ASSETS_PATH

        self.model = mujoco.MjModel.from_xml_path(os.path.join(assets_path, "reach_fixed_target.xml"))
        self.data = mujoco.MjData(self.model)

        # Action space with scaling
        self.num_dof = 6
        self.action_scale = action_scale
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_dof,),
            dtype=np.float32
        )

        # Target position settings
        self.randomize_target = randomize_target
        if target_range is None:
            # Default target range in robot base frame
            self.target_range = {
                'x': [0.1, 0.4],  # x range
                'y': [-0.3, 0.3],  # y range
                'z': [0.1, 0.4]   # z range
            }
        else:
            self.target_range = target_range

        # Observation space
        observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(self.num_dof,), dtype=np.float32),
            "arm_qvel": spaces.Box(low=-10.0, high=10.0, shape=(self.num_dof,), dtype=np.float32),
            "target_pos": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
            "ee_pos": spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32),
        }
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        # Environment parameters
        self.distance_threshold = distance_threshold
        self.control_decimation = n_substeps
        self.render_mode = render_mode

        # Target position
        self.target_pos = np.zeros(3, dtype=np.float32)

        # Rendering
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_obs(self):
        """Get observation from the environment."""
        return {
            "arm_qpos": self.data.qpos[:self.num_dof].astype(np.float32),
            "arm_qvel": self.data.qvel[:self.num_dof].astype(np.float32),
            "target_pos": self.target_pos.astype(np.float32),
            "ee_pos": self._get_ee_pos().astype(np.float32),
        }

    def _get_ee_pos(self):
        """Get end-effector position."""
        return self.data.site_xpos[0].copy()

    def _sample_target_pos(self):
        """Sample a random target position within the specified range."""
        return np.array([
            np.random.uniform(*self.target_range['x']),
            np.random.uniform(*self.target_range['y']),
            np.random.uniform(*self.target_range['z'])
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the simulation
        mujoco.mj_resetData(self.model, self.data)

        # Sample a new target position if randomize_target is True
        if self.randomize_target:
            self.target_pos = self._sample_target_pos()
        else:
            self.target_pos = np.array([0.25, 0.25, 0.15], dtype=np.float32)

        # Set the target position in the simulation
        self.model.site_pos[0] = self.target_pos

        # Reset the robot to a neutral position
        self.data.qpos[:self.num_dof] = [0.0, -1.57, 1.57, 0.0, 1.57, 0.0]
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def apply_action(self, action):
        """Apply action to the robot."""
        # Scale the action
        action = np.clip(action, -1.0, 1.0) * self.action_scale

        # Apply the action to the robot
        self.data.ctrl[:] = self.data.qpos[:self.num_dof] + action

    def step(self, action):
        # Apply the action
        self.apply_action(action)

        # Step the simulation
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)

        # Get current state
        obs = self._get_obs()
        ee_pos = obs["ee_pos"]

        # Calculate distance to target
        distance = np.linalg.norm(ee_pos - self.target_pos)

        # Calculate shaped rewards
        reward = self._compute_reward(ee_pos, distance)

        # Check termination conditions
        terminated = distance < self.distance_threshold
        truncated = False  # Can add time limit here if needed

        # Info dictionary
        info = {
            "is_success": 1 if terminated else 0,
            "distance": distance,
            "target_pos": self.target_pos.copy(),
            "ee_pos": ee_pos.copy()
        }

        if self.render_mode == "human":
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, ee_pos, distance):
        """Compute the reward based on the current state."""
        # Base reward for being close to target
        reward = -distance  # Negative distance as base reward

        # Large bonus for reaching the target
        if distance < self.distance_threshold:
            reward += 10.0

        # Additional shaping: reward for moving closer to target
        if hasattr(self, 'prev_distance'):
            reward += 0.1 * (self.prev_distance - distance)  # Reward for moving closer
        self.prev_distance = distance

        # Small penalty for large actions to encourage smoother movements
        # reward -= 0.01 * np.square(action).sum()

        return float(reward)

    def render(self):
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera_id=0)
            return self.rgb_array_renderer.render()
        return None

    def close(self):
        if hasattr(self, 'viewer'):
            self.viewer.close()
