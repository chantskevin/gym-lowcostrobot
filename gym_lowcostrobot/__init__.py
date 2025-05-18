import os

from gymnasium.envs.registration import register

__version__ = "0.0.2"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "low_cost_robot_6dof")
SO101_ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets", "so101_arm_6dof")

register(
    id="LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=50,
)

register(
    id="PickPlaceCube-v0",
    entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
    max_episode_steps=50,
)

register(
    id="PushCube-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeEnv",
    max_episode_steps=50,
)

register(
    id="ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    max_episode_steps=50,
)

register(
    id="StackTwoCubes-v0",
    entry_point="gym_lowcostrobot.envs:StackTwoCubesEnv",
    max_episode_steps=50,
)

register(
    id="PushCubeLoop-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeLoopEnv",
    max_episode_steps=50,
)

# Register SO101 arm environments
register(
    id="SO101LiftCube-v0",
    entry_point="gym_lowcostrobot.envs:LiftCubeEnv",
    max_episode_steps=150,
    kwargs={"robot_model": "so101"},
)

register(
    id="SO101PushCube-v0",
    entry_point="gym_lowcostrobot.envs:PushCubeEnv",
    max_episode_steps=150,
    kwargs={"robot_model": "so101"},
)

register(
    id="SO101ReachCube-v0",
    entry_point="gym_lowcostrobot.envs:ReachCubeEnv",
    max_episode_steps=150,
    kwargs={"robot_model": "so101"},
)

register(
    id="SO101StackTwoCubes-v0",
    entry_point="gym_lowcostrobot.envs:StackTwoCubesEnv",
    max_episode_steps=50,
    kwargs={"robot_model": "so101"},
)

register(
    id="SO101PickPlaceCube-v0",
    entry_point="gym_lowcostrobot.envs:PickPlaceCubeEnv",
    max_episode_steps=150,
    kwargs={"robot_model": "so101"},
)

register(
    id="SO101ReachFixedTarget-v0",
    entry_point="gym_lowcostrobot.envs:ReachFixedTargetEnv",
    max_episode_steps=150,
    kwargs={"robot_model": "so101"},
)
