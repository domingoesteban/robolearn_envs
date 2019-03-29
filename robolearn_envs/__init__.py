from gym.envs.registration import register

# TEMPLATE:
# register(
#     id='RoboLearn-CentauroEnv-v0',
#     entry_point='robolearn_envs.pybullet:CentauroEnv',
#     # tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
#     #     # max_episode_steps=1000,
#     #     # #timestep_limit=1000,
#     #     # reward_threshold=2500.0
# )

# -----------------------------------------------------------------
# TOY
# -----------------------------------------------------------------
register(
    id='RoboLearn-Navigation2D-v0',
    entry_point='robolearn_envs.simple_envs:Navigation2dEnv',
    kwargs={
    },
)
# register(
#     id='RoboLearn-Reacher2D3DofObstacleEnv-v0',
#     entry_point='robolearn_envs.pybullet:Reacher2D3DofObstacleEnv',
#     kwargs={'is_render': False,
#             'obs_with_img': False,
#             'rdn_tgt_pose': True,
#             'obs_distances': True},
# )
# register(
#     id='RoboLearn-Pusher2D3DofObstacleBulletEnvRender-v0',
#     entry_point='robolearn_envs.pybullet:Reacher2D3DofObstacleEnv',
#     kwargs={'is_render': True,
#             'obs_with_img': False,
#             'rdn_tgt_pose': True,
#             'obs_distances': True},
# )


# -----------------------------------------------------------------
# IIT
# -----------------------------------------------------------------

# ######## #
# CENTAURO #
# ######## #
register(
    id='RoboLearn-CentauroLocomotion-v0',
    entry_point='robolearn_envs.pybullet:CentauroLocomotionEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-CentauroLocomotionRender-v0',
    entry_point='robolearn_envs.pybullet:CentauroLocomotionEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-CentauroObstacle-v0',
    entry_point='robolearn_envs.pybullet:CentauroObstacleEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'RA',
        'control_mode': 'joint_tasktorque',
        'sim_timestep': 0.001,
        'frame_skip': 1,
        'obs_distances': True,
        'goal_tolerance': 0.02,
        'max_time': None,
    },
)
register(
    id='CentauroObstacleRender-v0',
    entry_point='robolearn_envs.pybullet:CentauroObstacleEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'RA',
        'control_mode': 'joint_tasktorque',
        'sim_timestep': 0.001,
        'frame_skip': 1,
        'obs_distances': True,
        'goal_tolerance': 0.02,
        'max_time': None,
    },
)
register(
    id='RoboLearn-CentauroReaching-v0',
    entry_point='robolearn_envs.pybullet:CentauroReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': False,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)
register(
    id='RoboLearn-CentauroReachingRender-v0',
    entry_point='robolearn_envs.pybullet:CentauroReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': True,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)

# ####### #
# WALKMAN #
# ####### #
register(
    id='RoboLearn-WalkmanLocomotion-v0',
    entry_point='robolearn_envs.pybullet:WalkmanLocomotionEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-WalkmanLocomotionRender-v0',
    entry_point='robolearn_envs.pybullet:WalkmanLocomotionEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-WalkmanReaching-v0',
    entry_point='robolearn_envs.pybullet:WalkmanReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': False,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)
register(
    id='RoboLearn-WalkmanReachingRender-v0',
    entry_point='robolearn_envs.pybullet:WalkmanReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': True,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)

# ####### #
# COGIMON #
# ####### #
register(
    id='RoboLearn-CogimonLocomotion-v0',
    entry_point='robolearn_envs.pybullet:CogimonLocomotionEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-CogimonLocomotionRender-v0',
    entry_point='robolearn_envs.pybullet:CogimonLocomotionEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-CogimonReaching-v0',
    entry_point='robolearn_envs.pybullet:CogimonReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': False,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)
register(
    id='RoboLearn-CogimonReachingRender-v0',
    entry_point='robolearn_envs.pybullet:CogimonReachingEnv',
    kwargs={
        'links_names': None,
        'goal_poses': None,
        'active_joints': 'RA',
        'control_mode': 'joint_torque',
        'fixed_base': None,
        'is_render': True,
        'sim_timestep': 0.001,
        'frame_skip': 10,
        'seed': None,
        'max_time': 10,
        'only_position': False,
        'goal_tolerance': 0.05,
        'goal_reward_weights': 1.e0,
        'ctrl_reward_weight': 1.0e-2,
    },
)

# ##### #
# COMAN #
# ##### #
register(
    id='RoboLearn-ComanLocomotion-v0',
    entry_point='robolearn_envs.pybullet:ComanLocomotionEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-ComanLocomotionRender-v0',
    entry_point='robolearn_envs.pybullet:ComanLocomotionEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)


# ### #
# HYQ #
# ### #
register(
    id='RoboLearn-HyqLocomotion-v0',
    entry_point='robolearn_envs.pybullet:HyqLocomotionEnv',
    kwargs={
        'is_render': False,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
register(
    id='RoboLearn-HyqLocomotionRender-v0',
    entry_point='robolearn_envs.pybullet:HyqLocomotionEnv',
    kwargs={
        'is_render': True,
        'active_joints': 'WB',
        'control_mode': 'joint_torque',
        'sim_timestep': 0.001,
        'frame_skip': 10,

    },
)
