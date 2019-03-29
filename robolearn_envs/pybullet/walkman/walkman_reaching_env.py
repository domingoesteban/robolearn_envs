from robolearn_envs.pybullet.task_envs \
    import RobotReachingEnv

from robolearn_envs.pybullet.walkman.walkman \
    import Walkman

from robolearn_envs.utils.transformations import pose_transform

DEFAULT_BODY = 'WB'

DEFAULT_BODY_NAMES = {
    'RA': ['RSoftHand'],
    'LA': ['LSoftHand'],
    'BA': ['RSoftHand', 'LSoftHand'],
    'RL': ['r_sole'],
    'LL': ['l_sole'],
    'WB': ['RSoftHand', 'LSoftHand', 'r_sole', 'l_sole', 'Waist'],
}

LOCO_GOAL_POSES = {
    'RSoftHand': [
        0.26367, -0.52271, 0.69914,
        -0.11695, -0.26127, 0.10429, 0.95246,
    ],
    'LSoftHand': [
        0.26366, 0.52269, 0.69913,
        0.11693, -0.26125, -0.10432, 0.95247,
    ],
    'r_sole': [
        -0.01125, -0.13684, 0.0,
        0., 0., 0., 1.,
    ],
    'l_sole': [
        -0.01126, 0.13684, 0.0,
        0., 0., 0., 1.,
    ],
    'Waist': [
        -0.07831, -0.00086, 1.05812,
        0., 0., 0., 1.,
    ],
}

LOCO_GOAL_OFFSETS = {
    'RSoftHand': [
        # 0., 0., 0.,
        0.3, -0.2, 0.3,
        0., 0., 0., 1.,
    ],
    'LSoftHand': [
        # 0., 0., 0.,
        0.3, 0.2, 0.3,
        0., 0., 0., 1.,
    ],
    'r_sole': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'l_sole': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'Waist': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
}


class WalkmanReachingEnv(RobotReachingEnv):
    def __init__(self,
                 links_names=None,
                 goal_poses=None,
                 active_joints='WB',
                 control_mode='joint_torque',
                 fixed_base=None,
                 is_render=False,
                 sim_timestep=1/240.,
                 frame_skip=1,
                 seed=None,
                 max_time=None,
                 only_position=False,
                 goal_tolerance=0.05,
                 goal_reward_weights=1.e0,
                 ctrl_reward_weight=1.0e-2,
                 ):
        # Robot
        # init_pose_name = 'N'
        init_pose_name = 'loco'
        init_pos = [0., 0., 1.04]
        init_ori = [0.0, 0.0, 0.0, 1.0]

        if fixed_base is None:
            if active_joints in ['RA', 'LA', 'BA', 'UB']:
                fixed_base = True
            else:
                fixed_base = False

        robot = Walkman(
            init_pos=init_pos,
            init_ori=init_ori,
            control_mode=control_mode,
            self_collision=True,
            active_joints=active_joints,
            init_pose_name=init_pose_name,
            fixed_base=fixed_base,
        )

        if links_names is None:
            links_names = DEFAULT_BODY_NAMES[DEFAULT_BODY]

        if goal_poses is None:
            if init_pose_name == 'loco':
                goal_poses = [
                    pose_transform(LOCO_GOAL_OFFSETS[body], LOCO_GOAL_POSES[body])
                    for body in links_names
                ]
            else:
                raise NotImplementedError

        RobotReachingEnv.__init__(
            self,
            robot=robot,
            links_names=links_names,
            goal_poses=goal_poses,
            is_render=is_render,
            sim_timestep=sim_timestep,
            frame_skip=frame_skip,
            seed=seed,
            max_time=max_time,
            only_position=only_position,
            goal_tolerance=goal_tolerance,
            goal_reward_weights=goal_reward_weights,
            ctrl_reward_weight=ctrl_reward_weight,
        )


if __name__ == "__main__":
    import numpy as np

    # render = False
    render = True

    # control_mode = 'joint_torque'
    control_mode = 'joint_tasktorque'

    # active_joints = 'WB'
    active_joints = 'RA'

    # zero_action = False
    zero_action = True

    H = 5000
    env = WalkmanReachingEnv(is_render=render, control_mode=control_mode,
                             active_joints=active_joints)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        if zero_action:
            action = np.zeros_like(env.action_space.low)
        else:
            action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)

        if done:
            print("The environment is done!")
            break

        if render:
            env.render()

    print("Correct!")
