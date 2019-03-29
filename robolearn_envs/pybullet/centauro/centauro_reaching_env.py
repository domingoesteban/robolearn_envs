from robolearn_envs.pybullet.task_envs \
    import RobotReachingEnv

from robolearn_envs.pybullet.centauro.centauro \
    import Centauro

from robolearn_envs.utils.transformations import pose_transform

DEFAULT_BODY = 'RA'

DEFAULT_LINK_NAMES = {
    'RA': ['arm2_8'],
    'LA': ['arm1_8'],
    'BA': ['arm2_8', 'arm1_8'],
    'FR': ['wheel_2'],
    'FL': ['wheel_1'],
    'BL': ['wheel_4'],
    'BR': ['wheel_3'],
    'RL': ['wheel_4', 'wheel_2'],
    'LL': ['wheel_3', 'wheel_1'],
    'WB': ['arm2_8', 'arm1_8', 'wheel_4', 'wheel_3', 'wheel_2', 'wheel_1', 'pelvis'],
}

LOCO_GOAL_POSES = {
    'arm2_8': [
        0.76798, -0.44165, 0.83033,
        -0.17506, -0.76756, -0.00643, 0.61658,
    ],
    'arm1_8': [
        0.75674, 0.45663, 0.83032,
        0.18465, -0.7653, 0.0142, 0.61646,
    ],
    'wheel_4': [
        -0.35142, -0.38273, 0.07895,
        0.34267, -0.61887, 0.62254, 0.33469,
    ],
    'wheel_3': [
        -0.36097, 0.37021, 0.07897,
        -0.61845, -0.34286, -0.33518, 0.62259,
    ],
    'wheel_2': [
        0.36151, -0.3741, 0.07895,
        0.34295, -0.61847, 0.62248, 0.33524,
    ],
    'wheel_1': [
        0.3521, 0.37874, 0.07895,
        -0.61888, -0.34257, -0.33472, 0.62258,
    ],
    'pelvis': [
        -0.01413, -0.00534, 0.81153,
        0., 0., 0., 1.,
    ],
}

LOCO_GOAL_OFFSETS = {
    'arm2_8': [
        # 0., 0., 0.,
        0.2, -0.2, 0.2,
        0., 0., 0., 1.,
    ],
    'arm1_8': [
        # 0., 0., 0.,
        0.2, 0.2, 0.2,
        0., 0., 0., 1.,
    ],
    'wheel_4': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'wheel_3': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'wheel_2': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'wheel_1': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'pelvis': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
}


class CentauroReachingEnv(RobotReachingEnv):
    def __init__(self,
                 links_names=None,
                 goal_poses=None,
                 active_joints='RA',
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
                 base_height_lims=(0.40, 1.5),
                 ):
        # Robot
        # init_pose_name = 'N'
        init_pose_name = 'loco'
        if init_pose_name == 'N':
            init_pos = [0, 0, 0.7975]
            init_ori = [-0., 0., 0., 1.]
        else:
            init_pos = [0, 0, 0.7975]
            init_ori = [-0., 0., 0., 1.]

        if fixed_base is None:
            if active_joints in ['RA', 'LA', 'BA', 'UB']:
                fixed_base = True
            else:
                fixed_base = False

        robot = Centauro(
            init_pos=init_pos,
            init_ori=init_ori,
            control_mode=control_mode,
            self_collision=True,
            active_joints=active_joints,
            init_pose_name=init_pose_name,
            fixed_base=fixed_base,
        )

        if links_names is None:
            links_names = DEFAULT_LINK_NAMES[DEFAULT_BODY]
        elif isinstance(links_names, str):
            links_names = [links_names]

        if goal_poses is None:
            if init_pose_name == 'loco':
                goal_poses = [
                    pose_transform(LOCO_GOAL_OFFSETS[link], LOCO_GOAL_POSES[link])
                    for link in links_names
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
            base_height_lims=base_height_lims,
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

    env = CentauroReachingEnv(is_render=render, control_mode=control_mode,
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
