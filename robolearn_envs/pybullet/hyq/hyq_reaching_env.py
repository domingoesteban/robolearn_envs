from robolearn_envs.pybullet.task_envs \
    import RobotReachingEnv

from robolearn_envs.pybullet.hyq.hyq \
    import Hyq

from robolearn_envs.utils.transformations import pose_transform

DEFAULT_BODY = 'WB'

DEFAULT_BODY_NAMES = {
    'RA': ['rh_lowerleg'],
    'LA': ['lh_lowerleg'],
    'BA': ['rh_lowerleg', 'lh_lowerleg'],
    'UB': ['rh_lowerleg', 'lh_lowerleg'],
    'RL': ['rf_lowerleg'],
    'LL': ['lf_lowerleg'],
    'WB': ['rh_lowerleg', 'lh_lowerleg', 'rf_lowerleg', 'lf_lowerleg', 'trunk'],
}

N_GOAL_POSES = {
    'rh_lowerleg': [
        -0.22037, -0.29191, 0.1802,
        -0.30738, 0.71069, 0.58099, 0.25077,
    ],
    'lh_lowerleg': [
        -0.22037, 0.2921,  0.18024,
        -0.25144, 0.5811, 0.7106, 0.30684,
    ],
    'rf_lowerleg': [
        0.22037, -0.2921, 0.18024,
        -0.7106, 0.30684, 0.25144, 0.5811,
    ],
    'lf_lowerleg': [
        0.22037, 0.29191, 0.17878,
        -0.58099, 0.25077, 0.30738, 0.71069,
    ],
    'trunk': [
        0.01, 0., 0.60304,
        0., 0., 0., 1.,
    ],
}

N_GOAL_OFFSETS = {
    'rh_lowerleg': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'lh_lowerleg': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'rf_lowerleg': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'lf_lowerleg': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
    'trunk': [
        0., 0., 0.,
        0., 0., 0., 1.,
    ],
}


class HyqReachingEnv(RobotReachingEnv):
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
                 base_height_lims=(0.40, 2.0),
                 ):
        # Robot
        init_pose_name = 'H'
        if init_pose_name == 'N':
            init_pos = [0.0, 0.0, 0.775]
            init_ori = [0.0, 0.0, 0.0, 1.0]
        elif init_pose_name == 'H':
            init_pos = [0.0, 0.0, 0.60]
            init_ori = [0.0, 0.0, 0.0, 1.0]
        else:
            init_pos = [0.0, 0.0, 0.758]
            init_ori = [0.0, 0.0, 0.0, 1.0]

        if fixed_base is None:
            if active_joints in ['RA', 'LA', 'BA', 'UB']:
                fixed_base = True
            else:
                fixed_base = False

        robot = Hyq(
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
            if init_pose_name == 'H':
                goal_poses = [
                    pose_transform(N_GOAL_OFFSETS[body], N_GOAL_POSES[body])
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
            base_height_lims=base_height_lims,
        )


if __name__ == "__main__":
    env = HyqReachingEnv(is_render=False)
    print('env_reset')
    env.reset()
    env.render()

    for tt in range(10):
        print('env_step: %02d' % tt)
        env.step(env.action_space.sample())
        env.render()

    print("Correct!")
