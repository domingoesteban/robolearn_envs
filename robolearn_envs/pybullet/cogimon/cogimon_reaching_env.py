from robolearn_envs.pybullet.task_envs \
    import RobotReachingEnv

from robolearn_envs.pybullet.cogimon.cogimon \
    import Cogimon

from robolearn_envs.utils.transformations import pose_transform

DEFAULT_BODY = 'RA'

DEFAULT_LINK_NAMES = {
    'RA': ['RSoftHand'],
    'LA': ['LSoftHand'],
    'BA': ['RSoftHand', 'LSoftHand'],
    'RL': ['r_sole'],
    'LL': ['l_sole'],
    'WB': ['RSoftHand', 'LSoftHand', 'r_sole', 'l_sole', 'Waist'],
}

LOCO_GOAL_POSES = {
    'RSoftHand': [
        0.2523, -0.0960,  0.99265,
        0.09812, -0.69507,  0.16923, 0.69181,
    ],
    'LSoftHand': [
        0.2523, 0.0960,  0.99265,
        -0.09812, -0.69507,  -0.16923, 0.69181,
    ],
    'r_sole': [
        -0.0873, -0.1019, 0.,
        0., 0., 0., 1.,
    ],
    'l_sole': [
        -0.0873, 0.1044, 0.,
        0., 0., 0., 1.,
    ],
    'Waist': [
        -0.11134, 0., 0.92011,
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


class CogimonReachingEnv(RobotReachingEnv):
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
                 ):
        # Robot
        # init_pose_name = 'N'
        init_pose_name = 'loco'
        if init_pose_name == 'N':
            init_pos = [-5.04157857e-02, 1.87988503e-03, 9.88156105e-01]
            init_ori = [1.14532644e-04, -4.09837742e-02, 3.09986427e-03,
                        9.99154997e-01]
        else:
            init_pos = [-5.68993626e-02, 1.17457386e-03, 9.61556668e-01]
            init_ori = [4.01970997e-05, -2.97989642e-02, 3.99505371e-05,
                        9.99555911e-01]

        if fixed_base is None:
            if active_joints in ['RA', 'LA', 'BA', 'UB']:
                fixed_base = True
            else:
                fixed_base = False

        robot = Cogimon(
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
        )


if __name__ == "__main__":
    import numpy as np

    # render = False
    render = True

    control_mode = 'joint_torque'
    # control_mode = 'joint_tasktorque'

    active_joints = 'WB'
    # active_joints = 'RA'

    # zero_action = False
    zero_action = True

    H = 500

    env = CogimonReachingEnv(is_render=render, control_mode=control_mode,
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
