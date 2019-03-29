from robolearn_envs.pybullet.task_envs import RobotLocomotionEnv

from robolearn_envs.pybullet.coman.coman import Coman


class ComanLocomotionEnv(RobotLocomotionEnv):
    def __init__(self,
                 active_joints='WB',
                 control_mode='joint_torque',
                 is_render=False,
                 sim_timestep=0.001,
                 frame_skip=10,
                 seed=None,
                 max_time=None,
                 base_height_lims=(0.3, 1.5),
                 vel_rew_weight=5e-1,
                 alive_bonus=1e-4,
                 ctrl_rew_weight=1e-4,
                 vel_deviation_rew_weight=5e-3,
                 height_deviation_rew_weight=2e-1,
                 joint_limits_rew_weight=1e-3,
                 impact_rew_weight=1e-3,
                 ):

        # Robot
        init_pos = [0., 0., 0.520]
        init_ori = [0.0, 0.0, 0.0, 1.0]
        robot = Coman(
            init_pos=init_pos,
            init_ori=init_ori,
            control_mode=control_mode,
            self_collision=True,
            active_joints=active_joints,
            fixed_base=False,
        )
        # feet_names = ['l_sole', 'r_sole']
        feet_names = ['LFoot', 'RFoot']
        RobotLocomotionEnv.__init__(
            self,
            robot=robot,
            feet_names=feet_names,
            is_render=is_render,
            sim_timestep=sim_timestep,
            frame_skip=frame_skip,
            seed=seed,
            max_time=max_time,
            base_height_lims=base_height_lims,
            vel_rew_weight=vel_rew_weight,
            alive_bonus=alive_bonus,
            ctrl_rew_weight=ctrl_rew_weight,
            vel_deviation_rew_weight=vel_deviation_rew_weight,
            height_deviation_rew_weight=height_deviation_rew_weight,
            joint_limits_rew_weight=joint_limits_rew_weight,
            impact_rew_weight=impact_rew_weight,
        )


if __name__ == "__main__":
    env = ComanLocomotionEnv(is_render=False)
    print('env_reset')
    env.reset()
    env.render()

    for tt in range(10):
        print('env_step: %02d' % tt)
        env.step(env.action_space.sample())
        env.render()

    print("Correct!")
