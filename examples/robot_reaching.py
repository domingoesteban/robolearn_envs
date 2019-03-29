from __future__ import print_function
from builtins import input
import numpy as np
import time
import argparse

from robolearn_envs.pybullet import CentauroReachingEnv
from robolearn_envs.pybullet import CogimonReachingEnv
from robolearn_envs.pybullet import WalkmanReachingEnv
from robolearn_envs.pybullet import HyqReachingEnv

np.set_printoptions(precision=5, suppress=True)

# Time parameters
T = 5
# sim_timestep = 0.001
# frame_skip = 10
sim_timestep = 1/240.
# sim_timestep = 1.e-4
frame_skip = 1

# Available environments
robot_envs = {
    'centauro': CentauroReachingEnv,
    'cogimon': CogimonReachingEnv,
    'walkman': WalkmanReachingEnv,
    'hyq': HyqReachingEnv,
}

# Common environment parameters
env_params = {
    'is_render': True,
    'active_joints': 'RA',
    'control_mode': 'joint_torque',
    'sim_timestep': sim_timestep,
    'frame_skip': frame_skip,
}


def run_environment(args):
    # Get environment from arguments
    robot_name = args.robot_name
    environment = robot_envs[robot_name]

    # Modify control mode from arguments
    env_params['control_mode'] = 'joint_' + args.control_mode.lower()

    # Modify active joints from arguments
    env_params['active_joints'] = args.active_joints.upper()

    # Instantiate environment and reset
    start = time.time()
    env = environment(**env_params)
    obs = env.reset()
    end = time.time()
    print('***')
    print('elapsed time from INSTANTIATION AND RESET: %d'
          % int(T/env.dt), end - start)
    print('***')

    # # Reset the environment several times to check initial conditions
    # for ii in range(50):
    #     obs = env.reset()
    #     # input("press a key")
    #     time.sleep(0.2)

    # input("Press a key to start acting in the environment..")
    print('Link poses:\n', *['\t %s \n' % pos for pos in env.get_robot_body_poses()])
    start = time.time()
    for t in range(int(T/env.dt)):
        render_img = env.render()
        if env_params['control_mode'] == 'joint_position':
            action = env.robot_initial_configuration
        else:
            action = np.zeros(env.action_dim)

        obs, reward, done, info = env.step(action)

        if done:
            print('Env. done. Finishing the script.')
        #     break
        print('Reward: %.2f\n' % reward,
              'Link poses:\n', *['\t %s \n' % pos for pos in env.get_robot_body_poses()])
        # print('--')

        time.sleep(env.dt)

    end = time.time()
    print('***')
    print('elapsed time from SAMPLING: %d' % int(T/env.dt), end - start)
    print('***')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example of a locomotion environment for a specific robot."
    )
    parser.add_argument(
        'robot_name',
        type=str,
        nargs='?',
        default='cogimon',
        help="the robot's name (default: %(default)s)",
        choices=robot_envs.keys()
    )
    parser.add_argument(
        '--control_mode',
        '-m',
        type=str,
        # nargs='?',
        default=env_params['control_mode'][len('joint_'):],
        help="the robot's control mode (default: %(default)s)",
        choices=['position', 'velocity', 'torque', 'tasktorque']
    )
    parser.add_argument(
        '--active_joints',
        '-j',
        type=str,
        # nargs='?',
        default=env_params['active_joints'],
        help="the robot's active joints (default: %(default)s)",
        choices=['WB', 'UB', 'RA', 'LA', 'BA', 'RL', 'LL']
    )
    args = parser.parse_args()

    run_environment(args)

input("Press a key to finish the script...")
