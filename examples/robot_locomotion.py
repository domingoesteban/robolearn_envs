from __future__ import print_function
from builtins import input
import numpy as np
import time
import argparse

from robolearn_envs.pybullet import CentauroLocomotionEnv
from robolearn_envs.pybullet import CogimonLocomotionEnv
from robolearn_envs.pybullet import WalkmanLocomotionEnv
from robolearn_envs.pybullet import ComanLocomotionEnv
from robolearn_envs.pybullet import HyqLocomotionEnv

np.set_printoptions(precision=5, suppress=True)

# Time parameters
T = 5
sim_timestep = 0.001
frame_skip = 10

# Available environments
robot_envs = {
    'centauro': CentauroLocomotionEnv,
    'cogimon': CogimonLocomotionEnv,
    'coman': ComanLocomotionEnv,
    'walkman': WalkmanLocomotionEnv,
    'hyq': HyqLocomotionEnv,
}

# Common environment parameters
env_params = {
    'is_render': True,
    'active_joints': 'WB',
    'control_mode': 'joint_position',
    # 'control_mode': 'joint_velocity',
    # 'control_mode': 'joint_torque',
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
        print('Base pose:', env.robot_base_pose)
        # print('Init', env.robot.initial_configuration)
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
        default='centauro',
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
