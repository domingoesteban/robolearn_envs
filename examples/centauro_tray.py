from __future__ import print_function
from builtins import input
import numpy as np
import time

from robolearn_envs.pybullet import CentauroTrayEnv

np.set_printoptions(precision=2, suppress=True)

T = 5

goal = (0.75, 0.75)
# tgt_pose = (0.6, 0.25, 1.4660)
tgt_pose = None
# goals = None

env_params = {
    'is_render': True,
    # 'is_render': False,
    'active_joints': 'RA',
    'control_mode': 'joint_tasktorque',
    # 'control_mode': 'joint_torque',
    'sim_timestep': 0.001,
    'frame_skip': 1,
    'obs_distances': True,
    'tgt_cost_weight': 2.0,
    'balance_cost_weight': 0.6,
    'fall_cost_weight': 0.5,
    'balance_done_cost': 20.0,
    'tgt_done_reward': 100.0,
    'ctrl_cost_weight': 1.0e-4,
    'use_log_distances': True,
    'log_alpha_pos': 1e-2,
    'log_alpha_ori': 1e-2,
    'goal_tolerance': 0.05,
    'min_obj_height': 0.75,
    'max_obj_height': 1.10,
    'max_obj_distance': 0.20,
    'max_time': None,
    'random_tgt': True,
    'random_config': False,
}

start = time.time()
env = CentauroTrayEnv(**env_params)
obs = env.reset()
end = time.time()
print('***')
print('elapsed time from INSTANTIATION AND RESET: %d'
      % int(T/env.dt), end - start)
print('***')

for ii in range(5):
    print("Reseting environment!")
    obs = env.reset()
    print(env._check_termination()[0], obs[:7], '|', env.get_tray_pose()[:3])
    time.sleep(0.2)
    # input("Press a key to continue...")

start = time.time()
# input("Press a key to start acting in the environment..")
for t in range(int(T/env.dt)):
    # print('t =', t*env.dt)
    # print(env.get_tray_pose()[:3])
    render_img = env.render()
    action = env.action_space.sample()
    action = np.zeros_like(action)

    obs, reward, done, info = env.step(action)

    if done:
        print('Env. done. Finishing the script.')
        break

    time.sleep(env.dt)

end = time.time()
print('***')
print('elapsed time from SAMPLING: %d' % int(T/env.dt), end - start)
print('***')

input("Press a key to finish the script...")
