from __future__ import print_function
from builtins import input
import time
import numpy as np

from robolearn_envs.pybullet import CentauroObstacleEnv
import time

np.set_printoptions(precision=2, suppress=True)

T = 10

goal = (0.75, 0.75)
# tgt_pose = (0.6, 0.25, 1.4660)
tgt_pose = None
# goals = None

env_params = {
    'is_render': True,
    # 'is_render': False,
    'active_joints': 'RA',
    # 'control_mode': 'joint_position',
    # 'control_mode': 'joint_torque',
    'control_mode': 'joint_tasktorque',
    'sim_timestep': 0.001,
    'frame_skip': 1,
    # 'obs_distances': True,
    # 'tgt_cost_weight': 1.5,
    # 'fall_cost_weight': 1.0,
    # 'ctrl_cost_weight': 1.0e-2,
    # 'use_log_distances': False,
    # 'log_alpha': 1e-6,
    # 'goal_tolerance': 0.05,
    # 'max_time': None,
}


start = time.time()
env = CentauroObstacleEnv(**env_params)

# for ii in range(5):
#     obs = env.reset()
#     time.sleep(0.2)

# obs = env.reset()

# cond = [0.0, 0.3, np.deg2rad(10), 0.08, 10.1, 0.0]
cond = [0.0, 0.1, np.deg2rad(90), 0.08, 10.1, 0.0]
# env.add_tgt_obst_init_cond(tgt_state=cond[:3], obst_state=cond[3:])

obs = env.reset()

end = time.time()
print('***')
print('elapsed time from INSTANTIATION AND RESET: %d' % int(T/env.dt), end - start)
print('***')


start = time.time()
# input("Press a key to start acting in the environment..")
for t in range(int(T/env.dt)):
    # print('t =', t*env.dt)
    # render_img = env.render()
    action = env.action_space.sample()
    action = np.zeros_like(action)
    # action = np.zeros(env.action_dim)

    # print(obs[tgt_idx])
    # input('temp')

    # print('env_obs', obs)
    # gravity_comp = env.robot.calculate_nonlinear_effects(obs[:env.get_action_dim()])
    # action[0] = 10

    # input('envie un timestep un toqueeee')
    obs, reward, done, info = env.step(action)

    # if done:
    #     break

    # print('tray_pose:', env.get_tray_pose()[:3],
    #       np.rad2deg(env.get_tray_pose(ori='rpy')[3:]))
    # print('reward:', reward)
    # print('rew_vector:', info['reward_vector'])
    # print('---')
    # env.render()

    # time.sleep(env.dt)

end = time.time()
print('***')
print('elapsed time from SAMPLING: %d' % int(T/env.dt), end - start)
print('***')

input("Press a key to finish the script...")
