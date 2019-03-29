from __future__ import print_function
from builtins import input
import time
import numpy as np

from robolearn_envs.pybullet import Reacher2DGoalCompoEnv

T = 50

goal = (0.75, 0.75)
# tgt_pose = (0.6, 0.25, 1.4660)
tgt_pose = None
# goals = None

env_params = {
    'is_render': True,
    # 'obs_distances': False,
    'obs_distances': True,
    'obs_with_img': False,
    # 'obs_with_ori': True,
    'obs_with_ori': False,
    'obs_with_goal': True,
    # 'obs_with_goal': False,
    # 'goal_pose': (0.65, 0.65),
    'goal_pose': (0.65, 0.35),
    'rdn_goal_pos': True,
    # 'rdn_goal_pos': False,
    'robot_config': None,
    'rdn_robot_config': True,
    'goal_cost_weight': 4.0e0,
    'ctrl_cost_weight': 5.0e-1,
    'goal_tolerance': 0.03,
    'use_log_distances': True,
    'log_alpha': 1e-6,
    # 'max_time': PATH_LENGTH*DT,
    'max_time': None,
    'sim_timestep': 1.e-3,
    'frame_skip': 10,
    'half_env': True,
    'subtask': None,
    'seed': 610,
}

env = Reacher2DGoalCompoEnv(**env_params)
print('Obs dim:', env.obs_dim)
print('State dim:', env.state_dim)
print('Act dim:', env.action_dim)

for ii in range(5):
    obs = env.reset()
    print('obs=', obs)
    time.sleep(0.2)

obs = env.reset()
# input("Press a key to start acting in the environment..")
for t in range(int(T/env.dt)):
    # print('t =', t*env.dt)
    render_img = env.render()
    action = env.action_space.sample()
    # action = np.zeros_like(action)
    # action[0] = 1
    
    obs, reward, done, info = env.step(action)
    # print('reward=', reward)
    # print('reward_vect=', info['reward_vector'])
    print('reward_multi=', info['reward_multigoal'])
    # print('obs', obs)
    # print('obs', obs[env.get_obs_info('target')['idx']])
    # print('---')
    # env.render()
    time.sleep(env.dt)

input("Press a key to finish the script...")
