from __future__ import print_function
from builtins import input
import time
import numpy as np

from robolearn_envs.pybullet import Pusher2DGoalCompoEnv

T = 50

goal = (0.75, 0.75)
# tgt_pose = (0.6, 0.25, 1.4660)
tgt_pose = None
# goals = None

env_params = {
    'is_render': True,
    'obs_distances': True,  # If True obs contain 'distance' vectors instead poses
    'obs_with_img': False,
    'obs_with_ori': False,
    'goal_pose': (0.65, 0.65),
    'rdn_goal_pose': True,
    'tgt_pose': (0.5, 0.25, 1.4660),
    'rdn_tgt_object_pose': True,
    'robot_config': None,
    'rdn_robot_config': True,
    'tgt_cost_weight': 3.0,
    'goal_cost_weight': 1.0,
    'ctrl_cost_weight': 1.0e-3,
    'goal_tolerance': 0.05,
    # 'max_time=PATH_LENGTH*DT,
    'max_time': None,
    'sim_timestep': 0.001,
    'frame_skip': 10,
    'subtask': None,
    'seed': 10,
    'sequential_reward': True,
}

env = Pusher2DGoalCompoEnv(**env_params)
print('Obs dim:', env.obs_dim)
print('State dim:', env.state_dim)
print('Act dim:', env.action_dim)

for ii in range(5):
    obs = env.reset()
    print(obs)
    time.sleep(0.2)

obs = env.reset()

# input("Press a key to start acting in the environment..")
for t in range(int(T/env.dt)):
    print('t =', t*env.dt)
    render_img = env.render()
    action = env.action_space.sample()
    # action = np.zeros_like(action)
    # action[0] = 1
    
    obs, reward, done, info = env.step(action)
    print('reward=', reward)
    print('reward_multigoal=', info['reward_multigoal'])
    print('---')
    # env.render()
    time.sleep(env.dt)

input("Press a key to finish the script...")
