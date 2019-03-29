from __future__ import print_function
from builtins import input
import time
import numpy as np

from robolearn_envs.pybullet import Reacher2DObstacleEnv

T = 5
env = Reacher2DObstacleEnv(is_render=True,
                           obs_with_img=False,
                           # rdn_tgt_pose=True,
                           rdn_tgt_pose=True,
                           obs_distances=True)

# for _ in range(3):
#     obs = env.reset()
#     input('AAAA')

obs = env.reset()
# input("Press a key to start acting in the environment..")
for t in range(int(T/env.dt)):
    print('t =', t*env.dt)
    render_img = env.render()
    action = env.action_space.sample() * 10
    
    action = np.zeros_like(action)
    action[0] = env.action_bounds[0][1]
    
    obs, reward, done, info = env.step(action)
    print('obs:', obs, '\naction:', action, '\nreward:', reward, '\ninfo', info)
    # input('aaa')
    # env.render()
    # time.sleep(env.dt)

input("Press a key to finish the script...")
