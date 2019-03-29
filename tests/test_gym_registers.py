import gym
import numpy as np
from context import robolearn_envs

all_envs = gym.envs.registry.all()
robolearn_env_ids = [env_spec.id for env_spec in all_envs
                     if env_spec.id.startswith('RoboLearn-')]


for env_id in robolearn_env_ids:
    print('-'*15)
    print("Environment: %s" % env_id)
    env = gym.make(env_id)

    obs = env.reset()
    print("\t Reset: OK")

    for t in range(50):
        print('\t Step %d: OK' % t)
        obs, reward, done, info = \
            env.step(np.zeros(np.prod(env.action_space.shape)))

    env.close()
    print("\t Close: OK")
