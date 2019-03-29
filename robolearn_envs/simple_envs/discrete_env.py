"""
Code adapted from:
https://github.com/berkeleydeeprlcourse/homework/blob/c1027d83cd542e67ebed982d44666e0d22a00141/hw2/HW2.ipynb
https://sites.google.com/view/deep-rl-bootcamp
"""


import numpy as np

import gym
from gym.utils import seeding


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


class DiscreteEnv(gym.Env):
    """Discrete environment.
    Environment with discrete state and action spaces.

    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary dict of dicts of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def __init__(self, nS, nA, P, isd):
        """

        Args:
            nS (int):
            nA (int):
            P (dict of dict of lists):
                P[s][a] == [(probability, nextstate, reward, done), ...]
            isd (list or np.ndarray): Initial state distribution.
        """
        self.P = P
        self.isd = isd
        self.last_action = None  # for rendering
        self.nS = nS
        self.nA = nA

        self.action_space = gym.spaces.Discrete(self.nA)
        self.observation_space = gym.spaces.Discrete(self.nS)

        self.np_random = None
        self.seed()

        self.s = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.s = categorical_sample(self.isd, self.np_random)
        self.last_action = None
        return self.s

    def step(self, a):
        transitions = self.P[self.s][a]
        # sample next state from p(s'):
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, d = transitions[i]
        self.s = s
        self.last_action = a
        return s, r, d, {'prob': p}

