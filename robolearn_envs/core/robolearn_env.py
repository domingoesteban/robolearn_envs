import numpy as np
import gym
from collections import namedtuple
from gym.utils import seeding

ObsInfo = namedtuple('ObsInfo', ('names', 'dimensions', 'idx'))
StateInfo = namedtuple('StateInfo', ('names', 'dimensions', 'idx'))
ObsType = namedtuple('ObsType', ('name', 'idx'))
StateType = namedtuple('StateType', ('name', 'idx'))


class RobolearnEnv(gym.Env):
    def __init__(self, seed=None):
        self.step_counter = 0
        self.action_space = None
        self.observation_space = None
        self.state_space = None
        self.done = False

        # Set the seed
        self.seed(seed)

    def _set_action(self, action):
        """Applies the given action to the environment.
        """
        raise NotImplementedError()

    def _update_env_obs(self):
        """Updates the environment observations.
        """
        raise NotImplementedError()

    def _update_env_state(self):
        """Updates the environment state.
        """
        raise NotImplementedError()

    def _compute_reward(self, state, action, next_state=None):
        """Compute the reward, based on state, action,
        (and possible next_state)
        """
        raise NotImplementedError()

    def _check_termination(self):
        """Check if some conditions are met.
        """
        raise NotImplementedError()

    def _reset_env(self, *args, **kwargs):
        """Reset everything in the environment"""
        raise NotImplementedError()

    def reset(self, *args, **kwargs):
        """Reset the environment.

        Args:
            *args:
            **kwargs:

        Returns:
            np.ndarray: Initial environment observation

        """
        self.step_counter = 0

        self._reset_env(*args, **kwargs)

        return self._update_env_obs()

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (float[Adim]): action provided by the environment

        Returns:
            next_observation (float[Odim]): agent's observation of the current
                environment
            reward (float): amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case
                further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful
                for debugging, and sometimes learning)

        """
        # Get current environment state: s_t
        state = self._update_env_state()

        # Apply action to environment: a_t
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        # Update env observation and state: s_t+1, o_t+1
        next_observation = self._update_env_obs()
        next_state = self._update_env_state()

        # Calculate if it is done
        self.done, done_info = self._check_termination()

        # Calculate the reward
        reward, reward_info = self._compute_reward(state, action, next_state)

        # Environment information dictionary
        info = {
            'next_state': next_state,
        }
        info.update(done_info)
        info.update(reward_info)

        self.step_counter += 1

        return next_observation, reward, self.done, info

    def seed(self, seed=None, hard=True):
        if hard:
            self.np_random, seed = seeding.np_random(seed)
            # TODO: use the same np_randomizer for robot as for env
            # self.robot.np_random = self.np_random
        else:
            raise NotImplementedError
        return [seed]

    def stop(self):
        """
        Something useful to do. E.g. stop command to robot.
        :return:
        """
        pass

    # ########## #
    # Attributes #
    # ########## #
    @property
    def step_counter(self):
        return self._step_counter

    @step_counter.setter
    def step_counter(self, num_step):
        self._step_counter = num_step

    @property
    def done(self):
        return self._is_done

    @done.setter
    def done(self, done):
        self._is_done = done

    @property
    def obs_dim(self):
        if self.observation_space is None:
            raise AttributeError("No obs_space has been defined for this "
                                 "environment.")
        else:
            return np.prod(self.observation_space.shape)

    @property
    def action_dim(self):
        if self.action_space is None:
            raise AttributeError("No action_space has been defined for this "
                                 "environment.")
        else:
            return np.prod(self.action_space.shape)

    @property
    def state_dim(self):
        if self.state_space is None:
            raise AttributeError("No state_space has been defined for this "
                                 "environment.")
        else:
            return np.prod(self.state_space.shape)

    @property
    def action_bounds(self):
        if self.action_space is None:
            raise AttributeError("No action_space has been defined for this "
                                 "environment.")
        else:
            return np.vstack((self.action_space.low, self.action_space.high))

    @property
    def low_action_bounds(self):
        if self.action_space is None:
            raise AttributeError("No action_space has been defined for this "
                                 "environment.")
        else:
            return self.action_space.low

    @property
    def high_action_bounds(self):
        if self.action_space is None:
            raise AttributeError("No action_space has been defined for this "
                                 "environment.")
        else:
            return self.action_space.high

    @property
    def name(self):
        return type(self).__name__.lower()

    @property
    def np_random(self):
        return self._np_random

    @np_random.setter
    def np_random(self, random_state):
        if not isinstance(random_state, np.random.RandomState):
            raise ValueError("It is not a numpy random state!.")
        self._np_random = random_state
