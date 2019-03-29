import numpy as np
from robolearn_envs.reward_functions.utils import convert_reward_cost_fcn


def quadratic_action_cost(action, weights=None, gradient=False, hessian=False):
    """Quadratic action cost

        :math:`cost = 0.5 \sum_{i}^{dimA} (a_i * w_i)^2`

    Args:
        action (np.ndarray):
        weights (np.ndarray or float or None):
        gradient (bool): Calculate cost gradient.
        hessian (bool): Calculate cost Hessian.

    Returns:
        cost
        cost info (dict):

    """
    if weights is None:
        weights = np.ones_like(action)
    else:
        # Check if iterable
        try:
            iter(weights)
            if len(weights) != len(action):
                raise ValueError("weights length different than action dim"
                                 "(%02d != %02d)" % (weights,
                                                     len(action)))
            weights = weights
        except TypeError as te:
            weights = np.repeat(weights, len(action))

    cost = 0.5 * np.sum(np.square(action*weights))

    cost_info = {}

    if gradient:
        cost_info['gradient'] = action*weights

    if hessian:
        cost_info['hessian'] = np.diag(weights)

    return cost, cost_info


def exp_quadratic_action_reward(action, weights=None, gradient=False, hessian=False):
    """Exponential quadratic action reward

        :math:`reward = 0.5 exp(\sum_{i}^{dimA} (a_i * w_i)^2)`

    Args:
        action (np.ndarray):
        weights (np.ndarray or float or None):
        gradient (bool): Calculate reward gradient.
        hessian (bool): Calculate reward Hessian.

    Returns:

    """
    # TODO: Check if t he reward is correct
    if weights is None:
        weights = np.ones_like(action)
    else:
        # Check if iterable
        try:
            iter(weights)
            if len(weights) != len(action):
                raise ValueError("weights length different than action dim"
                                 "(%02d != %02d)" % (weights,
                                                     len(action)))
            weights = weights
        except TypeError as te:
            weights = np.repeat(weights, len(action))

    reward = 0.5 * np.exp(-np.sum(np.square(action*weights)))

    reward_info = {}
    if gradient:
        raise NotImplementedError("Not implemented")
        # reward_info['gradient'] = action*weights

    if hessian:
        raise NotImplementedError("Not implemented")
        # reward_info['hessian'] = np.diag(weights)

    return reward, reward_info

