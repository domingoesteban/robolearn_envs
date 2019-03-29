import numpy as np
from robolearn_envs.reward_functions.utils import convert_reward_cost_fcn
from robolearn_envs.utils.transformations import compute_cartesian_error


def pose_distance_cost(desired, current, quad_weight=1., log_weight=0., alpha=1.e-5,
                       prev_dist_error=None, gradient=False, hessian=False):
    """Compute distance cost between 2 poses.
    Calculates cost according to:
        math:`cost = quad_weight*d^2 + log_weight*log(d^2 + alpha)`
    where math:`d = ||pose_error||_2`
    Args:
        desired (list or np.ndarray): desired pose (x, y, z, qx, qy, qz, qw)
        current (list or np.ndarray): current pose (x, y, z, qx, qy, qz, qw)
        quad_weight (float): Quadratic term weight.
        log_weight (float): Log (Lorentzian rho-function) term weight.
        alpha (float):
        prev_dist_error:
        gradient (bool): Return cost gradient.
        hessian (bool): Return cost Hessian.

    Returns:

    """
    if prev_dist_error is None:
        prev_ori_error = None
    else:
        prev_ori_error = prev_dist_error[3:]

    # Calculate error
    d = compute_cartesian_error(desired, current, prev_ori_diff=prev_ori_error)
    return distance_cost(d, quad_weight=quad_weight, log_weight=log_weight,
                         alpha=alpha, gradient=gradient, hessian=hessian)


def distance_cost(distances, quad_weight=1., log_weight=0., alpha=1e-5,
                  gradient=False, hessian=False):
    """ Compute distance cost.
    Calculates cost according to:
        math:`cost = quad_weight*d^2 + log_weight*log(d^2 + alpha)`
    where math:`d = ||distances||_2`

    Args:
        distances:
        quad_weight (float): Quadratic term weight.
        log_weight (float): Log (Lorentzian rho-function) term weight.
        alpha (float):
        gradient (bool): Return cost gradient.
        hessian (bool): Return cost Hessian.

    Returns:

    """
    cost = 0

    distances = np.atleast_2d(distances)
    d = np.linalg.norm(distances, axis=-1)

    # Quadratic penalty
    cost += quad_weight * d ** 2

    # Log penalty (Lorentzian rho-function)
    cost += log_weight * np.log(d ** 2 + alpha)

    cost_info = {}
    if gradient:
        raise NotImplementedError("Not implemented")

    if hessian:
        raise NotImplementedError("Not implemented")

    return cost, cost_info


def exponential_distance_reward(distances, temperature=1,
                                gradient=False, hessian=False):
    """
    Calculates reward according to:
    reward = exp(-(d**2))
    """

    distances = np.atleast_2d(distances)
    d = np.linalg.norm(distances, axis=-1)

    # cost = np.exp(-(d**2)/temperature)
    reward = np.exp(-d/temperature)

    reward_info = {}
    if gradient:
        raise NotImplementedError("Not implemented")

    if hessian:
        raise NotImplementedError("Not implemented")

    return reward, reward_info


def exponential_distance_cost(*args, **kwargs):
    return convert_reward_cost_fcn(exponential_distance_reward, *args, **kwargs)


def distance_reward(*args, **kwargs):
    return convert_reward_cost_fcn(distance_cost, *args, **kwargs)


def state_limits_cost(states, states_min, states_max,
                      gradient=False, hessian=False):

    states = np.atleast_2d(states)
    lim_diff = states - 0.5*(states_min[None] + states_max[None])

    cost = np.sum(lim_diff, axis=-1)

    cost_info = {}
    if gradient:
        cost_info['gradient'] = 2 * (states - lim_diff)

    if hessian:
        cost_info['hessian'] = np.tile(2*np.eye(states.shape[-1][None]),
                                       (states.shape[:-1], 1, 1))

    return cost, cost_info
