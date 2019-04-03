import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from robolearn_envs.utils.transformations import euler_from_quat
from robolearn_envs.utils.transformations import quat_difference
from gym.spaces import Box

from robolearn_envs.pybullet.common import Plane

from robolearn_envs.reward_functions.action_rewards import quadratic_action_cost
from robolearn_envs.reward_functions.state_rewards import state_limits_cost


class RobotLocomotionEnv(BulletEnv):
    def __init__(self,
                 robot,
                 feet_names,
                 is_render=False,
                 sim_timestep=1/240.,
                 frame_skip=1,
                 seed=None,
                 max_time=None,
                 base_height_lims=(0.30, 2.0),
                 vel_rew_weight=5e-1,
                 alive_bonus=1e-4,
                 ctrl_rew_weight=1e-4,
                 vel_deviation_rew_weight=5e-3,
                 rot_deviation_rew_weight=1e-2,
                 height_deviation_rew_weight=1e-1,
                 joint_limits_rew_weight=1e-3,
                 impact_rew_weight=1e-3,
                 ):

        super(RobotLocomotionEnv, self).__init__(
            sim_timestep=sim_timestep, frameskip=frame_skip,
            is_render=is_render, seed=seed,
        )

        # Plane
        # self._plane = PlaneObject(plane_type='stone', color=None)
        self.plane = Plane(plane_type='plane_simple', color=None)
        self.add_to_sim(self.plane)

        # Robot
        self.robot = robot
        self.add_to_sim(self.robot)
        self._init_robot_config = self.robot.initial_configuration
        self._feet_names = feet_names

        # Reset environment so we can get info from pybullet
        self.set_rendering(False)
        self._is_env_instantiation_complete = False
        self.reset()
        self._is_render = is_render

        # ACTUATION
        self.action_space = Box(
            self.robot.low_action_bounds,
            self.robot.high_action_bounds,
            dtype=np.float32
        )

        # OBSERVATION
        robot_state_dim = self.robot.observation_dim  # joint pos/vel
        base_link_dim = 5  # Z, qX, qY, qZ, qW
        n_contacts = len(feet_names)
        obs_dim = base_link_dim + robot_state_dim + n_contacts
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,),
            dtype=np.float32
        )
        self._observation = OrderedDict(
            base_state=np.zeros(base_link_dim),
            robot_state=np.zeros(robot_state_dim),
            contact_activation=np.zeros(n_contacts),
        )

        # STATE
        state_dim = base_link_dim + robot_state_dim + n_contacts
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            base_state=np.zeros(base_link_dim),
            robot_state=np.zeros(robot_state_dim),
            contact_activation=np.zeros(n_contacts),
        )

        # Environment settings
        self.max_time = max_time
        self.set_rendering(self._is_render)

        # Initial conditions
        self._initial_conditions = []
        self._current_init_cond = np.zeros(self.state_dim)

        # Locomotion task
        self._base_height_lims = base_height_lims
        self._feet_contacts = np.zeros(len(feet_names))

        # Reward weights
        self.vel_rew_weight = vel_rew_weight
        self.alive_bonus = alive_bonus
        self.ctrl_rew_weight = ctrl_rew_weight
        self.vel_deviation_rew_weight = vel_deviation_rew_weight
        self.rot_deviation_rew_weight = rot_deviation_rew_weight
        self.height_rew_weight = height_deviation_rew_weight
        self.joint_limits_rew_weight = joint_limits_rew_weight
        self.impact_rew_weight = impact_rew_weight
        # Max reward
        self._max_rewards = np.zeros(1)
        self._is_env_instantiation_complete = True

    def _reset_env(self, *args, **kwargs):
        """Reset the robot and the objects in the environment.

        Function called by self.reset()

        Args:
            *args:
            **kwargs:

        Returns:
            np.ndarray: Current environment observation

        """
        # Reset Simulation
        self.reset_simulation()

        # Reset ground in simulation
        self.plane.reset()
        self.plane.change_dynamics(
            linkIndex=-1,
            lateralFriction=0.99,
            spinningFriction=0,
            rollingFriction=0,
        )
        # Reset robot in simulation
        robot_state = self.robot.reset()

        # Change feet dynamics
        for foot_id in self.robot.get_link_ids(self._feet_names):
            self.robot.change_dynamics(
                linkIndex=foot_id,
                lateralFriction=0.99, spinningFriction=0, rollingFriction=0.0
            )

        # Update Environment State and Observation
        state = self._update_env_state()
        observation = self._update_env_obs()

        # Replace init_cond with current state
        if self._is_env_instantiation_complete:
            self._current_init_cond = np.copy(self._state)

        # Update max reward
        if self._is_env_instantiation_complete:
            self._max_reward = np.array([0.])
            # self._max_reward = self._calc_max_reward()

        # Visualization (if applicable)
        self.enable_vis()

        # Get initial base height
        self._init_base_pose = self.robot.get_base_pose()

        return observation

    # ############## #
    # ACTION related #
    # ############## #

    def _set_action(self, action):
        self.robot.apply_action(action)

        # Simulation step
        self.sim_step()

    # ############# #
    # STATE related #
    # ############# #
    def _update_env_state(self):
        if self._is_env_instantiation_complete:
            # Base state
            self._state['base_state'][:] = \
                self.robot.get_base_pose()[2:]

            # Robot state
            self._state['robot_state'][:] = self.robot.get_state(True)

            # Feet contacts
            self._state['contact_activation'][:] = \
                self.get_feet_contact_forces()

        return self.get_state()

    def get_state(self):
        if self._is_env_instantiation_complete:
            return np.concatenate(
                [state.flatten() for state in self._state.values()]
            )

    def set_state(self, state):
        """
        Set the state of the environment.
        Args:
            state (list or np.ndarray): Desired state of the environment.

        Returns:

        """
        state = np.array(state)
        if not self._is_env_instantiation_complete:
            raise ValueError("Not possible to set state!")
        if len(state) != self.state_dim:
            raise ValueError("Desired state size does not correspond to "
                             "the environment state. (%d != %d)" %
                             (len(state), self.state_dim))
        init_idx = 0
        for key, val in self._state.items():
            dim = np.prod(val.shape)
            self._state[key] = state[init_idx:init_idx+dim]
            # Set robot state
            if key == 'robot_state':
                self.robot.set_state(self._state[key])
            # elif key.endswith('_pose'):

            init_idx += dim

        self._update_env_state()
        self._update_env_obs()

    # ################### #
    # OBSERVATION related #
    # ################### #

    def _update_env_obs(self):
        if self._is_env_instantiation_complete:
            # Base state
            self._observation['base_state'][:] = self.robot.get_base_pose()[2:]

            # Robot state
            self._observation['robot_state'][:] = \
                self.robot.get_state(only_controlled_joints=True)

            # Feet contacts
            self._observation['contact_activation'][:] = \
                self.get_feet_contact_forces()

        return self.get_observation()

    def get_observation(self):
        if self._is_env_instantiation_complete:
            return np.concatenate(
                [obs.flatten() for obs in self._observation.values()]
            )

    def set_observation(self, observation):
        if not self._is_env_instantiation_complete:
            raise ValueError("Not possible to set observation!")
        if len(observation) != self.obs_dim:
            raise ValueError("Desired observation size does not correpond to "
                             "the environment observation dim. (%d != %d)" %
                             (len(observation), self.obs_dim))
        init_idx = 0
        for key, val in self._observation:
            dim = np.prod(self._observation[val].shape)
            self._observation[val] = observation[init_idx:init_idx+dim]
            init_idx += dim

    # ############ #
    # STEP related #
    # ############ #

    def _check_termination(self):
        if self.max_time is None:
            end_time = False
        else:
            end_time = self.env_time >= self.max_time

        base_height = self.robot.get_base_pose()[2]
        base_min, base_max = self._base_height_lims
        height_limit = base_height < base_min or base_height > base_max

        any_infinite = not np.isfinite(self.get_observation()).all()

        if end_time or height_limit or any_infinite:
            done = True
        else:
            done = False

        if end_time:
            termination_trigger = 'time'
        elif height_limit:
            termination_trigger = 'fall'
        elif any_infinite:
            termination_trigger = 'obs_error'
        else:
            termination_trigger = False

        info = {
            'termination_trigger': termination_trigger
        }

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        base_pose = self.robot.get_base_pose()
        base_vel = self.robot.get_base_velocity()

        # X-direction velocity reward
        des_x_vel = 1.0
        # lin_vel_reward = self.vel_rew_weight * com_vel[0]
        lin_vel_reward = self.vel_rew_weight * np.exp(
            -np.sum(np.square(des_x_vel - base_vel[0]))/3.e-1
        )

        # Alive bonus
        alive_bonus = self.alive_bonus

        # Action cost
        lb = self.low_action_bounds
        ub = self.high_action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_rew = .5 * self.ctrl_rew_weight * np.exp(
            -np.sum(np.square(action / scaling))/1.e2
        )
        # ctrl_rew = .5 * self.ctrl_cost_weight * np.exp(
        #     -np.sum(np.square(action))/1.e-5
        # )

        # Y and Z direction cost
        vel_deviation_rew = self.vel_deviation_rew_weight * np.exp(
            -np.sum(np.square(base_vel[1:3]))/1.e0
        )
        # print(euler_from_quat(base_pose[3:]))
        # Orientation difference
        rot_deviation_rew = self.rot_deviation_rew_weight * np.exp(
            -np.sum(np.square(
                quat_difference(base_pose[3:], self._init_base_pose[3:])
            ))/3.e-2
        )

        # Initial weight
        height_deviation_rew = self.height_rew_weight * np.exp(
            -np.square(self._init_base_pose[2] - base_pose[2])/1.e-2
        )

        # Joint limits
        # n_joints_at_limit = np.count_nonzero(
        #     np.abs(self.robot.get_controlled_joints_relative_positions()) > 0.99)
        # n_joints_at_limit = 0
        joint_rel_pos = self.robot.get_controlled_joints_relative_positions()
        joint_limit_rew = self.joint_limits_rew_weight * np.exp(
            -np.sum(np.square(joint_rel_pos))/1.e0
        )

        # Contact cost
        feet_contact_forces = self.get_feet_contact_forces()
        contact_reward = self.impact_rew_weight * np.exp(np.sum(
            -np.square(feet_contact_forces)) / 1.e0
        )

        reward_composition = np.array([
            lin_vel_reward,
            alive_bonus,
            ctrl_rew,
            vel_deviation_rew,
            rot_deviation_rew,
            height_deviation_rew,
            joint_limit_rew,
            contact_reward,
        ])
        reward = reward_composition.sum().item()
        np.set_printoptions(precision=4, suppress=True)

        info = {
            'reward_composition': reward_composition
        }

        return reward, info

    # ######################### #
    # Useful methods/attributes #
    # ######################### #
    @property
    def robot_initial_configuration(self):
        return self.robot.initial_configuration

    @property
    def robot_base_pose(self):
        return self.robot.get_base_pose()

    @property
    def robot_n_ordered_joints(self):
        return self.robot.n_ordered_joints

    def get_feet_contact_forces(self, force_lims=(-np.inf, np.inf)):
        foot_ids = self.robot.get_link_ids(self._feet_names)
        collisions = np.zeros(len(foot_ids))
        for ff, foot_id in enumerate(foot_ids):
            foot_id = None
            collision = self.get_external_forces(self.robot, self.plane, foot_id)
            collisions[ff] = collision
        collisions = np.clip(collisions, force_lims[0], force_lims[1])
        return collisions
