import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from gym.spaces import Box

from robolearn_envs.pybullet.common import Plane
from robolearn_envs.pybullet.common import GoalArrow
from robolearn_envs.pybullet.common import GoalCircle
from robolearn_envs.pybullet.common import GoalSphere

from robolearn_envs.reward_functions.state_rewards \
    import exponential_distance_reward, exponential_distance_cost
from robolearn_envs.reward_functions.action_rewards import quadratic_action_cost

from robolearn_envs.utils.transformations import compute_cartesian_error


class RobotReachingEnv(BulletEnv):
    def __init__(self,
                 robot,
                 links_names,
                 goal_poses,
                 is_render=False,
                 sim_timestep=1/240.,
                 frame_skip=1,
                 seed=None,
                 max_time=None,
                 only_position=None,
                 goal_tolerance=0.05,
                 goal_marker='arrow',
                 goal_reward_weights=1.e0,
                 ctrl_reward_weight=1.0e-2,
                 base_height_lims=(0.0, 5.0),
                 ):
        super(RobotReachingEnv, self).__init__(
            sim_timestep=sim_timestep, frameskip=frame_skip,
            is_render=is_render, seed=seed,
        )

        # Plane
        # self._plane = Plane(plane_type='stone',
        self.plane = Plane(plane_type='plane_simple', color=None)
        self.add_to_sim(self.plane)

        # Robot
        self.robot = robot
        self.add_to_sim(self.robot)
        self._init_robot_config = self.robot.initial_configuration
        self._link_names = links_names

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
        if only_position:
            self._pose_dim = 3  # X, Y, Z
        else:
            self._pose_dim = 7  # X, Y, Z, qX, qY, qZ, qW
        obs_dim = robot_state_dim + self._pose_dim*len(links_names)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,),
            dtype=np.float32
        )
        self._observation = OrderedDict()
        self._observation['robot_state'] = np.zeros(robot_state_dim)
        for body_name in links_names:
            self._observation[str(body_name)+'_pose'] = np.zeros(self._pose_dim)

        # STATE
        state_dim = robot_state_dim + 7*len(links_names)
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
        )
        for body_name in links_names:
            self._state[str(body_name)+'_pose'] = np.zeros(7)

        # Environment settings
        self.max_time = max_time
        self.set_rendering(self._is_render)

        # Initial conditions
        self._initial_conditions = []
        self._current_init_cond = np.zeros(self.state_dim)

        # Reaching task
        self.goal_poses = [np.array(pose) for pose in goal_poses]
        self._goal_tolerances = goal_tolerance*np.ones(len(links_names))
        goal_colors = ['green', 'orange', 'blue', 'red', 'pink', 'black', 'white']
        if goal_marker.lower() == 'arrow':
            marker_type = GoalArrow
        elif goal_marker.lower() == 'circle':
            marker_type = GoalCircle
        else:
            marker_type = GoalSphere
        self._goals = []
        for gg in range(len(links_names)):
            marker = marker_type(
                init_pos=self.goal_poses[gg],
                tolerance=self._goal_tolerances[gg],
                color=goal_colors[gg],
            )
            self._goals.append(marker)
            self.add_to_sim(marker)

        self._base_height_lims = base_height_lims

        # Rewards
        self._goal_reward_weights = goal_reward_weights*np.ones(len(links_names))
        self._ctrl_reward_weight = ctrl_reward_weight
        self._prev_body_pose_errors = [None for _ in range(len(links_names))]
        # Max reward
        self._max_reward = np.zeros(1)
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

        condition = kwargs.pop('condition', None)

        # Reset robot in simulation
        robot_state = self.robot.reset()

        # Reset goal(s) in simulation
        if self._is_env_instantiation_complete:
            for gg, goal_pose in enumerate(self.goal_poses):
                self._goals[gg].reset(pose=goal_pose)

        if self._is_env_instantiation_complete:
            self._prev_body_pose_errors = \
                [None for _ in range(len(self._link_names))]

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
            # Robot state
            self._state['robot_state'][:] = \
                self.robot.get_state(only_controlled_joints=True)

            # Link poses
            body_poses = self.robot.get_link_poses(self._link_names)
            for body_name, body_pose in zip(self._link_names, body_poses):
                self._state[str(body_name)+'_pose'] = body_pose

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
            # TODO: Set the goal state

            init_idx += dim

        self._update_env_state()
        self._update_env_obs()

    # ################### #
    # OBSERVATION related #
    # ################### #

    def _update_env_obs(self):
        if self._is_env_instantiation_complete:
            # Robot state
            self._observation['robot_state'][:] = \
                self.robot.get_state(only_controlled_joints=True)

            # Link poses
            body_poses = self.robot.get_link_poses(self._link_names)
            for body_name, body_pose in zip(self._link_names, body_poses):
                self._observation[str(body_name)+'_pose'][:] = \
                    body_pose[:self._pose_dim]

            # Compute pose error
            for pp in range(len(self._link_names)):
                prev_tgt_tray_ori_error = \
                    None if self._prev_body_pose_errors[pp] is None \
                    else self._prev_body_pose_errors[pp][3:]
                self._prev_body_pose_errors[pp] = compute_cartesian_error(
                    self.goal_poses[pp],
                    body_poses[pp],
                    prev_ori_diff=prev_tgt_tray_ori_error
                )

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

        info = {}

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        state = self._state
        # Link poses
        body_poses = []
        for body_name in self._link_names:
            body_poses.append(state[str(body_name)+'_pose'])

        # Body reaching reward
        body_reaching_reward = np.zeros(len(self._link_names))
        for bb in range(len(body_poses)):
            # Compute pose difference
            pose_diff = compute_cartesian_error(
                self.goal_poses[bb],
                body_poses[bb],
                prev_ori_diff=self._prev_body_pose_errors[bb][3:]
            )

            # Distance to goal reward cost (position)
            pos_weights = np.array([1.0, 1.0, 1.0])
            pos_norm = np.linalg.norm(pos_weights * pose_diff[:3])
            pos_reward = exponential_distance_reward(
                pos_norm, temperature=0.4)[0]

            # Distance to goal reward (orientation)
            ori_weights = np.array([1.0, 1.0, 1.0])
            ori_norm = np.linalg.norm(ori_weights * pose_diff[3:])
            ori_reward = exponential_distance_reward(
                ori_norm, temperature=0.8)[0]

            # Distance to goal reward
            body_reaching_reward[bb] = self._goal_reward_weights[bb] * \
                (pos_reward + ori_reward * 0.01)

        # Control action reward
        action_cost = quadratic_action_cost(action=action)[0]
        action_reward = -self._ctrl_reward_weight * action_cost
        action_reward = action_reward[np.newaxis]

        reward_composition = np.concatenate((
            body_reaching_reward,
            action_reward
        ))
        reward = reward_composition.sum().item()

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
    def robot_n_ordered_joints(self):
        return self.robot.n_ordered_joints

    @property
    def robot_base_pose(self):
        return self.robot.get_base_pose()

    def get_robot_body_poses(self):
        return self.robot.get_link_poses(self._link_names)

    def get_robot_state(self, only_ordered_joints=True):
        return self.robot.get_state(only_ordered_joints)

    def get_goal_poses(self):
        return self.goal_poses.copy()

    @property
    def pose_dim(self):
        return self._pose_dim
