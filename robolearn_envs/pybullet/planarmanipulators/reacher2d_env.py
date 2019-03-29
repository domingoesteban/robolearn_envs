import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from gym.spaces import Box

from robolearn_envs.pybullet.planarmanipulators.planarmanipulator3dof \
    import PlanarManipulator3Dof
from robolearn_envs.pybullet.planarmanipulators.planarmanipulator4dof \
    import PlanarManipulator4Dof

from robolearn_envs.pybullet.common import GoalCircle
from robolearn_envs.pybullet.common import Plane
from robolearn_envs.pybullet.common import PusherBorder

from robolearn_envs.utils.transformations import euler_from_quat

from robolearn_envs.reward_functions.state_rewards import distance_cost
from robolearn_envs.reward_functions.action_rewards import quadratic_action_cost

HALF_GOAL_POS_RANGE_X = (0.40, 0.70)
HALF_GOAL_POS_RANGE_Y = (-0.2, 0.70)
FULL_GOAL_POS_RANGE_X = (-0.10, 0.70)
FULL_GOAL_POS_RANGE_Y = (-0.7, 0.70)
MAX_GOAL_POS_NORM = 0.8
MIN_GOAL_POS_NORM = 0.5
ROBOT_CONFIG_RANGE = [
    (-2.18, 2.18),
    (-2.18, 2.18),
    (-2.18, 2.18),
    (-2.18, 2.18),  # Only used for 4dof robot
]


class Reacher2DEnv(BulletEnv):
    def __init__(
            self,
            robot_dof=3,
            is_render=False,
            obs_distances=False,
            only_position=True,  # TODO: Scale yaw cost if only_position=False
            rdn_tgt_pos=True,
            tgt_pose=None,
            robot_config=None,
            rdn_robot_config=True,
            tgt_cost_weight=1.0,
            ctrl_cost_weight=1.0e-2,
            use_log_distances=False,
            log_alpha=1e-6,
            tgt_tolerance=0.05,
            max_time=None,  # seconds
            half_env=False,
            sim_timestep=1/240.,
            frame_skip=1,
            seed=None,
    ):
        """Planar3DoF manipulator seeking to reach a target.
        """
        super(Reacher2DEnv, self).__init__(
            sim_timestep=sim_timestep, frameskip=frame_skip,
            is_render=is_render
        )

        # Set the seed
        self.seed(seed)

        # Plane
        # self.plane = PlaneObject(plane_type='plane_simple')
        self.plane = Plane(plane_type='simple', color=None)
        self.add_to_sim(self.plane)

        # Border
        self._half_env = half_env
        self.border = PusherBorder(color='gray', half_env=self._half_env)
        self.add_to_sim(self.border)

        # Robot
        self._robot_config_is_rdn = rdn_robot_config
        if robot_config is None:
            self._robot_config_mean = \
                [np.deg2rad(-100), np.deg2rad(45), np.deg2rad(20), np.deg2rad(5)][:robot_dof]
        else:
            self._robot_config_mean = robot_config
        self._robot_config_var = [0.05, 0.2, 0.3, 0.3]
        if robot_dof == 3:
            self.robot = PlanarManipulator3Dof(
                self_collision=True,
                init_config=self._robot_config_mean,
                robot_model=None,
                control_mode='joint_torque',
            )
        elif robot_dof == 4:
            self.robot = PlanarManipulator4Dof(
                self_collision=True,
                init_config=self._robot_config_mean,
                robot_model=None,
                control_mode='joint_torque',
            )
        else:
            raise NotImplementedError("Only planar manipulators with 3 and 4"
                                      "DoFs have been implemented!")

        self.add_to_sim(self.robot)
        self._init_robot_config = self.robot.initial_configuration

        # Goal
        self._goal_pos_is_rdn = rdn_tgt_pos
        self._goal_state = np.inf * np.ones(3)
        if tgt_pose is None:
            if self._half_env:
                tgt_range_x = HALF_GOAL_POS_RANGE_X
                tgt_range_y = HALF_GOAL_POS_RANGE_Y
            else:
                tgt_range_x = FULL_GOAL_POS_RANGE_X
                tgt_range_y = FULL_GOAL_POS_RANGE_Y

            while (np.linalg.norm(self._goal_state[:2]) > MAX_GOAL_POS_NORM) or \
                    (np.linalg.norm(self._goal_state[:2]) < MIN_GOAL_POS_NORM):
                self._goal_state[0] = \
                    np.random.uniform(tgt_range_x[0], tgt_range_x[1])
                self._goal_state[1] = \
                    np.random.uniform(tgt_range_y[0], tgt_range_y[1])
        else:
            self._goal_state = tgt_pose
        self.goal = GoalCircle(init_pos=(0, 0, -0.0055), color='green')
        self.add_to_sim(self.goal)

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
        self._use_obs_distances = obs_distances
        if only_position:
            self._pose_dof = 2   # [x, y]
        else:
            self._pose_dof = 3   # [x, y, theta]
        if obs_distances:
            optitrack_dim = 1*self._pose_dof
        else:
            optitrack_dim = 2*self._pose_dof
        obs_dim = robot_state_dim + optitrack_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,),
            dtype=np.float32
        )
        self._observation = OrderedDict()
        self._observation['robot_state'] = np.zeros(robot_state_dim)
        if obs_distances:
            self._observation['goal_ee_diff'] = np.zeros(self._pose_dof)
        else:
            self._observation['goal_pose'] = np.zeros(self._pose_dof)
            self._observation['ee_pose'] = np.zeros(self._pose_dof)

        # STATE
        state_dim = robot_state_dim + 3*2  # robot_state + (XYyaw)*(GOAL, EE)
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
            goal_pose=np.zeros(3),
            ee_pose=np.zeros(3),
        )

        # Environment settings
        self.max_time = max_time
        if self._half_env:
            self.set_visualizer_data(distance=1.1, yaw=0.0, pitch=-89.95,
                                     target_pos=[0.35, 0.0, 0.0])
        else:
            self.set_visualizer_data(distance=1.1, yaw=-30, pitch=-90,
                                     target_pos=None)
        self.set_rendering(self._is_render)

        # Initial conditions
        self._initial_conditions = []
        self._initial_obs_conditions = []
        self._current_init_cond = np.zeros(self.state_dim)

        # Costs
        self._tgt_cost_weight = tgt_cost_weight
        self._tgt_tolerance = tgt_tolerance
        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_log_distances = use_log_distances
        self._log_alpha = log_alpha
        # Max reward
        self._max_reward = np.zeros(1)
        self._is_env_instantiation_complete = True

    def _reset_env(self, *args, **kwargs):
        """
        Reset the robot and the objects in the environment.
        Function called by self.reset()
        :param condition:
        :return:
        """
        # Reset Simulation
        self.reset_simulation()

        # Reset ground in simulation
        self.plane.reset()

        # Border
        self.border.reset()

        # Robot
        condition = kwargs.pop('condition', None)
        robot_state = self.robot.reset()
        if self._is_env_instantiation_complete:
            if condition is None:
                if self._robot_config_is_rdn:
                    wrong_config = True
                    while wrong_config:
                        init_robot_config = np.clip(
                            [
                                np.random.uniform(ROBOT_CONFIG_RANGE[0][0],
                                                  ROBOT_CONFIG_RANGE[0][1]),
                                np.random.uniform(ROBOT_CONFIG_RANGE[1][0],
                                                  ROBOT_CONFIG_RANGE[1][1]),
                                np.random.uniform(ROBOT_CONFIG_RANGE[2][0],
                                                  ROBOT_CONFIG_RANGE[2][1]),
                                np.random.uniform(ROBOT_CONFIG_RANGE[3][0],
                                                  ROBOT_CONFIG_RANGE[3][1]),
                            ][:self.robot.dof],
                            self.min_robot_config,
                            self.max_robot_config
                        )
                        self.robot.set_state(np.concatenate((
                            init_robot_config,
                            np.zeros_like(init_robot_config)))
                        )
                        x_lim = -0.10 if self._half_env else -0.5
                        wrong_config = self.get_ee_pose()[0] <= x_lim
                else:
                    init_robot_config = self.init_robot_config

            else:
                if isinstance(condition, int):
                    init_robot_config = self._init_robot_configs[condition]
                else:
                    raise NotImplementedError
                    # init_robot_config = np.clip(condition,
                    #                             self.min_robot_config,
                    #                             self.max_robot_config)
            self.robot.initial_configuration = init_robot_config

        # Goal
        if condition is None:
            if self._goal_pos_is_rdn:
                self._goal_state = np.array([np.inf, np.inf, 0])
                if self._half_env:
                    tgt_range_x = HALF_GOAL_POS_RANGE_X
                    tgt_range_y = HALF_GOAL_POS_RANGE_Y
                else:
                    tgt_range_x = FULL_GOAL_POS_RANGE_X
                    tgt_range_y = FULL_GOAL_POS_RANGE_Y

                while (np.linalg.norm(self._goal_state) > MAX_GOAL_POS_NORM) or \
                        (np.linalg.norm(self._goal_state) < MIN_GOAL_POS_NORM):
                    self._goal_state[0] = \
                        np.random.uniform(tgt_range_x[0], tgt_range_x[1])
                    self._goal_state[1] = \
                        np.random.uniform(tgt_range_y[0], tgt_range_y[1])
        else:
            self._goal_state = self._init_tgt_states[condition]

        self.goal.reset(pose=(self._goal_state[0], self._goal_state[1], -0.0055))

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

            self._state['goal_pose'][:] = self.get_goal_pose()
            self._state['ee_pose'][:] = self.get_ee_pose()
            # print(self._state['robot_state'][self.robot.dof:])

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
            state (float[Sdim]): Desired state of the environment.

        Returns:

        """
        if not self._is_env_instantiation_complete:
            raise ValueError("Not possible to set state!")
        if len(state) != self.state_dim:
            raise ValueError("Desired state size does not correpond to "
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
            # Robot state
            self._observation['robot_state'][:] = \
                self.robot.get_state(only_controlled_joints=True)

            goal_state = self.get_goal_pose()[:self._pose_dof]
            ee_state = self.get_ee_pose()[:self._pose_dof]

            if self._use_obs_distances:
                self._observation['goal_ee_diff'][:] = goal_state - ee_state
            else:
                self._observation['goal_pose'][:] = goal_state
                self._observation['ee_pose'][:] = ee_state

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
            ee_pos = self.get_ee_pose()[:2]
            tgt_pos = self.get_goal_pose()[:2]
            ee_tgt_dist = np.linalg.norm(ee_pos - tgt_pos)

            done = ee_tgt_dist <= self._tgt_tolerance
            done_trigger = 'success'
        else:
            if self.sim_time >= self.max_time:
                done = True
                done_trigger = 'time'
            else:
                done = False
                done_trigger = False

        info = {
            'done_trigger': done_trigger,
        }

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        ee_pos = self.get_ee_pose()[:self._pose_dof]
        target_pos = self._goal_state[:self._pose_dof]

        # Distance to object
        cost, _ = distance_cost(ee_pos - target_pos, log_weight=1.,
                                alpha=1e-2)
        ee_tgt_cost = self._tgt_cost_weight * cost

        # Control
        # ctrl_cost = np.square(action).sum()
        lb = self.low_action_bounds
        ub = self.high_action_bounds
        scaling = (ub - lb) * 0.5
        cost, _ = quadratic_action_cost(action, weights=1. / scaling)
        ctrl_cost = self._ctrl_cost_weight * cost

        reward_composition = np.array([
            -ee_tgt_cost,
            -ctrl_cost,
            ])

        reward = reward_composition.sum().item()

        info = {
            'reward_composition': reward_composition
        }

        return reward, info

    def _calc_max_reward(self):
        max_reward = self._compute_reward(np.zeros(self.state_dim),
                                          np.zeros(self.action_dim)
                                          )
        return max_reward

    def add_initial_condition(self, state):
        self.set_rendering(False)
        self.reset(-1)
        self.set_rendering(self._is_render)
        full_state = self.get_state()
        full_obs = self.get_observation()

        self._initial_conditions.append(full_state)
        self._initial_obs_conditions.append(full_obs)

    def clear_initial_condition(self, idx=None):
        if idx is None:
            self._initial_conditions.clear()
        else:
            self._initial_conditions.pop(idx)

    def get_conditions(self, cond=None):
        if cond is None:
            return list(self._initial_conditions)
        else:
            return self._initial_conditions[cond]

    def get_total_joints(self):
        return self.robot.total_joints

    def get_ee_pose(self):
        """ Returns pose of the end-effector, defined by (X, Y, yaw) """
        gripper_pose = self.robot._links['gripper_center'].get_pose()
        return self.convert_2d_pose(gripper_pose)

    def get_goal_pose(self):
        """ Returns pose of the goal, defined by (X, Y, yaw) """
        goal_pose = self.goal.get_pose()
        return self.convert_2d_pose(goal_pose)

    @staticmethod
    def convert_2d_pose(pose):
        """ It converts a pose defined by (x,y,z,ox,oy,oz,ow) to (x,z,theta)"""
        xy = pose[:2]
        ori = euler_from_quat(pose[3:])[2]
        # ori = normalize_angle(ori, range='pi')
        return np.array([xy[0], xy[1], ori])

    def set_robot_config(self, joint_config):
        self.robot.set_state(joint_config, np.zeros_like(joint_config))

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @property
    def initial_obs_conditions(self):
        return self._initial_obs_conditions

    @property
    def min_robot_config(self):
        limits_robot_config = self.robot.get_joint_limits()
        return [limit[0] for limit in limits_robot_config]

    @property
    def max_robot_config(self):
        limits_robot_config = self.robot.get_joint_limits()
        return [limit[1] for limit in limits_robot_config]

    @property
    def init_robot_config(self):
        return self.robot.initial_configuration


if __name__ == "__main__":
    render = True
    # render = False

    H = 10000

    env = Reacher2DEnv(robot_dof=4,
                       is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()
    input('ww')

    for tt in range(H):
        print('env_step: %02d' % tt)
        # next_obs, rew, done, info = env.step(env.action_space.sample())
        next_obs, rew, done, info = env.step(np.zeros_like(env.action_space.sample()))

        if done:
            print("The environment is done!")
            break

        if render:
            env.render()

    print("Correct!")
