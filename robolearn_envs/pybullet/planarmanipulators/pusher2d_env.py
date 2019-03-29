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
from robolearn_envs.pybullet.common import Cylinder2d

from robolearn_envs.utils.transformations import euler_from_quat
from robolearn_envs.utils.transformations import create_quat

from robolearn_envs.reward_functions.state_rewards \
    import exponential_distance_reward

# GOAL_RANGE_X = (0.55, 0.7)
# GOAL_RANGE_Y = (0.55, 0.7)
# HALF_TGT_RANGE_X = (0.40, 0.70)
# HALF_TGT_RANGE_Y = (-0.2, 0.70)
GOAL_RANGE_X = (0.63, 0.68)  # Paper
GOAL_RANGE_Y = (0.33, 0.38)  # Paper
HALF_TGT_RANGE_X = (0.50, 0.55)  # Paper
HALF_TGT_RANGE_Y = (0.1, 0.20)  # Paper
FULL_TGT_RANGE_X = (-0.10, 0.70)
FULL_TGT_RANGE_Y = (-0.7, 0.70)


GOAL_RANGE_X = (0.53, 0.78)  # Prueba 04/03/19
GOAL_RANGE_Y = (0.23, 0.48)  # Prueba 04/03/19
TGT_RANGE_X = (0.40, 0.65)  # Prueba 04/03/19
TGT_RANGE_Y = (0.0, 0.30)  # Prueba 04/03/19


MAX_GOAL_NORM = 0.8
MAX_CYLINDER_POS_NORM = 0.65
MIN_CYLINDER_POS_NORM = 0.45
ROBOT_CONFIG_RANGE = [
    (-1.35, 1.35),
    (-0.90, 0.9),
    (-1.5, 1.5),
    (-1.5, 1.5),  # Only used for 4dof robot
]


class Pusher2DBulletEnv(BulletEnv):
    def __init__(
            self,
            robot_dof=3,
            is_render=False,
            obs_distances=False,
            only_position=True,  # TODO: Scale yaw cost if only_position=False
            rdn_goal_pose=True,
            goal_pose=None,
            tgt_pose=None,
            rdn_tgt_object_pose=True,
            robot_config=None,
            rdn_robot_config=True,
            tgt_cost_weight=1.0,
            goal_cost_weight=3.0,
            ctrl_cost_weight=1.0e-2,
            goal_tolerance=0.05,
            max_time=None,  # seconds
            half_env=True,
            sim_timestep=1/240.,
            frame_skip=1,
            seed=None,
    ):
        """Planar3DoF robot seeking to push a cylinder to a target position.
        """
        super(Pusher2DBulletEnv, self).__init__(
            sim_timestep=sim_timestep, frameskip=frame_skip,
            is_render=is_render, seed=seed,
        )

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
                self_collision=False,
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
        self._goal_pos_is_rdn = rdn_goal_pose
        self._goal_state = np.inf * np.ones(3)
        if tgt_pose is None:
            if self._half_env:
                tgt_range_x = HALF_TGT_RANGE_X
                tgt_range_y = HALF_TGT_RANGE_Y
            else:
                tgt_range_x = FULL_TGT_RANGE_X
                tgt_range_y = FULL_TGT_RANGE_Y

            while (np.linalg.norm(self._goal_state[:2]) > MAX_CYLINDER_POS_NORM) or \
                    (np.linalg.norm(self._goal_state[:2]) < MIN_CYLINDER_POS_NORM):
                self._goal_state[0] = \
                    np.random.uniform(tgt_range_x[0], tgt_range_x[1])
                self._goal_state[1] = \
                    np.random.uniform(tgt_range_y[0], tgt_range_y[1])
        else:
            self._goal_state = tgt_pose
        self.goal = GoalCircle(init_pos=(0, 0, -0.0055), color='green')
        self.add_to_sim(self.goal)

        # Cylinder Object
        self.obj_pose_is_rdn = rdn_tgt_object_pose
        if tgt_pose is None:
            self._obj_state_mean = [0.6, 0.25, 1.4660]
        else:
            self._obj_state_mean = tgt_pose
        # self.goal_state_mean = [[0.8+0*0.1+0*0.105,  0*0.105,   0.],
        #                        ]
        self._obj_state_var = [0.05 * 1, 0.2 * .1, 0.2]
        self._obj_height = 0.055

        self.cylinder = Cylinder2d(init_pos=[self._obj_state_mean[0],
                                             self._obj_state_mean[1],
                                             self._obj_height],
                                   color='yellow')
        self.add_to_sim(self.cylinder)

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
            optitrack_dim = 2*self._pose_dof
        else:
            optitrack_dim = 3*self._pose_dof
        obs_dim = robot_state_dim + optitrack_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,),
            dtype=np.float32
        )
        self._observation = OrderedDict()
        self._observation['robot_state'] = np.zeros(robot_state_dim)
        if obs_distances:
            self._observation['obj_ee_diff'] = np.zeros(self._pose_dof)
            self._observation['goal_obj_diff'] = np.zeros(self._pose_dof)
        else:
            self._observation['cylinder_pose'] = np.zeros(self._pose_dof)
            self._observation['goal_pose'] = np.zeros(self._pose_dof)
            self._observation['ee_pose'] = np.zeros(self._pose_dof)

        # STATE
        state_dim = robot_state_dim + 3*3  # robot_state + (XYyaw)*(EE, OBJ, GOAL)
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
            object_pose=np.zeros(3),
            goal_pose=np.zeros(3),
            ee_pose=np.zeros(3),
        )

        # Environment settings
        self.max_time = max_time
        if self._half_env:
            # self.set_visualizer_data(distance=1.1, yaw=-30, pitch=-90,
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
        self._goal_cost_weight = goal_cost_weight
        self._goal_tolerance = goal_tolerance
        self._ctrl_cost_weight = ctrl_cost_weight
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
        if self._is_env_instantiation_complete:
            if condition is None:
                if self._goal_pos_is_rdn:
                    self._goal_state = np.array([np.inf, np.inf, 0])
                    if self._half_env:
                        tgt_range_x = HALF_TGT_RANGE_X
                        tgt_range_y = HALF_TGT_RANGE_Y
                    else:
                        tgt_range_x = FULL_TGT_RANGE_X
                        tgt_range_y = FULL_TGT_RANGE_Y

                    while (np.linalg.norm(self._goal_state) > MAX_CYLINDER_POS_NORM) or \
                            (np.linalg.norm(self._goal_state) < MIN_CYLINDER_POS_NORM):
                        self._goal_state[0] = \
                            np.random.uniform(tgt_range_x[0], tgt_range_x[1])
                        self._goal_state[1] = \
                            np.random.uniform(tgt_range_y[0], tgt_range_y[1])
            else:
                self._goal_state = self._init_tgt_states[condition]

        self.goal.reset(pose=(self._goal_state[0], self._goal_state[1], -0.0055))

        # Target Object(s)
        self.cylinder.reset()
        if self._is_env_instantiation_complete:
            xy_offset = np.array([np.inf, np.inf])
            if self.obj_pose_is_rdn:
                wrong_obj_pose = True
                while (np.linalg.norm(self._obj_state_mean[:2] + xy_offset) >
                       MAX_CYLINDER_POS_NORM) or \
                      (np.linalg.norm(self._obj_state_mean[:2] + xy_offset) <
                       MIN_CYLINDER_POS_NORM) or wrong_obj_pose:
                    xy_offset = \
                        self.np_random.randn(2)*np.sqrt(self._obj_state_var[:2])

                    # Set XY position
                    des_pos = np.zeros(3)
                    des_pos[:2] = self._obj_state_mean[:2] + xy_offset
                    des_pos[0] = np.clip(des_pos[0], TGT_RANGE_X[0], TGT_RANGE_X[1])
                    des_pos[1] = np.clip(des_pos[1], TGT_RANGE_Y[0], TGT_RANGE_Y[1])
                    # Set the height (Z)
                    des_pos[2] = self._obj_height
                    # Set the orientation (Yaw)
                    yaw_offset = 0
                    des_yaw = self._obj_state_mean[2] + yaw_offset
                    # Set new cylinder pose
                    self.cylinder.set_pose(position=des_pos,
                                           orientation=create_quat(rot_yaw=des_yaw))
                    wrong_obj_pose = self.is_robot_touching_cylinder()

                yaw_offset = \
                    self.np_random.randn(1)*np.sqrt(self._obj_state_var[2])
            else:
                raise NotImplementedError
                # xy_offset = np.zeros(2)
                # yaw_offset = 0

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

            self._state['object_pose'][:] = self.get_object_pose()
            self._state['goal_pose'][:] = self.get_goal_pose()
            self._state['ee_pose'][:] = self.get_ee_pose()

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

            obj_state = self.get_object_pose()[:self._pose_dof]
            goal_state = self.get_goal_pose()[:self._pose_dof]
            ee_state = self.get_ee_pose()[:self._pose_dof]

            if self._use_obs_distances:
                self._observation['obj_ee_diff'][:] = obj_state - ee_state
                self._observation['goal_obj_diff'][:] = goal_state - obj_state
            else:
                self._observation['cylinder_pose'][:] = obj_state
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

            x_tol = -0.10 if self._half_env else -0.50
            done = ee_pos[0] <= x_tol
            if done:
                done_trigger = 'failure'
            else:
                done_trigger = False

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
        goal_pose = self.get_goal_pose()[:self._pose_dof]
        object_pose = self.get_object_pose()[:self._pose_dof]
        ee_pose = self.get_ee_pose()[:self._pose_dof]

        # Distance object to goal
        goal_obj_reward, _ = exponential_distance_reward(
            goal_pose - object_pose, temperature=0.1)
        goal_obj_reward = self._goal_cost_weight * goal_obj_reward

        # Give distance object to goal reward only if touching it
        goal_obj_reward *= (self.is_robot_touching_cylinder(only_with_ee=True)
                            and self.is_cylinder_inside_gripper())

        # Distance ee to object
        obj_ee_reward = exponential_distance_reward(object_pose - ee_pose,
                                                    temperature=0.6)[0].item()
        ee_obj_reward = self._tgt_cost_weight * obj_ee_reward

        # Control
        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()

        reward_composition = np.array([
            ee_obj_reward,
            goal_obj_reward,
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

    def set_tgt_pos(self, tgt_pose):
        self._tgt_state = tgt_pose

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
        gripper_pose = self.robot.links['gripper_center'].get_pose()
        return self.convert_2d_pose(gripper_pose)

    def get_object_pose(self):
        """ Returns pose of the object object, defined by (X, Y, yaw) """
        tgt_pose = self.cylinder.get_pose()
        return self.convert_2d_pose(tgt_pose)

    def get_goal_pose(self):
        """ Returns pose of the goal, defined by (X, Y, yaw) """
        goal_pose = self.goal.get_pose()
        return self.convert_2d_pose(goal_pose)

    def is_robot_touching_cylinder(self, only_with_ee=False):
        if only_with_ee:
            robot_body = self.robot.links['gripper']
        else:
            robot_body = None

        is_touching = len(self.get_contacts(self.robot, self.cylinder,
                                            robot_body)) > 0
        return is_touching

    def is_cylinder_inside_gripper(self):
        is_inside_gripper = np.linalg.norm(
            self.get_object_pose()[:2] - self.get_ee_pose()[:2]) < 0.03
        return is_inside_gripper

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

    env = Pusher2DBulletEnv(robot_dof=4,
                            is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        # next_obs, rew, done, info = env.step(env.action_space.sample())
        next_obs, rew, done, info = env.step(np.zeros_like(env.action_space.sample()))

        if done:
            print("The environment is done!")
            # break

        if render:
            env.render()

    print("Correct!")
