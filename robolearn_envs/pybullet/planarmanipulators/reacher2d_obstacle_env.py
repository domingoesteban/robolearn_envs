import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from gym.spaces import Box

from robolearn_envs.pybullet.planarmanipulators.planarmanipulator3dof \
    import PlanarManipulator3Dof
from robolearn_envs.pybullet.planarmanipulators.planarmanipulator4dof \
    import PlanarManipulator4Dof

from robolearn_envs.pybullet.common import GoalCircle
from robolearn_envs.pybullet.common import Cylinder
from robolearn_envs.pybullet.common import Plane
from robolearn_envs.pybullet.common import PusherBorder

from robolearn_envs.utils.transformations import create_quat
from robolearn_envs.utils.transformations import euler_from_quat


class Reacher2DObstacleEnv(BulletEnv):
    def __init__(
            self,
            robot_dof=3,
            is_render=False,
            obs_distances=True,
            only_position=False,
            rdn_tgt_pose=True,
            tgt_cost_weight=1.0,
            obst_cost_weight=2.0,
            ctrl_cost_weight=1.0e-2,
            safe_radius=0.17,
            inside_cost=1,
            outside_cost=0,
            half_env=False,
            max_time=None,  # seconds
            sim_timestep=1/240.,
            frame_skip=1,
            seed=None,
    ):
        """Planar3DoF robot seeking to reach a target.
        """
        super(Reacher2DObstacleEnv, self).__init__(
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
        init_robot_config = [np.deg2rad(-90), np.deg2rad(20), np.deg2rad(2), np.deg2rad(2)]
        # _init_robot_config = [np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]
        if robot_dof == 3:
            self.robot = PlanarManipulator3Dof(
                self_collision=True,
                init_config=init_robot_config,
                robot_model=None,
                control_mode='joint_torque',
            )
        elif robot_dof == 4:
            self.robot = PlanarManipulator4Dof(
                self_collision=True,
                init_config=init_robot_config,
                robot_model=None,
                control_mode='joint_torque',
            )
        else:
            raise NotImplementedError("Only planar manipulators with 3 and 4"
                                      "DoFs have been implemented!")
        self.add_to_sim(self.robot)
        self._init_robot_config = self.robot.initial_configuration
        self._ee_offset = 0.01

        self.tgt_pose_is_rdn = rdn_tgt_pose
        # Goal
        self.goal_state_mean = [0.3693, 0.6511, 1.4660]
        self.goal_state_var = [0.01, 0.01, 0.2]
        self.goal_height = 0.01
        self.goal = GoalCircle(init_pos=[self.goal_state_mean[0],
                                         self.goal_state_mean[1],
                                         self.goal_height],
                               color='green')
        self.add_to_sim(self.goal)

        # Obstacle
        self.obst_state_mean = [0.6545,  -0.0576,   0.]
        self.obst_state_var = [0.005, 0.01, 0.2]
        self.obst_height = 0.051
        self.obstacle = Cylinder(init_pos=[self.obst_state_mean[0],
                                           self.obst_state_mean[1],
                                           self.obst_height],
                                 cylinder_type='heavy',
                                 fixed_base=True,
                                 color='red')
        self.add_to_sim(self.obstacle)

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
            self._observation['goal_ee_diff'] = np.zeros(self._pose_dof)
            self._observation['ee_obstacle_diff'] = np.zeros(self._pose_dof)
        else:
            self._observation['goal_pose'] = np.zeros(self._pose_dof)
            self._observation['obstacle_pose'] = np.zeros(self._pose_dof)
            self._observation['ee_pose'] = np.zeros(self._pose_dof)

        # STATE
        state_dim = robot_state_dim + 3*3  # robot_state + (XYyaw)*(GOAL, OBST, EE)
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
            obstacle_pose=np.zeros(3),
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
        self.tgt_cost_weight = tgt_cost_weight
        self.obst_cost_weight = obst_cost_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.safe_radius = safe_radius
        self.inside_cost = inside_cost
        self.outside_cost = outside_cost

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
        robot_state = self.robot.reset()

        # Goal
        if self.tgt_pose_is_rdn:
            xy_offset = \
                self.np_random.randn(2)*np.sqrt(self.goal_state_var[:2])
            yaw_offset = \
                self.np_random.randn(1)*np.sqrt(self.goal_state_var[2])
        else:
            xy_offset = np.zeros(2)
            yaw_offset = 0
        # Set XY position
        des_pos = np.zeros(3)
        des_pos[:2] = self.goal_state_mean[:2] + xy_offset
        # Set the height (Z)
        des_pos[2] = self.goal_height
        # Set the orientation (Yaw)
        des_yaw = self.goal_state_mean[2] + yaw_offset
        # Set the pose of _object object
        des_pose = np.concatenate((des_pos, create_quat(rot_yaw=des_yaw)))
        self.goal.reset(pose=des_pose)

        # Obstacle
        if self.tgt_pose_is_rdn:
            xy_offset = \
                self.np_random.randn(2)*np.sqrt(self.obst_state_var[:2])
            yaw_offset = \
                self.np_random.randn(1)*np.sqrt(self.obst_state_var[2])
        else:
            xy_offset = np.zeros(2)
            yaw_offset = 0
        # Set XY position
        des_pos = np.zeros(3)
        des_pos[:2] = self.obst_state_mean[:2] + xy_offset
        # Set the height (Z)
        des_pos[2] = self.obst_height
        # Set the orientation (Yaw)
        des_yaw = self.obst_state_mean[2] + yaw_offset
        # Set the pose of _object object
        des_pose = np.concatenate((des_pos, create_quat(rot_yaw=des_yaw)))
        self.obstacle.reset(des_pose)

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
            self._state['obstacle_pose'][:] = self.get_goal_pose()
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

            ee_state = self.get_ee_pose()[:self._pose_dof]
            goal_state = self.get_goal_pose()[:self._pose_dof]
            obst_state = self.get_obstacle_pose()[:self._pose_dof]

            if self._use_obs_distances:
                # Goal-EE difference
                self._observation['goal_ee_diff'][:] = goal_state - ee_state
                # EE-Obst difference
                self._observation['ee_obstacle_diff'][:] = ee_state - obst_state
            else:
                self._observation['ee_pose'][:] = ee_state
                self._observation['goal_pose'][:] = goal_state
                self._observation['obstacle_pose'][:] = obst_state

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
        # TODO: Add collision termination

        if self.max_time is None:
            done = False
            done_trigger = False
        else:
            if self.env_time >= self.max_time:
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
        gripper_pos = self.get_ee_pose()[:2]

        # Distance to goal
        target_pos = self.get_goal_pose()[:2]
        vec = gripper_pos - target_pos
        tgt_dist_cost = .5 * np.square(np.linalg.norm(vec)) * \
                        self.tgt_cost_weight  # L2

        # Safe Distance to obstacle
        dist_scale = 8
        obst_pos = self.get_obstacle_pose()[:2]
        vec = gripper_pos[:2] - obst_pos
        dist = self.safe_radius - np.linalg.norm(vec)
        dist_violation = dist > 0
        safe_dist_cost = 0.5*np.square(dist*dist_scale) * \
                         (dist_violation*self.inside_cost +
                          ~dist_violation*self.outside_cost) * \
                         self.obst_cost_weight

        # safe_distance = np.sqrt([self.safe_radius**2/2, self.safe_radius**2/2])
        # dist = safe_distance - np.abs(vec)
        # dist_violation = dist > 0
        # safe_dist_cost = np.sum(dist*(dist_violation*self.inside_cost
        #                               + ~dist_violation*self.outside_cost)) * \
        #                  dist = self.safe_radius - np.linalg.norm(vec)
        # self.obst_cost_weight

        # Control
        # reward_ctrl = -np.square(action).sum()
        lb = self.robot.low_action_bounds
        ub = self.robot.high_action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_weight * np.sum(
            np.square(action / scaling))

        reward_composition = np.array([
            -tgt_dist_cost,
            -safe_dist_cost,
            -ctrl_cost
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
        self.tgt_pose = tgt_pose

    def add_custom_init_cond(self, cond):
        if len(cond) != self._obs_dim:
            raise ValueError('Wrong initial condition size. (%d != %d)'
                             % (len(cond), self._obs_dim))

        if self.init_custom_cond is None:
            self.init_custom_cond = []
            self.init_cond = []

        self.init_custom_cond.append(cond)
        self.init_cond.append(self.reset(-1))

    def clear_custom_init_cond(self, idx=None):
        if idx is None:
            self.init_custom_cond = []
            self.init_cond = []
        else:
            self.init_custom_cond.pop(idx)
            self.init_cond.pop(idx)

    def get_conditions(self, cond=None):
        if cond is None:
            return list(self._initial_conditions)
        else:
            return self._initial_conditions[cond]

    def get_custom_conditions(self, cond=None):
        if cond is None:
            return list(self.init_custom_cond)
        else:
            return self.init_custom_cond[cond]

    def get_total_joints(self):
        return self.robot.total_joints

    def get_ee_pose(self):
        """ Returns pose of the end-effector, defined by (X, Y, yaw) """
        gripper_pose = self.robot._links['gripper_center'].get_pose()
        return self.convert_2d_pose(gripper_pose)

    def get_obstacle_pose(self):
        """ Returns pose of the obstacle, defined by (X, Y, yaw) """
        gripper_pose = self.obstacle.get_pose()
        return self.convert_2d_pose(gripper_pose)

    def get_goal_pose(self):
        """ Returns pose of the goal, defined by (X, Y, yaw) """
        goal_pose = self.goal.get_pose()
        return self.convert_2d_pose(goal_pose)

    def is_robot_touching_cylinder(self):
        contact_pointsA = self._pb_client.getContactPoints(bodyA=self.robot._robot_uid)
        contact_pointsB = self._pb_client.getContactPoints(bodyB=self.robot._robot_uid)
        # print(len(contact_pointsA), len(contact_pointsB))
        if len(contact_pointsA) > 0 or len(contact_pointsB) > 0:
            return True
        else:
            return False

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

    H = 1000000

    env = Reacher2DObstacleEnv(robot_dof=4,
                               is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        next_obs, rew, done, info = env.step(env.action_space.sample())
        # next_obs, rew, done, info = env.step(np.zeros_like(env.action_space.sample()))

        if done:
            print("The environment is done!")
            break

        if render:
            env.render()

    print("Correct!")
