import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from gym.spaces import Box

from robolearn_envs.pybullet.centauro.centauro \
    import Centauro

from robolearn_envs.pybullet.common import GoalArrow
from robolearn_envs.pybullet.common import Cylinder
from robolearn_envs.pybullet.common import Drill
from robolearn_envs.pybullet.common import Plane
from robolearn_envs.pybullet.common import MetalTable1

from pybullet import getEulerFromQuaternion
from robolearn_envs.utils.transformations import pose_transform
from robolearn_envs.utils.transformations import compute_cartesian_error
from robolearn_envs.utils.transformations import create_quat_pose
from robolearn_envs.utils.transformations import euler_to_quat


class CentauroObstacleEnv(BulletEnv):
    def __init__(self,
                 is_render=False,
                 active_joints='RA',
                 control_mode='joint_tasktorque',
                 obs_distances=True,
                 goal_tolerance=0.02,
                 obstacle='cylinder',
                 max_time=None,
                 sim_timestep=1/240.,
                 frame_skip=1,
                 seed=None,
                 ):
        """
        Centauro robot seeking to move the righthand to a target position
        while avoiding an obstacle.
        """
        super(CentauroObstacleEnv,
              self).__init__(sim_timestep=sim_timestep, frameskip=frame_skip,
                             is_render=is_render, seed=seed)

        self._use_obs_distances = obs_distances

        # Environment/Scene
        self._pose_dof = 7  # [x, y, z, ow, ox, oy, oz]
        self._diff_dof = 6  # [dx, dy, dz, dR, dP, dY]

        # Plane
        # self._plane = PlaneObject(plane_type='stone',
        self._plane = Plane(plane_type='plane_simple',
                            color=None,
                            )
        self.add_to_sim(self._plane)

        # Table
        self._table = MetalTable1(
            init_pos=(1.0, 0., 0.),
        )
        self.add_to_sim(self._table)

        # Obstacle
        if obstacle == 'cylinder':
            self._obstacle_fixed_height = 0.75 + 0.10 + 0.001
            self._obstacle = Cylinder(
                init_pos=(1.0, 0.0, self._obstacle_fixed_height),
                cylinder_type='T',
                color='red',
                fixed_base=False,
            )
        else:
            self._obstacle_fixed_height = 0.8815
            self._obstacle = Drill(
                init_pos=(1.0, 0.0, self._obstacle_fixed_height),
                color='red',
                fixed_base=False,
            )
        self.add_to_sim(self._obstacle)

        obstacle_pos_offset = [0.08, 0.25, 0.0]
        obstacle_ori_offset = [0, 0.7071068, 0, 0.7071068]
        self.obstacle_offset = np.concatenate((obstacle_pos_offset,
                                               obstacle_ori_offset))
        self._init_obstacle_pose = np.zeros(7)
        self._prev_obst_hand_ori_diff = None

        # Target
        self.goal = GoalArrow(
            init_pos=(1., 0., 0.8),
            tolerance=0.025,
            color='green',
        )
        self.add_to_sim(self.goal)

        self._target_offset_mean = np.array([0.10, 0.4, 0.0,
                                             0.0, 0.0, 0.0])
        target_ori_offset = euler_to_quat(self._target_offset_mean[3:])
        self.target_offset = np.concatenate((self._target_offset_mean[:3],
                                             target_ori_offset))
        self._init_goal_pose = np.zeros(self._pose_dof)
        self._prev_tgt_hand_ori_diff = None

        # Robot
        init_pos = [0, 0, 0.7975]
        collision = True
        fixed_base = True
        # fixed_base = False
        self._robot = Centauro(
            init_config=None,
            init_pos=init_pos,
            control_mode=control_mode,
            self_collision=collision,
            active_joints=active_joints,
            robot_model=None,
            fixed_base=fixed_base,
        )
        self.add_to_sim(self._robot)

        init_config = self._robot.initial_configuration
        init_config[9] = np.deg2rad(45)
        # init_config[12] = np.deg2rad(15)
        # init_config[13] = np.deg2rad(32)
        # init_config[14] = np.deg2rad(90)
        self._init_robot_config_mean = init_config

        self._init_robot_config_std = np.zeros_like(init_config)

        self._init_robot_config = self._init_robot_config_mean

        # Reset environment so we can get info from pybullet
        self.set_rendering(False)
        self._is_env_instantiation_complete = False
        self.reset()
        self._is_render = is_render

        # ACTUATION
        self.action_space = Box(
            self._robot.low_action_bounds,
            self._robot.high_action_bounds,
            dtype=np.float32
        )

        # OBSERVATION
        robot_state_dim = self._robot.observation_dim  # joint pos/vel
        # Optitrack
        if obs_distances:
            optitrack_dim = self._diff_dof*2
        else:
            optitrack_dim = self._pose_dof*3

        obs_dim = robot_state_dim + optitrack_dim
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,),
            dtype=np.float32
        )
        self._observation = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
        )
        if obs_distances:
            self._observation['goal_ee_diff'] = np.zeros(self._diff_dof)
            self._observation['obstacle_ee_diff'] = np.zeros(self._diff_dof)
        else:
            self._observation['goal_pose'] = np.zeros(self._pose_dof)
            self._observation['obstacle_pose'] = np.zeros(self._pose_dof)
            self._observation['ee_pose'] = np.zeros(self._pose_dof)

        # STATE
        state_dim = robot_state_dim + self._pose_dof*3
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
        )
        self._state['goal_pose'] = np.zeros(self._pose_dof)
        self._state['obstacle_pose'] = np.zeros(self._pose_dof)
        self._state['ee_pose'] = np.zeros(self._pose_dof)

        # Environment settings
        self.max_time = max_time  # s
        self.set_rendering(self._is_render)
        # self.set_visualizer_data(distance=2.5, yaw=30.80, pitch=-37.55,
        #                          target_pos=[0.35, 0.0, 0.0])  # XY?
        # self.set_visualizer_data(distance=2.57, yaw=54.0, pitch=-54.75,
        #                          target_pos=[0.35, 0.0, 0.0])  # Humanoids
        # self.set_visualizer_data(distance=1.57, yaw=79.20, pitch=-43.95,
        #                          target_pos=[0.35, 0.0, 0.0])  # Debugging
        self.set_visualizer_data(distance=1.50, yaw=50.80, pitch=-27.95,
                                 target_pos=[0.44, 0.05, 0.65])  # Video

        # Initial conditions
        self._current_init_cond = np.zeros(self.state_dim)
        self._init_robot_configs = []
        self._init_goal_states = []
        self._init_obst_states = []
        self._initial_conditions = []
        self._initial_obs_conditions = []

        # Costs
        self._goal_tolerance = goal_tolerance

        # Update max reward
        self._max_rewards = np.array([0., 0., 0.])
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
        self._plane.reset()

        # Reset table in simulation
        self._table.reset(pose=None)

        # Robot
        condition = kwargs.pop('condition', None)
        if condition is not None:
            init_robot_config = self._init_robot_configs[condition]
        else:
            init_robot_config = self._robot.initial_configuration
        self._robot.initial_configuration = init_robot_config
        robot_state = self._robot.reset()

        # Object
        hand_pose = self.get_hand_pose()
        if condition is None:
            obstacle_offset = self.obstacle_offset
            obstacle_pose = pose_transform(hand_pose, obstacle_offset)
        else:
            # obstacle_offset = self._init_obst_states[condition]
            # obstacle_pose = pose_transform(hand_pose, obstacle_offset)
            obstacle_pose = self._init_obst_states[condition]
        obstacle_pose[2] = self._obstacle_fixed_height
        obstacle_pose[3:] = [0., 0., 0., 1.]
        self._obstacle.reset(pose=obstacle_pose)
        self._init_obstacle_pose = obstacle_pose

        # Goal
        hand_pose = self.get_hand_pose()
        if condition is None:
            tgt_offset = self.target_offset
            goal_pose = pose_transform(hand_pose, tgt_offset)
        else:
            # tgt_offset = self.tgt_init_conds[condition]
            goal_pose = self._init_goal_states[condition]
        self.goal.reset(pose=goal_pose)
        self._init_goal_pose = goal_pose

        self._prev_tgt_hand_ori_diff = None
        self._prev_obst_hand_ori_diff = None

        # Visualization (if applicable)
        self.enable_vis()

        # Update Environment State and Observation
        state = self._update_env_state()
        observation = self._update_env_obs()

        # Update max reward
        if self._is_env_instantiation_complete:
            self._max_rewards = np.array([0., 0., 0.])
            # self._max_rewards = self._calc_max_reward()

        return observation

    # ############## #
    # ACTION related #
    # ############## #

    def _set_action(self, action):
        self._robot.apply_action(action)

        # Simulation step
        self.sim_step()

    # ############# #
    # STATE related #
    # ############# #
    def _update_env_state(self):
        if self._is_env_instantiation_complete:
            # Robot state
            self._state['robot_state'][:] = \
                self._robot.get_state(only_controlled_joints=True)

            self._state['ee_pose'] = self.get_hand_pose(ori='quat')
            self._state['goal_pose'] = self.get_target_pose(ori='quat')
            self._state['obstacle_pose'] = self.get_obstacle_pose(ori='quat')

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
                self._robot.set_state(self._state[key])
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
                self._robot.get_state(only_controlled_joints=True)

            target_pose = self.get_target_pose(ori='quat')
            hand_pose = self.get_hand_pose(ori='quat')
            obstacle_pose = self.get_obstacle_pose(ori='quat')

            if self._use_obs_distances:
                # Tgt-Hand distance
                tgt_hand_diff = \
                    compute_cartesian_error(target_pose, hand_pose,
                                            prev_ori_diff=self._prev_tgt_hand_ori_diff)
                self._prev_tgt_hand_ori_diff = tgt_hand_diff[3:]

                # Obst-Hand distance
                obst_hand_diff = \
                    compute_cartesian_error(obstacle_pose, hand_pose,
                                            prev_ori_diff=self._prev_obst_hand_ori_diff)
                self._prev_obst_hand_ori_diff = obst_hand_diff[3:]

                self._observation['goal_ee_diff'][:] = tgt_hand_diff
                self._observation['obstacle_ee_diff'][:] = obst_hand_diff
            else:
                self._observation['goal_pose'][:] = target_pose
                self._observation['obstacle_pose'][:] = obstacle_pose
                self._observation['ee_pose'][:] = hand_pose

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
        # if self.max_time is None:
        #     obj_height = self.get_obstacle_pose()[2]
        #     is_falling = (obj_height < self._min_obj_height) or \
        #                  (obj_height > self._max_obj_height)
        #     # print(is_falling, obj_height, self._min_obj_height, self._max_obj_height)
        #     return is_falling, [is_falling, is_falling]
        # else:
        #     if self.env_time >= self.max_time:
        #         return True, [True]
        #     else:
        #         return False, [False]
        done = False

        info = {}

        return done, info

    def _compute_reward(self, *args, **kwargs):
        reward = 0
        info = {}
        return reward, info

    def _calc_max_reward(self):
        max_reward = np.zeros(3)
        # max_reward[0], _, (max_reward[1], max_reward[2]) = \
        #     self._compute_reward(self._observation,
        #                          np.zeros(self.action_dim))
        return max_reward

    def get_hand_pose(self, ori='quat'):
        """ Returns pose of the hand. """
        hand_pose = self._robot._links['right_hand'].get_pose()

        hand_offset = create_quat_pose(rot_pitch=np.pi/2)

        hand_pose = pose_transform(hand_pose, hand_offset)

        if ori == 'rpy':
            ori = getEulerFromQuaternion(hand_pose[3:])
            hand_pose = np.concatenate((hand_pose[:3], ori))

        return hand_pose

    def get_obstacle_pose(self, ori='quat'):
        """ Returns pose of the obstacle. """
        obstacle_pose = self._obstacle.get_pose()

        if ori == 'rpy':
            ori = getEulerFromQuaternion(obstacle_pose[3:])
            obstacle_pose = np.concatenate((obstacle_pose[:3], ori))

        return obstacle_pose

    def get_target_pose(self, ori='quat'):
        """ Returns pose of the target object. """
        tgt_pose = self.goal.get_pose()

        if ori == 'rpy':
            ori = getEulerFromQuaternion(tgt_pose[3:])
            tgt_pose = np.concatenate((tgt_pose[:3], ori))

        return tgt_pose

    def get_table_pose(self, ori='quat'):
        """ Returns pose of the table. """
        tgt_pose = self._table.get_pose()

        if ori == 'rpy':
            ori = getEulerFromQuaternion(tgt_pose[3:])
            tgt_pose = np.concatenate((tgt_pose[:3], ori))

        return tgt_pose

    def add_initial_condition(self, robot_config, tgt_state, obst_state):
        """

        :param tgt_state: Relative (X, Y, ...) to hand
        :param obst_state: Relative (X, Y, ...) to hand
        :return:
        """
        self._init_robot_configs.append(robot_config)
        self._init_goal_states.append(tgt_state)
        self._init_obst_states.append(obst_state)

        self.set_rendering(False)
        self.reset(-1)
        self.set_rendering(self._is_render)
        full_state = self.get_state()
        full_obs = self.get_observation()

        self._initial_conditions.append(full_state)
        self._initial_obs_conditions.append(full_obs)

    def clear_initial_condition(self, idx=None):
        if idx is None:
            self._init_robot_configs.clear()
            self._init_goal_states.clear()
            self._init_obst_states.clear()
            self._initial_conditions.clear()
        else:
            self._init_robot_configs.pop(idx)
            self._init_goal_states.pop(idx)
            self._init_obst_states.pop(idx)
            self._initial_conditions.pop(idx)

    @property
    def initial_conditions(self):
        return self._initial_conditions


if __name__ == "__main__":
    # render = False
    render = True

    H = 100

    env = CentauroObstacleEnv(is_render=render, obstacle='drill')
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        env.step(env.action_space.sample())
        if render:
            env.render()

    print("Correct!")
