import numpy as np
from collections import OrderedDict

from robolearn_envs.pybullet.core.bullet_env import BulletEnv
from gym.spaces import Box

from robolearn_envs.pybullet.centauro.centauro \
    import Centauro

from robolearn_envs.pybullet.common import GoalArrow
from robolearn_envs.pybullet.common import Cylinder
from robolearn_envs.pybullet.common import Bottle
from robolearn_envs.pybullet.common import Plane


from pybullet import getEulerFromQuaternion
from robolearn_envs.utils.transformations import pose_transform
from robolearn_envs.utils.transformations import compute_cartesian_error
from robolearn_envs.utils.transformations import create_quat
from robolearn_envs.utils.transformations import euler_to_quat
from robolearn_envs.reward_functions.state_rewards \
    import exponential_distance_reward, exponential_distance_cost
from robolearn_envs.reward_functions.action_rewards import quadratic_action_cost


class CentauroTrayEnv(BulletEnv):
    """Centauro robot seeking to move a tray to a target position.
    """
    def __init__(self,
                 is_render=False,
                 active_joints='RA',
                 control_mode='joint_tasktorque',
                 obs_distances=True,
                 tgt_cost_weight=1.5,
                 balance_cost_weight=1.0,
                 fall_cost_weight=1.0,
                 ctrl_cost_weight=1.0e-2,
                 balance_done_cost=100,
                 tgt_done_reward=1000,
                 use_log_distances=False,
                 log_alpha_pos=1e-2,
                 log_alpha_ori=1e-2,
                 goal_tolerance=0.10,
                 max_tgt_distance=0.5,
                 min_obj_height=0.80,
                 max_obj_height=0.90,
                 max_obj_distance=0.20,
                 max_time=None,
                 subtask=None,
                 random_tgt=False,
                 random_config=False,
                 random_init=False,
                 sim_timestep=1/240.,
                 frame_skip=1,
                 seed=None,
                 ):

        super(CentauroTrayEnv,
              self).__init__(sim_timestep=sim_timestep, frameskip=frame_skip,
                             is_render=is_render, seed=seed)

        if random_init:
            self._random_tgt = True
            self._random_config = True
        else:
            self._random_tgt = random_tgt
            self._random_config = random_config

        self._use_obs_distances = obs_distances

        self._n_subgoals = 2

        # Environment/Scene
        self.done_multigoal = [False, False]
        self._pose_dof = 7  # [x, y, z, ow, ox, oy, oz]
        self._diff_dof = 6  # [dx, dy, dz, dR, dP, dY]

        # Plane
        # self._plane = PlaneObject(plane_type='stone',
        self._plane = Plane(plane_type='plane_simple',
                            color=None,
                            )
        self.add_to_sim(self._plane)

        # Object
        # self._object = Bottle(
        self._object = Cylinder(
            init_pos=(0.69089588, -0.46379092, 0.98),
            color='orange',
        )
        self.add_to_sim(self._object)
        self._object_offset_mean = [0.0, 0.0, 0.062, 0.0, 0.0, 0.0]
        self._object_offset_std = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        object_ori_offset = euler_to_quat(self._object_offset_mean[3:])
        self.object_offset = np.concatenate((self._object_offset_mean[:3],
                                             object_ori_offset))
        self._init_object_pose = np.zeros(self._pose_dof)
        self._prev_obj_tray_error = None

        # Target
        self._target = GoalArrow(
            init_pos=(1., 0., 0.8),
            tolerance=goal_tolerance,
            color='green',
        )
        self.add_to_sim(self._target)

        # self._target_offset_mean = np.array([0.10, 0.20, 0.05, 0.0, 0.0, 0.0])
        self._target_offset_mean = np.array([-0.10, 0.30, -0.05, 0.0, 0.0, 0.5])
        # self._target_offset_mean = np.array([0.00, 0.00, 0.00, .00, 0.00, 0.00])
        # self._target_offset_std = np.array([0.20, 0.3, 0.15, 0.00, 0.00, 0.6])
        self._target_offset_std = np.array([0.01, 0.01, 0.01, 0.00, 0.00, 0.01])
        target_ori_offset = euler_to_quat(self._target_offset_mean[3:])
        self.target_offset = np.concatenate((self._target_offset_mean[:3],
                                             target_ori_offset))
        self._target_pose_lims = np.array([
            [0.85, 0.95],
            [-0.15, 0.05],
            [0.87, 0.92],
            [0.00, 0.00],
            [0.00, 0.00],
            [-0.05, 0.70]
        ])
        self._init_target_pose = np.zeros(self._pose_dof)
        self._prev_tgt_tray_error = None

        # Robot
        init_pos = [0, 0, 0.7975]
        collision = True
        fixed_base = True
        self._robot = Centauro(
            init_config=None,
            init_pos=init_pos,
            control_mode=control_mode,
            self_collision=collision,
            active_joints=active_joints,
            robot_model='tray',
            fixed_base=fixed_base,
        )
        self.add_to_sim(self._robot)

        init_config = self._robot.initial_configuration
        init_config[12] = np.deg2rad(15)
        init_config[13] = np.deg2rad(32)
        init_config[14] = np.deg2rad(90)
        self._init_robot_config_mean = init_config

        self._init_robot_config_std = np.zeros_like(init_config)
        self._init_robot_config_std[8] = 0.15
        self._init_robot_config_std[9] = 0.15
        self._init_robot_config_std[10] = 0.15
        self._init_robot_config_std[11] = 0.15
        self._init_robot_config_std[12] = 0.3
        self._init_robot_config_std[13] = 0.3
        self._init_robot_config_std[14] = 0.3

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
            optitrack_dim = self._diff_dof*2 + self._diff_dof*2
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
            self._observation['tgt_tray_diff'] = np.zeros(self._diff_dof)
            self._observation['rate_tgt_tray_diff'] = np.zeros(self._diff_dof)
            self._observation['object_tray_diff'] = np.zeros(self._diff_dof)
            self._observation['rate_object_tray_diff'] = np.zeros(self._diff_dof)
        else:
            self._observation['tray_pose'] = np.zeros(self._pose_dof)
            self._observation['tgt_pose'] = np.zeros(self._pose_dof)
            self._observation['obj_pose'] = np.zeros(self._pose_dof)

        # STATE
        state_dim = robot_state_dim + self._pose_dof*3
        self.state_space = Box(
            low=-np.inf, high=np.inf, shape=(state_dim,),
            dtype=np.float32
        )
        self._state = OrderedDict(
            robot_state=np.zeros(robot_state_dim),
        )
        self._state['tray_pose'] = np.zeros(self._pose_dof)
        self._state['tgt_pose'] = np.zeros(self._pose_dof)
        self._state['obj_pose'] = np.zeros(self._pose_dof)

        # Environment settings
        self.max_time = max_time  # s
        self.set_rendering(self._is_render)
        self.set_visualizer_data(distance=2.5, pitch=-54.35, yaw=56.40,
                                 target_pos=[0.35, 0.0, 0.0])

        # Initial conditions
        self._current_init_cond = np.zeros(self.state_dim)
        self._init_robot_configs = []
        self._init_tgt_states = []
        self._init_obj_states = []
        self._initial_conditions = []
        self._initial_obs_conditions = []

        # Costs
        self._goal_tolerance = goal_tolerance
        self._min_obj_height = min_obj_height
        self._max_obj_height = max_obj_height
        self._max_obj_distance = max_obj_distance
        self._tgt_cost_weight = tgt_cost_weight
        self._balance_cost_weight = balance_cost_weight
        self._fall_cost_weight = fall_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._balance_done_cost = balance_done_cost
        self._tgt_done_reward = tgt_done_reward
        self._use_log_distances = use_log_distances
        self._log_alpha_pos = log_alpha_pos
        self._log_alpha_ori = log_alpha_ori
        # Update max reward
        self._max_rewards = np.array([0., 0., 0.])
        self._is_env_instantiation_complete = True

        self.active_subtask = subtask

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
        self._plane.reset()

        # Robot
        condition = kwargs.pop('condition', None)
        if condition is None:
            if self._random_config:
                n_joints = len(self._init_robot_config)
                self._init_robot_config = self._init_robot_config_mean + \
                    self._init_robot_config_std ** 2 * np.random.randn(n_joints)
        else:
            self._init_robot_config = self._init_robot_configs[condition]
        self._robot.initial_configuration = self._init_robot_config
        robot_state = self._robot.reset()

        # Rest object in simulation
        tray_pose = self.get_tray_pose()
        if condition is None:
            object_offset = self.object_offset
            object_pose = pose_transform(tray_pose, object_offset)
        else:
            object_pose = self.obj_init_conds[condition]

        self._object.reset(pose=object_pose)
        self._init_object_pose = object_pose

        # Target
        tray_pose = self.get_tray_pose()
        if condition is None:
            if self._random_tgt:
                # Sampling with uniform
                tgt_dist = np.inf
                while tgt_dist > 0.6 or tgt_dist < 0.2:
                    tgt_pos = np.random.uniform(
                        low=self._target_pose_lims[:3, 0],
                        high=self._target_pose_lims[:3, 1],
                    )
                    tgt_dist = np.linalg.norm(tgt_pos - tray_pose[:3])
                tgt_ori = np.random.uniform(
                    low=self._target_pose_lims[3:, 0],
                    high=self._target_pose_lims[3:, 1],
                )
                tgt_ori = create_quat(rot_yaw=tgt_ori[2])
                target_pose = np.concatenate((tgt_pos, tgt_ori))
            else:
                target_pose = pose_transform(tray_pose, self.target_offset)

        else:
            target_pose = self.tgt_init_conds[condition]
        self._target.reset(pose=target_pose)
        self._init_target_pose = target_pose

        if self._is_env_instantiation_complete:
            if self.active_subtask == 1:
                robot_uid = self._robot.id
                tray_index = self._robot._links['tray'].bodyPartIndex
                object_uid = self._object.id
                const_pos = [0.2, 0.0, 0.029+0.062]
                self.pbc.createConstraint(parentBodyUniqueId=robot_uid,
                                          parentLinkIndex=tray_index,
                                          childBodyUniqueId=object_uid,
                                          childLinkIndex=-1,
                                          jointType=self.pbc.JOINT_FIXED,
                                          jointAxis=(0., 0., 1.),
                                          parentFramePosition=const_pos,
                                          childFramePosition=(0., 0., 0.)
                                          )

        self._prev_tgt_tray_error = None
        self._prev_obj_tray_error = None

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

            # Relevant poses
            self._state['tray_pose'] = self.get_tray_pose()
            self._state['tgt_pose'] = self.get_target_pose()
            self._state['obj_pose'] = self.get_object_pose()

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

            tray_pose = self.get_tray_pose()
            tgt_pose = self.get_target_pose()
            obj_pose = self.get_object_pose()

            if self._use_obs_distances:
                # Tgt-Tray distance
                prev_tgt_tray_ori_error = \
                    None if self._prev_tgt_tray_error is None \
                        else self._prev_tgt_tray_error[3:]
                tgt_tray_error = \
                    compute_cartesian_error(tgt_pose, tray_pose,
                                            prev_ori_diff=prev_tgt_tray_ori_error)
                if self._prev_tgt_tray_error is None:
                    rate_tgt_tray_error = np.zeros_like(tgt_tray_error)
                else:
                    delta_tgt_tray_error = tgt_tray_error - self._prev_tgt_tray_error
                    # rate_tgt_tray_error = delta_tgt_tray_error / self.dt
                    rate_tgt_tray_error = delta_tgt_tray_error
                self._prev_tgt_tray_error = tgt_tray_error

                # Tray-Object distance
                prev_obj_tray_ori_error = \
                    None if self._prev_obj_tray_error is None \
                        else self._prev_obj_tray_error[3:]
                des_obj_pose = pose_transform(tray_pose, self.object_offset)
                object_tray_error = \
                    compute_cartesian_error(des_obj_pose, obj_pose,
                                            prev_ori_diff=prev_obj_tray_ori_error)
                if self._prev_obj_tray_error is None:
                    rate_object_tray_error = np.zeros_like(object_tray_error)
                else:
                    delta_object_tray_error = object_tray_error - self._prev_obj_tray_error
                    # rate_object_tray_error = delta_object_tray_error / self.dt
                    rate_object_tray_error = delta_object_tray_error
                self._prev_obj_tray_error = object_tray_error

                self._observation['tgt_tray_diff'][:] = tgt_tray_error
                self._observation['rate_tgt_tray_diff'][:] = rate_tgt_tray_error
                self._observation['object_tray_diff'][:] = object_tray_error
                self._observation['rate_object_tray_diff'][:] = rate_object_tray_error
            else:
                self._observation['tray_pose'][:] = tray_pose
                self._observation['tgt_pose'][:] = tgt_pose
                self._observation['obj_pose'][:] = obj_pose

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
            # Check if balancing task failed
            obj_height = self.get_object_pose()[2]
            is_falling = (obj_height < self._min_obj_height) or \
                         (obj_height > self._max_obj_height)

            obj_pos = self.get_object_pose()[:3]
            tray_pos = self.get_tray_pose()[:3]
            distance = np.linalg.norm(obj_pos - tray_pos)
            is_not_in_tray = distance > self._max_obj_distance

            balance_failed = is_falling or is_not_in_tray

            # Check if reaching task failed
            tray_x_fail = (tray_pos[0] < 0.5)  # or (tray_pos[0] > 1.0)
            tray_y_fail = (tray_pos[1] < -0.75) or \
                          (tray_pos[1] > 0.2)
            tray_z_fail = (tray_pos[2] < 0.55) or \
                          (tray_pos[2] > 1.5)
            tgt_reach_failed = tray_x_fail or tray_y_fail or tray_z_fail
            # print('fails-->', tray_x_fail, tray_y_fail, tray_z_fail, balance_failed)

            # TODO: Modifying this
            balance_failed = balance_failed or tgt_reach_failed

            # Any of both tasks failed
            is_done = balance_failed or tgt_reach_failed

            done = is_done
            # done_multigoal = [balance_failed, tgt_reach_failed]
            done_multigoal = [done for _ in range(self.n_subgoals)]
        else:
            if self.env_time >= self.max_time:
                done = True
                done_multigoal = [True for _ in range(self.n_subgoals)]
            else:
                done = False
                done_multigoal = [False for _ in range(self.n_subgoals)]

        done = done
        self.done_multigoal = done_multigoal

        info = {
            'done_multigoal': done_multigoal
        }

        if self.active_subtask is not None:
            done = done_multigoal[self.active_subtask].item()

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        object_pose = self._state['obj_pose']
        target_pose = self._state['tgt_pose']
        tray_pose = self._state['tray_pose']

        des_obj_pose = pose_transform(tray_pose, self.object_offset)
        object_tray_difference = compute_cartesian_error(
            des_obj_pose,
            object_pose,
            prev_ori_diff=self._prev_obj_tray_error[3:]
        )
        tgt_tray_difference = compute_cartesian_error(
            target_pose,
            tray_pose,
            prev_ori_diff=self._prev_tgt_tray_error[3:]
        )

        # Object balance cost (roll and pitch orientation)
        ori_distance_weights = np.array([5., 5.])
        obj_ori_norm = \
            np.linalg.norm(ori_distance_weights *
                           getEulerFromQuaternion(object_pose[3:])[:2]
                           )
        obj_balance_reward = self._balance_cost_weight * \
            exponential_distance_reward(obj_ori_norm,
                                        # temperature=1.)[0]  # Antes 27-02 9pm
                                        temperature=0.8)[0]

        # Falling cost
        pos_dist_weight = 50
        falling_norm = np.linalg.norm(pos_dist_weight*object_tray_difference[2])
        falling_reward = self._fall_cost_weight * \
            exponential_distance_reward(falling_norm,
                                        temperature=1.)[0]

        # Distance to Target cost (position)
        pos_distance_weights = np.array([1.0, 1.0, 1.5])
        tgt_tray_pos_norm = np.linalg.norm(pos_distance_weights *
                                           tgt_tray_difference[:3])
        tgt_pos_reward = exponential_distance_reward(
            tgt_tray_pos_norm, temperature=0.4)[0]  # 10-01
            # tgt_tray_pos_norm, temperature=0.8)[0]

        # Distance to Target cost (orientation)
        ori_distance_weights = np.array([1.0, 1.0, 5.0])
        tgt_tray_ori_norm = np.linalg.norm(ori_distance_weights *
                                           tgt_tray_difference[3:])
        tgt_ori_reward = exponential_distance_reward(
            tgt_tray_ori_norm, temperature=0.8)[0]

        # Distance to Target: tgt + distance
        tgt_reward = self._tgt_cost_weight * \
            (tgt_pos_reward + tgt_ori_reward * 0.01)
        # print('rew-->', tgt_tray_pos_norm, tgt_pos_reward, tgt_ori_reward * 0.01)
        # print('rew-->', tgt_pos_reward, tgt_ori_reward * 0.01)

        # Control
        action_cost = quadratic_action_cost(action=action)[0]
        ctrl_cost = self._ctrl_cost_weight * action_cost

        # Done Cost
        balance_done_reward = 0

        tgt_done_reward = 0

        reward_composition = np.array([
            -ctrl_cost,
            obj_balance_reward,
            falling_reward,
            balance_done_reward,
            tgt_reward,
            tgt_done_reward,
            -self._max_rewards[0],
        ])

        reward_subtasks = np.array([
            # Balance Task
            (- ctrl_cost + obj_balance_reward + falling_reward + balance_done_reward
             - self._max_rewards[1]),
            # Reaching Task
            (- ctrl_cost + tgt_reward + tgt_done_reward
             - self._max_rewards[2]),
        ]).squeeze()

        # reward = reward_subtasks.mean().item()
        reward = reward_composition.sum().item()

        info = {
            'reward_composition': reward_composition,
            'reward_multigoal': reward_subtasks,
        }

        if self.active_subtask is not None:
            reward = float(reward_subtasks[self.active_subtask])

        return reward, info

    def _calc_max_reward(self):
        max_reward = np.zeros(3)
        state = np.zeros(self._state_dim)
        des_tray_pose = self.get_target_pose()
        des_object_pose = pose_transform(des_tray_pose, self.object_offset)
        state[self.get_state_info('object')['idx']] = des_object_pose
        state[self.get_state_info('target')['idx']] = des_tray_pose
        state[self.get_state_info('tray')['idx']] = des_tray_pose

        max_reward[0], _, (max_reward[1], max_reward[2]) = \
            self._compute_reward(state,
                                 np.zeros(self.action_dim),
                                 [False, False])
        return max_reward

    def add_initial_condition(self, robot_config, tgt_state, obj_state):
        """

        :param robot_config:
        :param tgt_state: Relative (X, Y, ...) to hand
        :param obj_state: Relative (X, Y, ...) to hand
        :return:
        """
        self._init_robot_configs.append(robot_config)
        self._init_tgt_states.append(tgt_state)
        self._init_obj_states.append(obj_state)

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
            self._init_tgt_states.clear()
            self._init_obj_states.clear()
            self._initial_conditions.clear()
        else:
            self._init_robot_configs.pop(idx)
            self._init_tgt_states.pop(idx)
            self._init_obj_states.pop(idx)
            self._initial_conditions.pop(idx)

    def get_conditions(self, cond=None):
        if cond is None:
            return list(self._initial_conditions)
        else:
            return self._initial_conditions[cond]

    def get_total_joints(self):
        return self._robot.total_joints

    def get_tray_pose(self, ori='quat'):
        """
        Returns pose of the tray.
        :param ori: Orientation representation. Options: 'quat' or 'rpy'
        :return:
        """
        tray_pose = self._robot.get_body_pose('tray')
        tray_offset = np.array([0.2, 0.0, 0.029, 0, 0, 0, 1])
        tray_pose = pose_transform(tray_pose, tray_offset)

        if ori == 'rpy':
            ori = getEulerFromQuaternion(tray_pose[3:])
            tray_pose = np.concatenate((tray_pose[:3], ori))

        return tray_pose

    def get_object_pose(self, ori='quat'):
        """
        Returns pose of the balanced object.
        :param ori: Orientation representation. Options: 'quat' or 'rpy'
        :return:
        """
        object_pose = self._object.get_pose()

        if ori == 'rpy':
            ori = getEulerFromQuaternion(object_pose[3:])
            object_pose = np.concatenate((object_pose[:3], ori))

        return object_pose

    def get_target_pose(self, ori='quat'):
        """ Returns pose of the target object. """
        tgt_pose = self._target.get_pose()

        if ori == 'rpy':
            ori = getEulerFromQuaternion(tgt_pose[3:])
            tgt_pose = np.concatenate((tgt_pose[:3], ori))

        return tgt_pose

    @property
    def initial_conditions(self):
        return self._initial_conditions

    @property
    def initial_obs_conditions(self):
        return self._initial_obs_conditions

    def set_robot_config(self, joint_config):
        self._robot.set_state(joint_config, np.zeros_like(joint_config))

    def viewer_setup(self):
        raise NotImplementedError

    def log_diagnostics(self, paths):
        all_rewards = np.concatenate([
            np.expand_dims(path['reward_functions'], axis=0)
            for path in paths
        ], axis=0,
        )
        print("Average Return:", np.mean(np.sum(all_rewards, axis=1)))

    @property
    def n_subgoals(self):
        return self._n_subgoals

    @property
    def active_subtask(self):
        return self._active_subtask

    @active_subtask.setter
    def active_subtask(self, subtask):
        if subtask not in list(range(0, self.n_subgoals)) + [None]:
            print("Wrong option '%s'!" % subtask)
        self._active_subtask = None if subtask == -1 else subtask

    def get_active_subtask(self):
        return self.active_subtask

    def set_active_subtask(self, subtask):
        self.active_subtask = subtask


if __name__ == "__main__":
    # render = False
    render = True

    H = 100

    env = CentauroTrayEnv(is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        next_obs, rew, done, info = env.step(env.action_space.sample())
        if render:
            env.render()

        if done:
            print('The environment is done!')
            break

    print("Correct!")
