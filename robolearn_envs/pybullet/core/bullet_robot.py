"""
Based on PyBullet Gym examples file: robot_bases.py
https://github.com/bulletphysics/bullet3
"""
from __future__ import print_function
import numpy as np
from collections import OrderedDict
import os.path as osp
import logging
from robolearn_envs.core.robolearn_robot import RoboLearnRobot
from robolearn_envs.pybullet.core.bullet_utils import Link
from robolearn_envs.pybullet.core.bullet_utils import Joint
from robolearn_envs.pybullet.core.bullet_utils import get_joint_info
from robolearn_envs.pybullet.core.bullet_sensors import Camera
from robolearn_envs.pybullet.core.bullet_sensors import Encoder
from robolearn_envs.pybullet.core.bullet_actuators import DirectJointController
from robolearn_envs.pybullet.core.bullet_multibody import BulletMultibody

from robolearn_envs.utils.robot_model import RobotModel


class BulletRobot(RoboLearnRobot, BulletMultibody):
    def __init__(
            self,
            model_file,
            base_name,
            init_pos=None,
            init_ori=None,
            joint_names=None,

            controlled_joint_idxs=None,
            init_config=None,
            self_collision=True,
            use_file_inertia=True,
            fixed_base=False,
            robot_model=False,
            pybullet_client=None,
    ):
        """

        Args:
            model_file (str):
            base_name (str):
            init_pos:
            init_ori:
            joint_names:
            controlled_joint_idxs:
            init_config:
            self_collision:
            use_file_inertia:
            fixed_base:
            robot_model:
            pybullet_client:
        """
        BulletMultibody.__init__(self, pybullet_client)

        model_type = osp.splitext(model_file)[1][1:]
        if model_type not in ['urdf', 'mjcf', 'sdf']:
            raise NotImplementedError("Wrong model_type: %s in %s. "
                                      "Only .urdf, .mjfc, .sdf are  supported"
                                      % (model_type, model_file))

        self._model_type = model_type
        self.model_xml = model_file
        self._base_name = base_name

        self._self_collision = self_collision
        self._use_file_inertia = use_file_inertia
        self._is_fixed_base = fixed_base

        self._joint_names = joint_names

        # Initial base pose
        if init_pos is None:
            init_pos = [0., 0., 0.]
        self._init_base_pos = np.array(init_pos)

        if init_ori is None:
            init_ori = [0., 0., 0., 1.]
        self._init_base_ori = np.array(init_ori)

        # Initial Configuration
        self._initial_configuration = init_config

        self._base_link = None  # Robot base link

        self._pbactive_joint_idxs = []  # Idxs pybullet active joints
        self._pbordered_joint_idxs = []  # Idxs pybullet ordered joints
        self._pbcontrolled_joint_idxs = []  # Idxs pybullet controlled joints

        self.controlled_joint_idxs = controlled_joint_idxs
        self._end_effectors_ids = None
        self._end_effectors_names = None
        # self._dof = 0
        self.initial_state = None

        # Robot Dynamic Model
        if robot_model:
            if self._model_type != 'urdf':
                raise ValueError("robot_model only works with urdf")
            self._robot_model = RobotModel(model_file, not fixed_base)

        # Logger
        self.logger = logging.getLogger('pybullet')
        self.logger.setLevel(logging.WARNING)
        self.logger.setLevel(logging.ERROR)
        # self.logger.setLevel(logging.INFO)
        # console = logging.StreamHandler()
        # self.logger.addHandler(console)

    def spawn_in_pb(self):
        # TODO: Use self.use_file_inertia for urdf
        if self._model_type == 'urdf':
            load_fcn = self.pbc.loadURDF
        elif self._model_type == 'mjcf':
            load_fcn = self.pbc.loadMJCF
        elif self._model_type == 'sdf':
            load_fcn = self.pbc.loadSDF
        else:
            load_fcn = self.pbc.loadURDF

        # Spawn the robot again
        if self._model_type == 'urdf':
            if self._self_collision:
                # flags = self.pbc.URDF_USE_SELF_COLLISION
                # flags = self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                # flags = self.pbc.URDF_USE_SELF_COLLISION + \
                #         self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
                # flags = self.pbc.URDF_USE_SELF_COLLISION + \
                #         self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                # flags = self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT + \
                #         self.pbc.URDF_USE_INERTIA_FROM_FILE
                flags = self.pbc.URDF_USE_SELF_COLLISION + \
                        self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS + \
                        self.pbc.URDF_USE_INERTIA_FROM_FILE
            else:
                flags = self.pbc.URDF_USE_INERTIA_FROM_FILE

            model_uid = load_fcn(
                self.model_xml,
                basePosition=self._init_base_pos,
                baseOrientation=self._init_base_ori,
                flags=flags,
                useFixedBase=self._is_fixed_base,
            )
        else:
            model_uid = load_fcn(
                self.model_xml,
            )

        self.logger.info('*'*40)
        self.logger.info('pbROBOT | Adding robot to scene')

        links = OrderedDict()
        joints = OrderedDict()
        ordered_joints = []
        pbactive_joint_idxs = []
        pbordered_joint_idxs = []
        pbcontrolled_joint_idxs = []

        self.logger.info('pbROBOT | At this moment, the world has %d bodies '
                         'in total.' % self.pbc.getNumBodies())

        # Some xml files (as mujoco have multiple bodies)
        if not isinstance(model_uid, tuple) or not isinstance(model_uid, list):
            model_uid = [model_uid]
        self._bodies_uids = model_uid

        self.logger.info('pbROBOT | Evaluating the %d bodies of file.')
        for i, b_idx in enumerate(model_uid):
            self.logger.info('')
            # print('Body', bb, '--', self.pbc.getBodyInfo(bb))
            self.logger.info('pbROBOT | file_body id %d has %d Joints'
                             % (b_idx, self.pbc.getNumJoints(b_idx)))

            if self.pbc.getNumJoints(b_idx) == 0:
                link_name, robot_name = self.pbc.getBodyInfo(b_idx)
                robot_name = robot_name.decode("utf8")
                link_name = link_name.decode("utf8")
                links[link_name] = Link(link_name, b_idx, -1, pbc=self.pbc)

                self.logger.info('pbROBOT | Body id: %d has 0 joints and '
                                 'part_name %s has been added to links dict.'
                                 % (b_idx, link_name))
                if len(model_uid) == 1:
                    self.logger.warning('pbROBOT | This is not a robotBody but '
                                        'creating robot_body and robot_uid '
                                        'anyway.')
                    self._base_link = links[link_name]
                    self.id = b_idx

            for jj in range(self.pbc.getNumJoints(b_idx)):
                self.logger.info('pbROBOT | Joint %d' % jj)
                # Get info from joints
                joint_info = self.pbc.getJointInfo(b_idx, jj)
                # joint_info = get_joint_info(self.pbc, b_idx, jj)
                joint_idx = joint_info[0]  # Pb joint idx
                joint_name = joint_info[1].decode("utf8")
                link_name = joint_info[12].decode("utf8")
                q_index = joint_info[3]
                if q_index > -1:
                    state_idx = len(pbactive_joint_idxs)
                    pbactive_joint_idxs.append(jj)
                else:
                    state_idx = 0

                self.logger.info('robot_joint:%s moves part_name:%s'
                                 % (joint_name, link_name))

                # All links are added to the dictionary
                links[link_name] = Link(link_name, b_idx, jj,
                                        pbc=self.pbc)
                self.logger.info('Adding part_name %s to links dict'
                                 % link_name)

                joint = Joint(joint_name, b_idx, joint_idx,
                              state_idx, pbc=self.pbc)

                # Compare to base of robot
                if self._base_link is None:
                    if link_name == self._base_name:
                        self.logger.info('part_name matches base_name! '
                                         'Using it as robot_body and uid')
                        self.id = b_idx
                        self._base_link = links[link_name]
                    elif jj == 0:  # i == 0 and j == 0:
                        self.logger.warning('Link name (%s) from first joint '
                                            'does not match base_name (%s)! '
                                            'Using link parent as robot_body '
                                            'and uid'
                                            % (link_name, self._base_name))
                        self.id = b_idx
                        links[self._base_name] = Link(self._base_name,
                                                      b_idx, -1,
                                                      pbc=self.pbc)
                        self._base_link = links[self._base_name]
                    else:
                        self.logger.error('Problem defining robot_body')
                        raise AttributeError('Error defining robot_body')
                    self.logger.info('The robot_body is now: %s'
                                     % self._base_link.body_name)

                if self._joint_names is None:
                    # Only consider active joints (with DoF)
                    if joint.u_idx > 0:
                        joints[joint_name] = joint
                        ordered_joints.append(joints[joint_name])
                        self.logger.info('joint %s | lower:%f upper:%f'
                                         % (joint_name,
                                            joints[joint_name].lower_limit,
                                            joints[joint_name].upper_limit))
                        pbordered_joint_idxs.append(jj)
                else:
                    if joint.u_idx > 0:
                        joints[joint_name] = joint

                        # Defining active joints using self._joint_names
                        if joint_name in self._joint_names:
                            # print('ACTIVATED', joint.joint_name, joint.u_idx)
                            ordered_joints.append(joints[joint_name])
                            pbordered_joint_idxs.append(jj)
                            self.logger.info('joint[%d] %s | lower:%f upper:%f | '
                                             'type: %d | q_idx: %d | u_idx:%d'
                                             % (joint_idx,
                                                joint_name,
                                                joints[joint_name].lower_limit,
                                                joints[joint_name].upper_limit,
                                                joints[joint_name].joint_type,
                                                joints[joint_name].q_idx,
                                                joints[joint_name].u_idx,
                                                ))
                        else:
                            # print('DISABLED', joint.joint_name, joint.u_idx)
                            joint.disable_motor()
                            self.logger.info('joint[%d] %s DISABLED.'
                                             % (joint_idx,
                                                joint_name,
                                                ))

        self._links = links
        self._joints = joints
        self._ordered_joints = ordered_joints
        self._pbactive_joint_idxs = pbactive_joint_idxs
        self._pbordered_joint_idxs = pbordered_joint_idxs
        self._pbcontrolled_joint_idxs = pbcontrolled_joint_idxs

    def reset(self):
        self._links = None
        self._joints = None
        self._base_link = None
        self._ordered_joints = None
        self._pbactive_joint_idxs = []
        self._pbordered_joint_idxs = []
        self._pbcontrolled_joint_idxs = []

        self.remove_sensor()
        self.remove_actuator()

        self.spawn_in_pb()

        # By default all the joints are controlled
        if self.controlled_joint_idxs is None:
            self.controlled_joint_idxs = list(range(self.n_ordered_joints))

        # Reorder joints according to name list
        self.reorder_joints_by_name(self._joint_names)

        # Reset function for specific robots
        self.reset_robot()

        self.initial_state = self.get_state()

        return self.get_state()

    # ####### #
    # Sensors #
    # ####### #

    def add_camera(self, body_name, dist=3, width=320, height=320,
                   update_state=True, name=None):
        camera = Camera(self._links[body_name], dist=dist,
                        width=width, height=height)
        if update_state:
            self.add_sensor(camera, name=name)
        return camera

    def add_encoder(self, position=True, velocity=True, torque=False,
                    update_state=True, name=None, only_controlled_joints=False):
        if only_controlled_joints:
            joints = self.controlled_joints
        else:
            joints = self._ordered_joints

        encoder = Encoder(joints, position=position,
                          velocity=velocity, torque=torque)
        if update_state:
            self.add_sensor(encoder, name=name)
        return encoder

    # ######### #
    # Actuators #
    # ######### #

    def add_joint_pos_controller(self, noise=None, update_actuation=True,
                                 name=None, position_limits=None,
                                 only_controlled_joints=False):

        if only_controlled_joints:
            act_dim = self.n_controlled_joints
        else:
            act_dim = self.n_ordered_joints

        if position_limits is None:
            position_limits = self.get_joint_limits(only_controlled_joints)

        pos_low = [lim[0] for lim in position_limits]
        pos_high = [lim[1] for lim in position_limits]

        controller = DirectJointController(
            lambda pos: self.set_joint_positions(pos, only_controlled_joints),
            act_dim,
            acts_min=pos_low,
            acts_max=pos_high,
        )

        if update_actuation:
            self.add_actuator(controller, name=name)
        return controller

    def add_joint_vel_controller(self, noise=None, update_actuation=True,
                                 name=None, velocity_limits=None,
                                 only_controlled_joints=False):

        if only_controlled_joints:
            act_dim = self.n_controlled_joints
        else:
            act_dim = self.n_ordered_joints

        if velocity_limits is None:
            velocity_limits = self.get_velocity_limits(only_controlled_joints)

        vel_low = [lim[0] for lim in velocity_limits]
        vel_high = [lim[1] for lim in velocity_limits]

        self.enable_torque_control(only_controlled_joints)

        controller = DirectJointController(
            lambda torques: self.set_joint_velocities(torques, only_controlled_joints),
            act_dim,
            acts_min=vel_low,
            acts_max=vel_high,
        )

        if update_actuation:
            self.add_actuator(controller, name=name)
        return controller

    def add_joint_torque_controller(self, noise=None,
                                    update_actuation=True, name=None,
                                    torque_limits=None,
                                    only_controlled_joints=False):

        if only_controlled_joints:
            act_dim = self.n_controlled_joints
        else:
            act_dim = self.n_ordered_joints

        if torque_limits is None:
            torque_limits = self.get_torque_limits(only_controlled_joints)

        torque_low = [lim[0] for lim in torque_limits]
        torque_high = [lim[1] for lim in torque_limits]

        controller = DirectJointController(
            lambda toqs: self.set_joint_torques(toqs, only_controlled_joints),
            act_dim,
            acts_min=torque_low,
            acts_max=torque_high,
        )

        if update_actuation:
            self.add_actuator(controller, name=name)
        return controller

    def add_task_torque_controller(self, noise=None,
                                   update_actuation=True, name=None,
                                   task_torque_limits=None,
                                   torque_limits=None,
                                   only_controlled_joints=False):

        if only_controlled_joints:
            act_dim = self.n_controlled_joints
        else:
            act_dim = self.n_ordered_joints

        if task_torque_limits is None:
            task_torque_limits = [(-20.0, 20.0) for _ in range(act_dim)]

        torque_low = [lim[0] for lim in task_torque_limits]
        torque_high = [lim[1] for lim in task_torque_limits]

        controller = DirectJointController(
            lambda toqs: self.set_joint_tasktorques(toqs, only_controlled_joints),
            act_dim,
            acts_min=torque_low,
            acts_max=torque_high,
        )

        if update_actuation:
            self.add_actuator(controller, name=name)
        return controller

    # ############# #
    # Links Methods #
    # ############# #
    @property
    def links(self):
        return self._links

    def get_link_poses(self, body_names):
        if not isinstance(body_names, list) and \
                not isinstance(body_names, tuple):
            body_names = [body_names]
        poses = []
        for body_name in body_names:
            poses.append(self._links[body_name].get_pose())
        return poses

    def get_link_velocities(self, body_names):
        if not isinstance(body_names, list) and \
                not isinstance(body_names, tuple):
            body_names = [body_names]
        speeds = []
        for body_name in body_names:
            speeds.append(self._links[body_name].speed())
        return speeds

    def get_link_contacts(self, body_names):
        if not isinstance(body_names, list) and \
                not isinstance(body_names, tuple):
            body_names = [body_names]

        contacts = []
        for body_name in body_names:
            body = self._links[body_name]
            contact_ids = set((x[2], x[4]) for x in body.contact_list())
            contacts.append(list(contact_ids))

        return contacts

    def get_link_masses(self, body_names):
        link_ids = [link.bodyPartIndex for link in self._links]
        link_masses = np.array([
            self.pbc.getDynamicsInfo(self.id, link_id)[0]
            for link_id in link_ids
        ])
        return link_masses

    # ############## #
    # Joints Methods #
    # ############## #

    def get_joint_limits(self, only_controlled_joints=False):
        if only_controlled_joints:
            joint_iterator = self._ordered_joints
        else:
            joint_iterator = self.controlled_joints

        return [(joint.lower_limit, joint.upper_limit)
                for joint in joint_iterator]

    def get_velocity_limits(self, only_controlled_joints=False):
        if only_controlled_joints:
            joint_iterator = self.controlled_joints
        else:
            joint_iterator = self._ordered_joints

        return [(-joint.joint_max_velocity, joint.joint_max_velocity)
                for joint in joint_iterator]

    def get_torque_limits(self, only_controlled_joints=False):
        if only_controlled_joints:
            joint_iterator = self.controlled_joints
        else:
            joint_iterator = self._ordered_joints

        return [(-joint.joint_max_force, joint.joint_max_force)
                for joint in joint_iterator]

    def get_joint_positions(self, only_ordered_joints=True):
        if only_ordered_joints:
            joint_iterator = self._ordered_joints
        else:
            joint_iterator = self._joints.values()

        return np.array(
            [joint.get_position() for joint in joint_iterator]
        )

    def get_joint_velocities(self, only_ordered_joints=True):
        if only_ordered_joints:
            joint_iterator = self._ordered_joints
        else:
            joint_iterator = self._joints.values()
        return np.array([joint.get_velocity() for joint in joint_iterator])

    def get_joint_torques(self, only_ordered_joints=True):
        if only_ordered_joints:
            joint_iterator = self._ordered_joints
        else:
            joint_iterator = self._joints.values()
        return np.array([joint.get_torque() for joint in joint_iterator])

    def get_joint_relative_positions(self, only_ordered_joints=True):
        if only_ordered_joints:
            joint_iterator = self._ordered_joints
        else:
            joint_iterator = self._joints.values()
        return np.array(
            [joint.get_relative_position() for joint in joint_iterator]
        )

    def get_controlled_joints_positions(self):
        joint_positions = self.get_joint_positions(only_ordered_joints=True)
        return joint_positions[self.controlled_joint_idxs]

    def get_controlled_joints_relative_positions(self):
        j_rel_pos = self.get_joint_relative_positions(only_ordered_joints=True)
        return j_rel_pos[self.controlled_joint_idxs]

    # ########### #
    # Full Robot  #
    # ########### #
    @property
    def dof(self):
        return self.n_controlled_joints

    @property
    def n_total_links(self):
        return self.n_total_joints

    @property
    def n_total_joints(self):
        return self.pbc.getNumJoints(self.id)

    @property
    def n_pb_joints(self):
        return len(self._pbactive_joint_idxs)

    @property
    def n_ordered_links(self):
        return self.n_ordered_joints

    @property
    def n_ordered_joints(self):
        return len(self._ordered_joints)

    @property
    def ordered_joints(self):
        return self._ordered_joints

    @property
    def ordered_joint_pb_idxs(self):
        return self._pbordered_joint_idxs

    @property
    def n_pbactive_links(self):
        return self.n_pb_joints

    @property
    def controlled_joint_idxs(self):
        return self._controlled_joint_idxs

    @controlled_joint_idxs.setter
    def controlled_joint_idxs(self, j_idxs):
        self._controlled_joint_idxs = j_idxs

    @property
    def controlled_joint_pb_idxs(self):
        return [self._pbordered_joint_idxs[pb_idx]
                for pb_idx in self.controlled_joint_idxs]

    @property
    def n_controlled_links(self):
        return self.n_controlled_joints

    @property
    def n_controlled_joints(self):
        return len(self.controlled_joints)

    @property
    def controlled_joints(self):
        return [self._ordered_joints[idx] for idx in self.controlled_joint_idxs]

    @property
    def total_links(self):
        return len(self._links)

    @property
    def total_joints(self):
        return len(self._ordered_joints)

    @property
    def total_controlled_joints(self):
        return len(self.controlled_joints)

    def get_total_joints(self):
        return self.total_joints

    @property
    def initial_configuration(self):
        return self._initial_configuration

    @initial_configuration.setter
    def initial_configuration(self, config):
        self._initial_configuration = config

    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state):
        self._initial_state = state

    @property
    def is_fixed_base(self):
        return self._is_fixed_base

    def set_state(self, des_state, only_controlled_joints=True):
        """ Set the robot state of the ordered joints.

        Set the robot's state (joint positions and velocities)

        Args:
            des_state (list or tuple or np.ndarray): Desired state
            only_controlled_joints (bool): Consider only the controlled joints

        Returns:
            None

        """
        if only_controlled_joints:
            joint_iterator = self.controlled_joints
        else:
            joint_iterator = self._ordered_joints

        total_joints = len(joint_iterator)

        if len(des_state) != total_joints*2:
            raise ValueError('State size does correspond to current robot '
                             '(%d != %d)' % (len(des_state),
                                             total_joints * 2))

        for jj, joint in enumerate(joint_iterator):
            joint.set_state(des_state[jj], des_state[total_joints+jj])

    def get_state(self, only_controlled_joints=False):
        """Get the state (joint positions and velocities) of the robot.

        Args:
            only_controlled_joints (bool): return only the state of the ordered
                 joints.

        Returns:

        """
        if only_controlled_joints:
            joints = self.controlled_joints
        else:
            joints = self._ordered_joints
        njoints = len(joints)
        state = np.zeros(njoints*2)
        for jj, joint in enumerate(joints):
            state[jj], state[jj+njoints] = joint.get_state()

        return state

    def set_joint_positions(self, positions, only_controlled_joints=True):
        """Apply given joint positions

        Args:
            positions (np.ndarray): Desired positions
            only_controlled_joints (bool): Set only controlled joints positions

        Returns:

        """
        if only_controlled_joints:
            joint_ids = self.controlled_joint_pb_idxs
        else:
            joint_ids = self.ordered_joint_pb_idxs

        self.pbc.setJointMotorControlArray(
            self.id,
            joint_ids,
            self.pbc.POSITION_CONTROL,
            targetPosition=positions,
        )

    def set_joint_velocities(self, velocities, only_controlled_joints=True,
                             max_forces=None):
        """Apply given velocities

        Args:
            velocities (np.ndarray):
            only_controlled_joints (bool):
            max_forces (np.ndarray or None):

        Returns:
            None

        """
        if only_controlled_joints:
            joint_ids = self.controlled_joint_pb_idxs
        else:
            joint_ids = self.ordered_joint_pb_idxs

        if max_forces is None:
            self.pbc.setJointMotorControlArray(
                self.id,
                joint_ids,
                self.pbc.VELOCITY_CONTROL,
                forces=velocities,
            )
        else:
            self.pbc.setJointMotorControlArray(
                self.id,
                joint_ids,
                self.pbc.VELOCITY_CONTROL,
                forces=velocities,
                max_forces=max_forces,
            )

    def set_joint_torques(self, torques, only_controlled_joints=True):
        """Apply given torques

        Args:
            torques (np.ndarray): Desired joint torques
            only_controlled_joints (bool):

        Returns:
            None

        """
        if only_controlled_joints:
            joint_ids = self.controlled_joint_pb_idxs
        else:
            joint_ids = self.ordered_joint_pb_idxs

        self.pbc.setJointMotorControlArray(
            self.id,
            joint_ids,
            self.pbc.TORQUE_CONTROL,
            forces=torques
        )

    def set_joint_tasktorques(self, tasktorques, only_controlled_joints=True):
        """Apply given 'task' torques.

        Args:
            tasktorques (np.ndarray):
            only_controlled_joints (bool):

        Returns:
            None

        """
        nl_torques = self.get_nonlinear_effect_torques(only_controlled_joints)

        self.set_joint_torques(tasktorques + nl_torques)

    def enable_torque_control(self, only_controlled_joints=True):
        if only_controlled_joints:
            joint_ids = self.controlled_joint_pb_idxs
        else:
            joint_ids = self.ordered_joint_pb_idxs

        # disable motor in order to use direct torque control
        self.pbc.setJointMotorControlArray(
            self.id,
            joint_ids,
            self.pbc.VELOCITY_CONTROL,
            forces=[0.0]*len(joint_ids)
        )

    # ########## #
    # Kinematics #
    # ########## #

    def compute_jacobian(self, link, joint_pos, joint_vel=None,
                         joint_acc=None, link_point=None):
        """Compute Jacobian

        Args:
            link (str or int):
            joint_pos (np.ndarray or None):
            joint_vel (np.ndarray or None):
            joint_acc (np.ndarray or None):
            link_point (np.ndarray or None):

        Returns:
            np.ndarray: (Geometric) Jacobian

        """

        if isinstance(link, str) and link not in self._links:
            raise ValueError("Link name %s is not a robot body_part"
                             % link)

        if link_point is None:
            link_point = [0., 0., 0.]

        # pos = [0]*self.n_pb_joints
        # vel = [0]*self.n_pb_joints
        # acc = [0]*self.n_pb_joints
        full_state = self.pbc.getJointStates(self.id, self._pbactive_joint_idxs)
        pos = [ss[0] for ss in full_state]
        vel = [ss[1] for ss in full_state]
        acc = [0]*self.n_pb_joints

        for jj in range(len(self._ordered_joints)):
            pb_joint_id = self._ordered_joints[jj].uIndex - 6
            pos[pb_joint_id] = joint_pos[jj]
            if joint_vel is not None:
                vel[pb_joint_id] = joint_vel[jj]
            if joint_acc is not None:
                acc[pb_joint_id] = joint_acc[jj]

        jac_t, jac_r = self.pbc.calculateJacobian(
            self.id, self._links[link].bodyPartIndex, link_point,
            pos, vel, acc
        )

        total_joints = len(self._ordered_joints)
        if not self._is_fixed_base:
            total_joints += 6

        jac_t = np.array(jac_t)
        jac_r = np.array(jac_r)
        jac = np.concatenate((jac_t, jac_r), axis=0)

        jacobian = np.zeros((6, total_joints))
        if not self._is_fixed_base:
            init_jac_idx = 6
        else:
            init_jac_idx = 0

        jacobian[:, :init_jac_idx] = jac[:, :init_jac_idx]

        for jj, joint in enumerate(self._ordered_joints):
            pb_joint_id = self._ordered_joints[jj].uIndex - 6
            jacobian[:, jj] = jac[:, pb_joint_id]

        return jacobian

    def get_jacobian(self, link, joint_pos, joint_vel=None,
                     joint_acc=None, link_point=None,
                     only_controlled_joints=True):

        full_joint_pos = np.zeros(self.total_joints)
        full_joint_vel = np.zeros(self.total_joints)
        full_joint_acc = np.zeros(self.total_joints)

        if only_controlled_joints:
            full_joint_pos[self.controlled_joint_idxs] = joint_pos
            if joint_vel is not None:
                full_joint_vel[self.controlled_joint_idxs] = joint_vel

            if joint_acc is not None:
                full_joint_acc[self.controlled_joint_idxs] = joint_acc

        full_jacobian = self.compute_jacobian(link=link,
                                              joint_pos=full_joint_pos,
                                              joint_vel=full_joint_vel,
                                              joint_acc=full_joint_acc,
                                              link_point=link_point)
        if only_controlled_joints:
            jacobian = full_jacobian[:, self.controlled_joint_idxs]
        else:
            jacobian = full_jacobian

        return jacobian

    def calculate_inverse_dynamics(self, jacc, jpos=None, jvel=None):
        """Compute torques in inverse dynamics

        Args:
            jacc (np.ndarray): (Ordered) Joint Accelerations
            jpos (np.ndarray or None): (Ordered) Joint Positions
            jvel (np.ndarray or None):  (Ordered) Joint Velocities

        Returns:
            np.ndarray: (Ordered) Joint Torques

        """
        # NOTE: Only pybullet joints are considered. And not the Base DoF.
        full_state = self.pbc.getJointStates(self.id, self._pbactive_joint_idxs)
        pos = [ss[0] for ss in full_state]
        vel = [ss[1] for ss in full_state]
        acc = [0]*self.n_pb_joints

        for jj in range(len(self._ordered_joints)):
            pb_joint_id = self._ordered_joints[jj].dof_idx
            acc[pb_joint_id] = jacc[jj]
            if jpos is not None:
                pos[pb_joint_id] = jpos[jj]
            if jvel is not None:
                vel[pb_joint_id] = jvel[jj]

        all_torques = self.pbc.calculateInverseDynamics(self.id,
                                                        pos, vel, acc)

        jtorq = np.zeros_like(jacc)
        for jj in range(len(self._ordered_joints)):
            pb_joint_id = self._ordered_joints[jj].dof_idx
            jtorq[jj] = all_torques[pb_joint_id]

        return jtorq

    def calculate_nonlinear_effects(self):
        """Compute nonlinear effects in (ordered) joints.

        Returns:
            np.ndarray: (Ordered) nonlinear effects torques

        """
        joint_pos = None
        joint_vel = None
        joint_acc = np.zeros(self.n_ordered_joints)

        return self.calculate_inverse_dynamics(
            jacc=joint_acc,
            jpos=joint_pos,
            jvel=joint_vel
        )

    def get_nonlinear_effect_torques(self, only_controlled_joints=True):
        """Get the nonlinear effects

        Args:
            only_controlled_joints (bool):

        Returns:
            np.ndarray: (Ordered) nonlinear effects torques

        """
        torque_ordered_joints = self.calculate_nonlinear_effects()
        if only_controlled_joints:
            return torque_ordered_joints[self.controlled_joint_idxs]
        else:
            return torque_ordered_joints

    def get_body_parts_names(self):
        return list(self._links.keys())

    def get_joints_names(self):
        return [joint.joint_name for joint in self._ordered_joints]


    # ##### #
    # Utils #
    # ##### #

    def get_image(self, camera_name='RGBCamera'):
        return self._sensors[camera_name].read()

    def reorder_joints_by_name(self, joint_names):
        """Change the order of the robot's joints according to a list.

        Args:
            joint_names (list of str): Joint Names

        Returns:
            None

        """
        if joint_names is not None:
            # Reorder the Joints
            if len(self._ordered_joints) != len(joint_names):
                raise AttributeError('Requested joint name list (%d) is '
                                     'different than current joints %d'
                                     % (len(joint_names),
                                        len(self._ordered_joints)))

            self._pbordered_joint_idxs = []
            for jj, j_name in enumerate(joint_names):
                self._ordered_joints[jj] = self._joints[j_name]
                self._pbordered_joint_idxs.append(self._joints[j_name].joint_index)

    def get_body_pose(self, body_name):
        """Get the pose of a robot body(link).

        The body pose includes the position and orientation represented in
        quaternions.

        Args:
            body_name (str):

        Returns:
            np.ndarray: Body pose

        """
        return self._links[body_name].get_pose()

    def get_link_speed(self, body_name):
        return self._links[body_name].speed()

    def get_base_speed(self):
        return self.get_link_speed(self._base_name)
