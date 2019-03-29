from __future__ import print_function
import os
import numpy as np
from robolearn_envs.pybullet.core.bullet_robot import BulletRobot
from robolearn_envs.robot_models.coman.coman_params import JOINT_NAMES
from robolearn_envs.robot_models.coman.coman_params import INIT_CONFIG
from robolearn_envs.robot_models.coman.coman_params import INIT_CONFIG_N_POSE
from robolearn_envs.robot_models.coman.coman_params import BODY_PARTS
from robolearn_envs.robot_models.coman import model_path
import warnings


class Coman(BulletRobot):
    def __init__(self,
                 robot_name='base_link',
                 init_config=None,
                 init_pos=None,
                 init_ori=None,
                 self_collision=True,
                 control_mode='joint_position',
                 active_joints='WB',
                 init_pose_name='loco',
                 fixed_base=False,
                 ):

        # Robot URDF file
        xml_file = os.path.join(
            model_path,
            'coman.urdf'
        )

        if self_collision:
            self_collision = False
            warnings.warn("Self-collision is deactivated in COMAN!")

        if active_joints not in BODY_PARTS:
            raise ValueError("Wrong active_joint name '%s'. "
                             "Available options: %s."
                             % (active_joints, list(BODY_PARTS.keys())))

        super(Coman, self).__init__(
            model_file=xml_file,
            base_name=robot_name,
            init_pos=init_pos,
            init_ori=init_ori,
            joint_names=JOINT_NAMES,
            controlled_joint_idxs=BODY_PARTS[active_joints],
            self_collision=self_collision,
            fixed_base=fixed_base,
        )

        if control_mode not in [
            'joint_position',
            'joint_velocity',
            'joint_torque',
            'joint_tasktorque'
        ]:
            raise ValueError('Wrong control mode %s' % control_mode)
        self._control_mode = control_mode

        self.init_pose_name = init_pose_name

        if init_config is None:
            self.initial_configuration = INIT_CONFIG_N_POSE
        else:
            self.initial_configuration = init_config

    def reset_robot(self, state=None):
        # Replace with requested state
        if state is not None:
            self._initial_state = state

        initial_state = np.zeros(self.total_joints*2)

        initial_state[:self.total_joints] = self.initial_configuration

        for jj, joint in enumerate(self._ordered_joints):
            joint.reset_position(initial_state[jj],
                                 initial_state[self.total_joints+jj])

        # Sensors
        self.add_encoder(name='Encoder', only_controlled_joints=True)
        # self.add_camera('HeadHokuyo', name='CameraHokuyo')

        # Controllers
        if self._control_mode.lower() == 'joint_position':
            self.add_joint_pos_controller(name='PositionJointController',
                                          only_controlled_joints=True,
                                          )
        elif self._control_mode.lower() == 'joint_velocity':
            self.add_joint_vel_controller(name='VelocityJointController',
                                          only_controlled_joints=True,
                                          )
        elif self._control_mode.lower() == 'joint_torque':
            # Deactivate active joints for Torque Control
            for joint in self.controlled_joints:
                joint.disable_motor()
            self.add_joint_torque_controller(name='TorqueJointController',
                                             only_controlled_joints=True,
                                             )
        elif self._control_mode.lower() == 'joint_tasktorque':
            # Deactivate active joints for Torque Control
            for joint in self.controlled_joints:
                joint.disable_motor()
            limits = [(-5.0, 5.0) for _ in range(self.n_controlled_joints)]
            self.add_task_torque_controller(name='JointTaskTorqueController',
                                            only_controlled_joints=True,
                                            task_torque_limits=limits,
                                            torque_limits=None,
                                            )
        else:
            raise ValueError('No Control type selected')

        # Deactivate no active joints
        for jj, joint in enumerate(self._ordered_joints):
            if jj not in self.controlled_joint_idxs:
                joint.lock_motor(initial_state[jj], 0)
                # joint.lock_motor(INIT_CONFIG[jj], 0)
