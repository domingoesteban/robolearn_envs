from __future__ import print_function
import os
import numpy as np
from robolearn_envs.pybullet.core.bullet_robot import BulletRobot
from robolearn_envs.robot_models.planarmanipulator import model_path


class PlanarManipulator4Dof(BulletRobot):
    def __init__(self,
                 robot_name='base_link',
                 init_config=None,
                 init_pos=None,
                 init_ori=None,
                 self_collision=False,
                 control_mode='joint_torque',
                 robot_model=None,
                 ):

        # Robot URDF file
        if robot_model is None:
            xml_file = os.path.join(
                model_path,
                'pusher4dof.urdf'
            )
        else:
            raise NotImplementedError
            # xml_file = os.path.join(
            #     model_path,
            #     'pusher_pusher3dof.urdf'
            # )

        joint_names = ['joint0', 'joint1', 'joint2', 'joint3']

        super(PlanarManipulator4Dof, self).__init__(
            model_file=xml_file,
            base_name=robot_name,
            init_pos=init_pos,
            init_ori=init_ori,
            joint_names=joint_names,
            self_collision=self_collision,
            fixed_base=True,
        )

        if control_mode not in [
            'joint_position',
            'joint_velocity',
            'joint_torque',
        ]:
            raise ValueError('Wrong control mode %s' % control_mode)
        self._control_mode = control_mode

        # Initial / default values
        if init_config is not None:
            self.initial_configuration = init_config
        else:
            self.initial_configuration = \
                [np.deg2rad(-100), np.deg2rad(45), np.deg2rad(20), np.deg2rad(5)]

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
        self.add_encoder(name='Encoder', only_controlled_joints=False)

        # Controllers
        if self._control_mode.lower() == 'joint_position':
            self.add_joint_pos_controller(name='JointPositionController',
                                          only_controlled_joints=True,
                                          )
        elif self._control_mode.lower() == 'joint_velocity':
            self.add_joint_vel_controller(name='JointVelocityController',
                                          only_controlled_joints=True,
                                          )
        elif self._control_mode.lower() == 'joint_torque':
            # Deactivate active joints for Torque Control
            for jj, joint in enumerate(self.controlled_joints):
                joint.disable_motor()
            # EFFORT_LIMITS = [5, 5, 5, 5]
            EFFORT_LIMITS = [20, 20, 20, 20]
            torque_limits = [(-limit, limit) for limit in EFFORT_LIMITS]
            self.add_joint_torque_controller(name='JointTorqueController',
                                             torque_limits=torque_limits,
                                             only_controlled_joints=False,
                                             )
        else:
            raise ValueError('No Control type selected')

        # Color
        color_list = [[0.0, 0.4, 0.6, 1]
                      for _ in range(self.total_links)]
        color_list[0] = [0.9, 0.4, 0.6, 1]
        color_list[-1] = [0.0, 0.8, 0.6, 1]
        self.set_body_colors(color_list)
