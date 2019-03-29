import numpy as np
from robolearn_envs.core.robolearn_actuators import RoboLearnActuator


class DirectJointController(RoboLearnActuator):
    def __init__(self, actuations_fcn, act_dim, acts_min=None, acts_max=None):

        self._state = np.zeros(act_dim)

        self._actuations_fcn = actuations_fcn

        if acts_min is None:
            acts_min = np.ones(act_dim) * -np.inf

        if acts_max is None:
            acts_max = np.ones(act_dim) * np.inf

        self._acts_min = acts_min
        self._acts_max = acts_max

        self._limits = [(a_min, a_max)
                        for a_min, a_max in zip(acts_min, acts_max)]

    def actuate(self, action):
        self._actuations_fcn(
            np.clip(action, self._acts_min, self._acts_max)
        )


class JointPositionController(RoboLearnActuator):
    def __init__(self, joint_list, noise=None, joint_limits=None):
        self.joints = joint_list
        self._n_joints = len(joint_list)
        self._noise = noise

        self._state = np.zeros(self._n_joints)

        if joint_limits is None:
            self._limits = \
                [(joint.lower_limit, joint.upper_limit)
                 for joint in self._joint_list]
        else:
            self._limits = joint_limits

    def actuate(self, action):
        for jj, joint in enumerate(self._joint_list):
            joint.set_position(
                np.clip(action[jj], self._limits[jj][0], self._limits[jj][1]
                        ).item()
            )

    @property
    def joints(self):
        return self._joint_list

    @joints.setter
    def joints(self, joint_list):
        self._joint_list = joint_list


class JointVelocityController(RoboLearnActuator):
    def __init__(self, joint_list, noise=None, velocity_limits=None):
        self.joints = joint_list
        self._n_joints = len(joint_list)
        self._noise = noise

        self._state = np.zeros(self._n_joints)

        if velocity_limits is None:
            self._limits = \
                [(-joint.joint_max_velocity, joint.joint_max_velocity)
                 for joint in self._joint_list]
        else:
            self._limits = velocity_limits

    def actuate(self, action):
        for jj, joint in enumerate(self._joint_list):
            joint.set_velocity(
                np.clip(action[jj],
                        self._limits[jj][0],
                        self._limits[jj][1]
                        ).item()

            )

    @property
    def joints(self):
        return self._joint_list

    @joints.setter
    def joints(self, joint_list):
        self._joint_list = joint_list


class JointTorqueController(RoboLearnActuator):
    def __init__(self, joint_list, noise=None, torque_limits=None):
        self.joints = joint_list
        self._n_joints = len(joint_list)
        self._noise = noise

        self._state = np.zeros(self._n_joints)

        if torque_limits is None:
            self._limits = \
                [(-joint.joint_max_force, joint.joint_max_force)
                 for joint in self._joint_list]
        else:
            self._limits = torque_limits

        # Deactivate first
        for jj, joint in enumerate(self._joint_list):
            joint.disable_motor()
            # joint.lock_motor()

    def actuate(self, action):
        bodies = self._joint_list[0].pbc.getNumBodies()
        for jj, joint in enumerate(self._joint_list):
            # joint.disable_motor()
            # print(joint.joint_name)
            joint.set_motor_torque(
                np.clip(action[jj],
                        self._limits[jj][0],
                        self._limits[jj][1]
                        ).item()

            )

    @property
    def joints(self):
        return self._joint_list

    @joints.setter
    def joints(self, joint_list):
        self._joint_list = joint_list


class JointTaskTorqueController(RoboLearnActuator):
    def __init__(self, joint_list, nonlinear_effect_fcn, noise=None,
                 task_torque_limits=None, torque_limits=None, ):
        self.joints = joint_list
        self._nonlinear_effect_fcn = nonlinear_effect_fcn

        self._n_joints = len(joint_list)
        self._noise = noise

        self._state = np.zeros(self._n_joints)

        if torque_limits is None:
            self._torque_limits = \
                [(-joint.joint_max_force, joint.joint_max_force)
                 for joint in self._joint_list]
        else:
            self._torque_limits = torque_limits

        if task_torque_limits is None:
            self._limits = \
                [(-np.inf, np.inf) for _ in self._joint_list]
        else:
            self._limits = task_torque_limits

        # Deactivate first
        for jj, joint in enumerate(self._joint_list):
            joint.disable_motor()

    def actuate(self, action):
        nonlinear_effect_torques = self._nonlinear_effect_fcn()

        for jj, joint in enumerate(self._joint_list):
            joint.set_motor_torque(
                np.clip(np.clip(action[jj],
                                self._limits[jj][0],
                                self._limits[jj][1],
                                )
                        + nonlinear_effect_torques[jj],
                        self._torque_limits[jj][0],
                        self._torque_limits[jj][1]
                        ).item()
            )

    @property
    def joints(self):
        return self._joint_list

    @joints.setter
    def joints(self, joint_list):
        self._joint_list = joint_list


