from __future__ import print_function
import numpy as np


class PoseHelper(object):  # dummy class to comply to original interface
    def __init__(self, body_part, pbc):
        self.pbc = pbc

        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return self.pbc.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()


class Link(object):
    def __init__(self, body_name, body_uid, bodyPartIndex, pbc):
        """

        :param body_name:
        :param bodies:
        :param bodyIndex:
        :param bodyPartIndex:
        """
        self.pbc = pbc
        self.multibody_uid = body_uid

        self.body_name = body_name
        self.bodyPartIndex = bodyPartIndex

        self.initialPosition = self.current_position()
        self.initialOrientation = self.current_orientation()
        self.bp_pose = PoseHelper(self, self.pbc)

    @property
    def id(self):
        return self.bodyPartIndex

    @id.setter
    def id(self, bullet_body_id):
        self.bodyPartIndex = bullet_body_id

    @property
    def multibody_uid(self):
        return self._multibody_uid

    @multibody_uid.setter
    def multibody_uid(self, pb_uid):
        self._multibody_uid = pb_uid

    @property
    def name(self):
        return self.body_name

    # A method you will most probably need a lot to get pose and orientation
    def state_fields_of_pose_of(self, body_id, link_id=-1):
        if link_id == -1:
            (x, y, z), (a, b, c, d) = \
                self.pbc.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = \
                self.pbc.getLinkState(body_id, link_id)
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.multibody_uid,
                                            self.bodyPartIndex)

    def get_position(self):
        return self.get_pose()[:3]

    def current_position(self):
        return self.get_position()

    def get_orientation(self):
        return self.get_pose()[3:]

    def current_orientation(self):
        return self.get_orientation()

    def pose(self):
        return self.bp_pose

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), (wx, wy, wz) = self.pbc.getBaseVelocity(
                self.multibody_uid
            )
        else:
            (x,y,z), (a,b,c,d), _,_,_,_, (vx, vy, vz), (wx, wy, wz) = \
                self.pbc.getLinkState(self.multibody_uid, self.bodyPartIndex,
                                      computeLinkVelocity=1)
        return np.array([vx, vy, vz, wx, wy, wz])

    def get_velocity(self):
        return self.speed()

    def reset_position(self, position):
        self.pbc.resetBasePositionAndOrientation(self.multibody_uid,
                                                 position,
                                                 self.get_orientation())

    def reset_orientation(self, orientation):
        self.pbc.resetBasePositionAndOrientation(self.multibody_uid,
                                                 self.get_position(),
                                                 orientation)

    def reset_pose(self, position, orientation):
        self.pbc.resetBasePositionAndOrientation(self.multibody_uid,
                                                 position,
                                                 orientation)

    def contact_list(self):
        return self.pbc.getContactPoints(self.multibody_uid, -1,
                                         self.bodyPartIndex, -1)


class Joint(object):
    def __init__(self, joint_name, body_uid, joint_index, state_idx, pbc):
        """

        :param joint_name:
        :param body_uid:
        :param joint_index:
        """
        self.pbc = pbc
        self.multibody_uid = body_uid

        self.joint_index = joint_index
        self.joint_name = joint_name
        self.dof_idx = state_idx

        # Get additional info
        joint_info = self.pbc.getJointInfo(self.multibody_uid, self.joint_index)
        self.joint_type = joint_info[2]
        self.q_idx = joint_info[3]
        self.u_idx = joint_info[4]
        self.lower_limit = joint_info[8]
        self.upper_limit = joint_info[9]
        self.joint_max_force = joint_info[10]
        self.joint_max_velocity = joint_info[11]

    @property
    def multibody_uid(self):
        return self._multibody_uid

    @multibody_uid.setter
    def multibody_uid(self, pb_uid):
        self._multibody_uid = pb_uid

    def get_state(self):
        x, vx, _, _ = self.pbc.getJointState(self.multibody_uid,
                                             self.joint_index)
        return x, vx

    def get_position(self):
        return self.pbc.getJointState(self.multibody_uid, self.joint_index)[0]

    def current_position(self):  # just some synonym method
        return self.get_position()

    def get_velocity(self):
        return self.pbc.getJointState(self.multibody_uid, self.joint_index)[1]

    def get_torque(self):
        return self.pbc.getJointState(self.multibody_uid, self.joint_index)[3]

    def get_motor_torque(self):  # just some synonym method
        return self.get_torque()

    def get_relative_position(self):
        pos = self.get_position()
        pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
        return 2 * (pos - pos_mid) / (self.upper_limit - self.lower_limit)

    def get_reaction_forces(self):
        return self.pbc.getJointState(self.multibody_uid, self.joint_index)[2]

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lower_limit + self.upper_limit)
        return (
            2 * (pos - pos_mid) / (self.upper_limit - self.lower_limit),
            0.1 * vel
        )

    def set_state(self, x, vx):
        self.pbc.resetJointState(self.multibody_uid, self.joint_index, x, vx)

    def set_position(self, position):
        self.pbc.setJointMotorControl2(self.multibody_uid, self.joint_index,
                                       self.pbc.POSITION_CONTROL,
                                       targetPosition=position)

    def set_velocity(self, velocity):
        self.pbc.setJointMotorControl2(self.multibody_uid, self.joint_index,
                                       self.pbc.VELOCITY_CONTROL,
                                       targetVelocity=velocity)

    def set_torque(self, torque):
        # Disable the Motors for Torque Control
        # p.setJointMotorControl2(self.multibody_uid,
        #                         self.joint_index, p.VELOCITY_CONTROL,
        #                         targetVelocity=0, force=0)
        self.pbc.setJointMotorControl2(self.multibody_uid,
                                       self.joint_index,
                                       self.pbc.TORQUE_CONTROL,
                                       force=torque)
        # p.setJointMotorControl2(body_index=self.multibody_uid,
        #                         joint_index=self.joint_index,
        #                         controlMode=p.TORQUE_CONTROL, force=torque)
        #                         # positionGain=0.0, velocityGain=0.0)

    def set_motor_torque(self, torque):  # just some synonym method
        self.set_torque(torque)

    def reset_current_position(self, position, velocity):  # just some synonym method
        self.reset_position(position, velocity)

    def reset_position(self, position, velocity):
        self.pbc.resetJointState(self.multibody_uid, self.joint_index,
                                 targetValue=position, targetVelocity=velocity)
        # self.disable_motor()

    def disable_motor(self):
        self.pbc.setJointMotorControl2(
            self.multibody_uid,
            self.joint_index,
            controlMode=self.pbc.VELOCITY_CONTROL,
            force=0.0,
            # # targetVelocity=0,
            # positionGain=0.0,
            # velocityGain=0.0,
        )

    def lock_motor(self, tgt_pos=0, tgt_vel=0):
        self.pbc.setJointMotorControl2(
            self.multibody_uid,
            self.joint_index,
            # controlMode=p.VELOCITY_CONTROL, force=0
            controlMode=self.pbc.POSITION_CONTROL,
            # controlMode=p.VELOCITY_CONTROL,
            targetPosition=tgt_pos,
            targetVelocity=tgt_vel,
            # positionGain=0.1, velocityGain=0.1
        )


def get_joint_info(pbc, multibody_uid, joint_idx):
    """Dictionary with info about multi-body joint.

    Args:
        pbc: pybullet client
        multibody_uid: Pybullet unique ID of multibody
        joint_idx: Pybullet joint index in multibody

    Returns:
        dict: Python dictionary with the following keys:
            0 jointIndex (int): Pybullet joint index (same than joint_idx)
            1 jointName (str): Name of the joint, as specified in xml file
            2 jointType (int): type of the joint, this also implies the number
                of position and velocity variables. JOINT_REVOLUTE,
                JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED.
            3 qIndex (int): the first position index in the positional state
                variables for this body
            4 uIndex (int): the first velocity index in the velocity state
                variables for this body
            5 flags (int): reserved
            6 jointDamping (float): the joint damping value, as specified in the
                URDF file
            7 jointFriction (float): the joint friction value, as specified in
                the URDF file
            8 jointLowerLimit (float): Positional lower limit for slider and
                revolute (hinge) joints.
            9 jointUpperLimit (float): Positional upper limit for slider and
                revolute joints. Values ignored in case upper_lim <lower_lim.
            10 jointMaxForce (float): Maximum force specified in URDF
                (possibly other file formats) Note that this value is not
                automatically used. You can use maxForce in 'setJointMotorControl2'.
            11 jointMaxVelocity (float): Maximum velocity specified in URDF.
                Note that the maximum velocity is not used in actual motor
                control commands at the moment.
            12 linkName (str): the name of the link, as specified in the URDF
                (or SDF etc.) file
            13 jointAxis (vec3): joint axis in local frame (ignored for JOINT_FIXED)
            14 parentFramePos (vec3): joint position in parent frame
            15 parentFrameOrn (vec3): joint orientation in parent frame
            16 parentIndex (int): parent link index, -1 for base

    """
    joint_info = pbc.getJointInfo(multibody_uid, joint_idx)
    joint_info_dict = {
        'joint_idx': joint_info[0],  # Pb joint idx
        'joint_name': joint_info[1].decode("utf8"),
        'joint_type': joint_info[2],
        'q_idx': joint_info[3],
        'u_idx': joint_info[4],
        'damping': joint_info[6],
        'friction': joint_info[7],
        'lower_limit': joint_info[8],
        'upper_limit': joint_info[9],
        'joint_max_force': joint_info[10],
        'joint_max_velocity': joint_info[11],
        'link_name': joint_info[12].decode("utf8"),
        'parent_pos': np.array(joint_info[14]),
        'parent_ori': np.array(joint_info[15]),
        'parent_idx': joint_info[16],
    }

    return joint_info_dict
