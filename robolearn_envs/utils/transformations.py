import numpy as np
from transforms3d.quaternions import mat2quat
from transforms3d.quaternions import quat2mat
from transforms3d.euler import euler2mat
from pybullet import getEulerFromQuaternion


def multiply_quat(quat1, quat2):
    w = quat1[3]*quat2[3] - quat1[0]*quat2[0] - quat1[1]*quat2[1] - quat1[2]*quat2[2]
    x = quat1[3]*quat2[0] + quat1[0]*quat2[3] + quat1[1]*quat2[2] - quat1[2]*quat2[1]
    y = quat1[3]*quat2[1] + quat1[1]*quat2[3] + quat1[2]*quat2[0] - quat1[0]*quat2[2]
    z = quat1[3]*quat2[2] + quat1[2]*quat2[3] + quat1[0]*quat2[1] - quat1[1]*quat2[0]
    return [x, y, z, w]


def quaternion_inner(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
    return q1.dot(q2)
    #return quaternion_multiply(q1, q2)


def quaternion_multiply(q1, q2):
    q1 = np.array(q1)
    q2 = np.array(q2)
    w1 = q1[-1]
    w2 = q2[-1]
    v1 = q1[:3]
    v2 = q2[:3]
    return np.r_[w1*v2+w2*v1 + np.cross(v1, v2), w1*w2-v1.dot(v2)]


def inv_quat(quat):
    return np.array([-quat[0], -quat[1], -quat[2], quat[3]])


def quat_vector_cross(quat_vec):
    return np.array([[0, -quat_vec[2], quat_vec[1]],
                     [quat_vec[2], 0, -quat_vec[0]],
                     [-quat_vec[1], quat_vec[0], 0]])


def quat_difference(final_quat, init_quat, prev_diff=None):
    """
    It calculates difference = final_quat - init_quat
    Args:
        final_quat: (x,y,z,w)
        init_quat: (x,y,z,w)
        prev_diff: (dR, dP, dY)

    Returns:

    """
    difference = init_quat[3]*final_quat[:3] - final_quat[3]*init_quat[:3] - np.cross(final_quat[:3], init_quat[:3])  # Previous
    # difference = final_quat[3]*init_quat[:3] - init_quat[3]*final_quat[:3] + quat_vector_cross(final_quat[:3]).dot(init_quat[:3])  # From Nakanishi

    if prev_diff is not None:
        if difference.dot(prev_diff) < 0:
            difference *= -1

    return difference


def homogeneous_matrix(rot=np.identity(3), pos=np.zeros(3)):
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rot[:3, :3]
    transform_matrix[:3, -1] = pos[:3]
    return transform_matrix


def compute_cartesian_error(des, current, rotation_rep='quat', first='pos',
                            prev_ori_diff=None):
    """Cartesian error between two poses
    Compute the cartesian error between two poses: error = des-current
    Args:
        des:Desired cartesian pose (orientation+position)
        current:Actual cartesian pose (orientation+position)
        rotation_rep:Orientation units
        first:'ori' or 'pos'
        prev_ori_diff:

    Returns:
        np.ndarray: Cartesian error

    """
    if first == 'pos':
        position_error = des[:3] - current[:3]
    else:
        position_error = des[-3:] - current[-3:]

    if rotation_rep == 'quat':
        if first == 'pos':
            orientation_error = quat_difference(des[-4:], current[-4:],
                                                prev_diff=prev_ori_diff)
        else:
            orientation_error = quat_difference(des[:4], current[:4],
                                                prev_diff=prev_ori_diff)
    elif rotation_rep == 'rpy':
        if first == 'pos':
            orientation_error = des[-3:] - current[-3:]
        else:
            orientation_error = des[:3] - current[:3]
    else:
        raise NotImplementedError("Only quaternion has been implemented")

    if first == 'pos':
        return np.concatenate((position_error, orientation_error))
    else:
        return np.concatenate((orientation_error, position_error))


def create_quat_pose(pos_x=0, pos_y=0, pos_z=0, rot_roll=0, rot_pitch=0,
                     rot_yaw=0, first='pos', order='xyzw'):
    """Create a quaternion pose
    Rotation assuming first yaw, then pitch, and then yaw.
    Args:
        pos_x:
        pos_y:
        pos_z:
        rot_roll:
        rot_pitch:
        rot_yaw:
        first: 'pos' or 'ori'
        order: 'xyzw' or 'wxyz'

    Returns:
        np.ndarray: Resulted quaternion pose

    """
    pose = np.zeros(7)
    quat = create_quat(rot_roll=rot_roll, rot_pitch=rot_pitch, rot_yaw=rot_yaw,
                       order=order)
    if first == 'ori':
        pose[:4] = quat
        pose[4] = pos_x
        pose[5] = pos_y
        pose[6] = pos_z
    else:
        pose[0] = pos_x
        pose[1] = pos_y
        pose[2] = pos_z
        pose[3:] = quat

    return pose


def create_quat(rot_roll=0, rot_pitch=0, rot_yaw=0, order='xyzw'):
    # pose[:4] = tf.transformations.quaternion_from_matrix(tf.transformations.euler_matrix(rot_roll, rot_pitch, rot_yaw))
    # rot = x_rotation(rot_roll)*y_rotation(rot_pitch)*z_rotation(rot_yaw)
    rot = euler2mat(rot_roll, rot_pitch, rot_yaw, 'sxyz')
    quat = mat2quat(np.array(rot).astype(float))
    if order == 'wxyz':
        return quat
    elif order == 'xyzw':
        return np.take(quat, [1, 2, 3, 0])
    else:
        raise AttributeError('Wrong order option')


def euler_to_quat(euler, order='xyzw'):
    return create_quat(rot_roll=euler[0], rot_pitch=euler[1], rot_yaw=euler[2],
                       order=order)


def euler_from_quat(quat, order='xyzw'):
    """

    Args:
        quat:
        order (string): 'xyzw' or 'wxyz'

    Returns:
        np.ndarray: [rot_x, rot_y, rot_z]

    """
    if order == 'xyzw':
        pass
    elif order == 'wxyz':
        quat = np.take(quat, [3, 0, 1, 2])
    else:
        raise AttributeError("Wrong order option %s" % order)

    return getEulerFromQuaternion(quat)

    # return mat2euler(np.array(quat2mat(quat)).astype(float))


def pose_transform(frame_pose, relative_pose):
    """
    pose is pos + quat: (x, y, z, w)
    Args:
        frame_pose:
        relative_pose:

    Returns:

    """
    frame_pose = np.array(frame_pose)
    relative_pose = np.array(relative_pose)

    frame_matrix = np.eye(4)
    frame_matrix[:3, :3] = quat2mat(frame_pose[[6, 3, 4, 5]])
    frame_matrix[:3, -1] = frame_pose[:3]

    relative_matrix = np.eye(4)
    relative_matrix[:3, :3] = quat2mat(relative_pose[[6, 3, 4, 5]])
    relative_matrix[:3, -1] = relative_pose[:3]

    new_pos_matrix = frame_matrix.dot(relative_matrix)

    pose = np.zeros(7)
    pose[:3] = new_pos_matrix[:3, -1]
    pose[3:] = mat2quat(new_pos_matrix[:3, :3])[[1, 2, 3, 0]]
    return pose


def pose_transforms(frame_poses, rel_poses):
    assert len(frame_poses) == len(rel_poses)
    return [pose_transform(fr, rel) for (fr, rel) in zip(frame_poses, rel_poses)]


def ori_transform(frame_ori, relative_ori):
    frame_matrix = quat2mat(frame_ori[[3, 0, 1, 2]])
    relative_matrix = quat2mat(relative_ori[[3, 0, 1, 2]])

    new_ori_matrix = frame_matrix.dot(relative_matrix)

    return mat2quat(new_ori_matrix)[[1, 2, 3, 0]]


def normalize_angle(angle, range='pi'):
    """

    Args:
        angle (float or double): Angle
        range (str): 'pi' or '2pi'

    Returns:
        float or double

    """
    if range == 'pi':
        # reduce the angle
        angle = angle % np.pi

        # Force it to be the positive remainder, so that 0 <= angle < 360
        angle = (angle + np.pi) % np.pi

        # Force into the minimum absolute value residue class, so that
        # -180 < angle <= 180
        if angle > np.pi/2:
            angle -= np.pi

        return angle

    else:
        raise NotImplementedError('Only implemented with -pi/pi')


def jacobian_rpy_rate_to_angular_velocity(rpy_angle):
    """Jacobian from RPY angle rates to angular velocities.
    """
    r, p, y = rpy_angle
    T = np.array([
        [1., 0., np.sin(p)],
        [0., np.cos(r), -np.cos(p) * np.sin(r)],
        [0., np.sin(r), np.cos(p) * np.cos(r)]
    ])
    return T


def jacobian_analytical_to_geometric(analytical_jacobian, ori_rpy):
    """
    http://www.diag.uniroma1.it/~deluca/rob1_en/11_DifferentialKinematics.pdf,
    page 23
    Args:
        analytical_jacobian:
        ori_rpy:

    Returns:
        np.ndarray: Geometric Jacobian

    """
    T = jacobian_rpy_rate_to_angular_velocity(ori_rpy)
    Gjacobian = np.vstack(
        (np.hstack((np.identity(3), np.zeros((3, 3)))),
         np.hstack((np.zeros((3, 3)), T)))
    ).dot(analytical_jacobian)
    return Gjacobian


def jacobian_geometric_to_analytical(geo_jacobian, ori_rpy):
    """
    http://www.diag.uniroma1.it/~deluca/rob1_en/11_DifferentialKinematics.pdf,
    page 23
    Args:
        geo_jacobian: 
        ori_rpy: 

    Returns:
        np.ndarray: Analytical Jacobian

    """
    T = jacobian_rpy_rate_to_angular_velocity(ori_rpy)
    Ajacobian = np.vstack(
        (np.hstack((np.identity(3), np.zeros((3, 3)))),
         np.hstack((np.zeros((3, 3)), np.linalg.inv(T))))
    ).dot(geo_jacobian)
    return Ajacobian
