import numpy as np
import scipy.optimize
import time

import robolearn_envs.utils.tf_transformations as tf_transformations
from robolearn_envs.utils.transformations import homogeneous_matrix

try:
    import urdf_parser_py
except ImportError:
    pass

try:
    import rbdl
except ImportError:
    pass


class RobotModel(object):
    """
    RobotModel class. Kinematic/Dynamic model of robot.
    Main functions are calculated with the RBDL package.
    """

    def __init__(self, urdf_file, floating_base=False):
        self.floating_base = floating_base
        self.model = rbdl.loadModel(urdf_file, verbose=False,
                                    floating_base=floating_base)
        self.q = np.zeros(self.model.q_size)
        self.qdot = np.zeros(self.model.qdot_size)
        self.qddot = np.zeros(self.model.qdot_size)
        self.tau = np.zeros(self.model.qdot_size)

        self.q_size = self.q.shape[0]
        self.qdot_size = self.qdot.shape[0]
        self.qddot_size = self.qddot.shape[0]

        # Update model
        self.update()

        floating_base_body_id = 0
        self.fb_origin_offset = rbdl.CalcBodyToBaseCoordinates(
            self.model,
            self.q * 0,
            floating_base_body_id,
            np.zeros(3),
            False
        )

    def update(self):
        """
        Update RBDL model with current joint positions, velocities and
        accelerations.
        :return: None
        """
        # rbdl.UpdateKinematicsCustom(_rbdl_model, q_ptr, qdot_ptr, qddot_ptr)
        rbdl.UpdateKinematics(self.model, self.q, self.qdot, self.qddot)

    def get_joint_positions(self):
        """
        Get the Joint Positions of the model.
        :return: Array with the current joint positions.
        """
        return self.q

    def get_joint_velocities(self):
        """
        Get the Joint Velocities of the model.
        :return: Array with the current joint velocities.
        """
        return self.qdot

    def get_joint_accelerations(self):
        """
        Get the Joint Accelerations of the model.
        :return: Array with the current joint accelerations.
        """
        return self.qddot

    def set_joint_positions(self, des_q):
        """
        Set the Joint Positions of the model
        :param des_q: Desired array of joint positions.
        :return: None
        """
        self.q = des_q.copy()

    def set_joint_velocities(self, desired_qdot):
        """
        Set the Joint Velocities of the model
        :param desired_qdot: Desired array of joint velocities.
        :return: None
        """
        self.qdot = desired_qdot.copy()

    def set_joint_accelerations(self, desired_qddot):
        """
        Set the Joint Accelerations of the model
        :param desired_qddot: Desired array of joint accelerations.
        :return: None
        """
        self.qddot = desired_qddot.copy()

    def get_link_id(self, body_name):
        """
        Get the ID (index) of the requested body (link) in the model of the
        robot.
        Returns -1 if the body is not considered in the model of the robot.
        :param body_name: Name of the body (link)
        :return: Body (link) id.
        """
        if body_name.lower() == 'world':
            body_id = self.model.GetBodyId('ROOT')
        else:
            body_id = self.model.GetBodyId(body_name)

        if body_id > self.model.q_size:
            return -1
        else:
            return body_id

    def fk(self, body_name, q=None, body_offset=np.zeros(3),
           update_kinematics=True, rotation_rep='quat'):
        """
        Forward kinematics for a joint configuration.
        :param body_name: Name of the body
        :param q: Joint configuration. If it is not specified, the current
                  q value of the model is used.
        :param body_offset: Body offset.
        :param update_kinematics: Update model with the joint configuration.
        :param rotation_rep: Representation used to represent the rotation of
               the body: 'quat' or 'rpy'
        :return: Pose of the requested body.
        """
        if body_name == 'Waist' and not self.floating_base:
            body_name = 'ROOT'

        if q is None:
            q = self.q

        body_id = self.model.GetBodyId(body_name)
        pos = rbdl.CalcBodyToBaseCoordinates(
            self.model,
            q,
            body_id,
            body_offset,
            update_kinematics=update_kinematics
        )

        rot = rbdl.CalcBodyWorldOrientation(
            self.model,
            q,
            body_id,
            update_kinematics=update_kinematics
        ).T

        if rotation_rep == 'rpy':
            orient = np.array(
                tf_transformations.euler_from_matrix(
                    homogeneous_matrix(rot=rot), axes='sxyz'
                )
            )
        elif rotation_rep == 'quat':
            orient = tf_transformations.quaternion_from_matrix(
                homogeneous_matrix(rot=rot)
            )
        else:
            raise TypeError("Wrong requested rotation representation: %s" %
                            rotation_rep)

        return np.concatenate((orient, pos))

    def ik(self, body_name, desired_pose, body_offset=np.zeros(3),
           mask_joints=None, q_init=None, joints_limits=None,
           method='optimization', regularization_parameter=None):
        """
        
        :param body_name: 
        :param desired_pose: orientation+position
        :param body_offset: 
        :param mask_joints: 
        :param q_init: 
        :param joints_limits: 
        :param method: 
        :param regularization_parameter: 
        :return: 
        """

        if mask_joints is None:
            mask_joints = []

        # Use current configuration as initial guess
        if q_init is None:
            q_init = self.q.copy()

        if method == 'optimization':
            def target_cost(joints):
                # squared_distance = np.linalg.norm(chain.forward_kinematics(x) - target)
                joints[mask_joints] = 0
                actual_pose = self.fk(body_name, q=joints,
                                      body_offset=body_offset,
                                      update_kinematics=True)
                squared_distance = compute_cartesian_error(desired_pose,
                                                           actual_pose,
                                                           rotation_rep='quat')
                squared_distance = np.linalg.norm(squared_distance)
                return squared_distance

            # If a regularization is selected
            if regularization_parameter is not None:
                def total_cost(x):
                    regularization = np.linalg.norm(x - q_init)
                    return target_cost(
                        x) + regularization_parameter * regularization
            else:
                def total_cost(x):
                    return target_cost(x)
            if joints_limits is None:
                real_bounds = [(None, None) for _ in range(self.model.q_size)]
            else:
                real_bounds = joints_limits
            options = {}
            max_iter = None
            if max_iter is not None:
                options['maxiter'] = max_iter
            return scipy.optimize.minimize(total_cost, q_init,
                                           method='L-BFGS-B',
                                           bounds=real_bounds,
                                           options=options).x

        elif method == 'iterative':
            gamma = 0.1  # 0.1
            stol = 1e-6
            J = np.zeros((6, self.model.qdot_size))
            nm = np.inf
            q = q_init
            qdot = np.zeros(self.model.qdot_size)
            actual_pose = self.fk(body_name, q=self.q, body_offset=body_offset,
                                  update_kinematics=True)
            while nm > stol:
                xdot = compute_cartesian_error(desired_pose, actual_pose,
                                               rotation_rep='quat')
                nm = np.linalg.norm(xdot)

                # Compute the jacobian matrix
                rbdl.CalcPointJacobian6D(self.model, q,
                                         self.model.GetBodyId(body_name),
                                         np.zeros(0), J,
                                         update_kinematics=True)
                J[:, mask_joints] = 0

                qdot[:] = np.linalg.lstsq(J, xdot)[0]
                # qdot = np.linalg.pinv(J).dot(xdot)
                # qdot = J.T.dot(xdot)

                # Integrate the computed velocities
                q[:] += qdot * gamma
                actual_pose = self.fk(body_name, q=q, body_offset=body_offset,
                                      update_kinematics=True)

            return q
        else:
            raise TypeError("Wrong IK method: %s" % method)

    def update_jacobian(self, J, body_name, q=None, body_offset=np.zeros(3),
                        update_kinematics=True):
        if body_name == 'Waist':
            body_name = 'ROOT'

        if q is None:
            q = self.q

        body_id = self.model.GetBodyId(body_name)

        rbdl.CalcPointJacobian6D(self.model, q, body_id, body_offset, J,
                                 update_kinematics)

    def jdqd(self, body_name, q=None, qdot=None, body_offset=np.zeros(3),
             update_kinematics=True):
        if body_name == 'Waist' and not self.floating_base:
            body_name = 'ROOT'

        if q is None:
            q = self.q

        if qdot is None:
            qdot = self.qdot

        body_id = self.model.GetBodyId(body_name)

        return rbdl.CalcPointAcceleration6D(self.model, q, qdot,
                                            np.zeros(self.qdot_size), body_id,
                                            body_offset,
                                            update_kinematics=update_kinematics)

    def update_torque(self, tau, q=None, qdot=None, qddot=None):
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        if qddot is None:
            qddot = self.qddot
        rbdl.InverseDynamics(self.model, q, qdot, qddot, tau)

    def inverse_dynamics(self, q=None, qdot=None, qddot=None):
        """
        Calculate the joint torques for the given joint positions, velocities
        and accelerations.
        If any of these are not specified, the current values of the model are
        used.
        :param q: Joint positions.
        :param qdot: Joint velocities.
        :param qddot: Joint accelerations.
        :return: Joint torques
        """
        tau = np.zeros(self.model.qdot_size)
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        if qddot is None:
            qddot = self.qddot

        rbdl.InverseDynamics(self.model, q, qdot, qddot, tau)
        return tau

    def forward_dynamics(self, tau, q=None, qdot=None):
        """
        Calculate the joint accelerations from given joint torques, joint
        positions and joint velocities.
        :param tau: Joint torques.
        :param q: Joint positions. It uses the current model positions if it is
                  not specified.
        :param qdot: Joint accelerations. It uses the current model
                     accelerations if it is not specified.
        :return: Joint Accelerations
        """
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot

        qddot = np.zeros(self.qdot_size)
        rbdl.ForwardDynamics(self.model, q, qdot, tau, qddot)
        return qddot

    def get_nonlinear_forces(self, q=None, qdot=None):
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        c_plus_g = np.zeros(self.qdot_size)
        rbdl.NonlinearEffects(self.model, q, qdot, c_plus_g)
        return g

    def update_nonlinear_forces(self, c_plus_g, q=None, qdot=None):
        if q is None:
            q = self.q
        if qdot is None:
            qdot = self.qdot
        rbdl.NonlinearEffects(self.model, q, qdot, c_plus_g)

    def get_gravity_forces(self, q=None):
        if q is None:
            q = self.q
        g = np.zeros(self.qdot_size)
        rbdl.NonlinearEffects(self.model, q, self.qdot * 0, g)
        return g

    def update_gravity_forces(self, g, q=None):
        if q is None:
            q = self.q
        rbdl.NonlinearEffects(self.model, q, self.qdot * 0, g)

    def get_inertia_matrix(self, q=None, update_kinematics=True):
        if q is None:
            q = self.q
        M = np.zeros((self.qdot_size, self.qdot_size))
        rbdl.CompositeRigidBodyAlgorithm(self.model, q, M, update_kinematics)
        return M

    def update_inertia_matrix(self, M, q=None, update_kinematics=True):
        if q is None:
            q = self.q
        rbdl.CompositeRigidBodyAlgorithm(self.model, q, M, update_kinematics)


class BodyState:
    """
    Author: Songyan Xin
    Modified by: Domingo Esteban
    """

    def __init__(self, model, q, qd, body_id, body_point_position,
                 update_kinematics=True):
        self.body_id = body_id
        self.body_point_position = body_point_position

        self.pos = rbdl.CalcBodyToBaseCoordinates(model, q, body_id,
                                                  body_point_position,
                                                  update_kinematics)
        self.rot = model.X_base[body_id].E.T
        self.transform = homogeneous_matrix(rot=self.rot, pos=self.pos)
        self.quaternion = tf_transformations.quaternion_from_matrix(
            homogeneous_matrix(rot=self.rot)
        )
        self.rpy = np.array(tf_transformations.euler_from_matrix(
            homogeneous_matrix(rot=self.rot), axes='sxyz'))
        self.pose = np.concatenate((self.rpy, self.pos))
        self.vel = rbdl.CalcPointVelocity6D(model, q, qd, body_id,
                                            body_point_position,
                                            update_kinematics)
        self.angular_vel = self.vel[:3]
        self.linear_vel = self.vel[-3:]
        self.Jdqd = rbdl.CalcPointAcceleration6D(
            model,
            q, qd, np.zeros(model.dof_count),
            body_id, body_point_position,
            update_kinematics
        )
        self.temp = np.zeros((6, model.dof_count + 1))  # this is a bug
        rbdl.CalcPointJacobian6D(model, q, body_id, body_point_position,
                                 self.temp, update_kinematics)
        self.J = self.temp[:6, :model.dof_count]


def fk(model, body_name, q=None, body_offset=np.zeros(3),
       update_kinematics=True, rotation_rep='quat'):
    """

    :param model:
    :param body_name:
    :param q:
    :param body_offset:
    :param update_kinematics:
    :param rotation_rep:
    :return:
    """
    body_name = body_name.encode()

    if q is None:
        q = np.zeros(model.q_size)

    body_id = model.GetBodyId(body_name)
    pos = rbdl.CalcBodyToBaseCoordinates(model,
                                         q,
                                         body_id,
                                         body_offset,
                                         update_kinematics=update_kinematics)

    rot = rbdl.CalcBodyWorldOrientation(model,
                                        q,
                                        body_id,
                                        update_kinematics=update_kinematics).T

    if rotation_rep == 'quat':
        orient = tf_transformations.quaternion_from_matrix(
            homogeneous_matrix(rot=rot))
    elif rotation_rep == 'rpy':
        orient = np.array(
            tf_transformations.euler_from_matrix(homogeneous_matrix(rot=rot),
                                                 axes='sxyz'))
    else:
        return AttributeError(
            "Wrong rotation representation %s. Only 'rpy' and 'quat' available." % rotation_rep)
    return np.concatenate((orient, pos))


if __name__ == "__main__":
    from robolearn_envs.robot_models import get_urdf_path

    np.set_printoptions(precision=4, suppress=True, linewidth=1000)
    urdf_file = get_urdf_path('cogimon/cogimon').encode()
    model = rbdl.loadModel(urdf_file, verbose=False, floating_base=True)
    end_effector1 = 'RWrMot3'
    l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
    r_soft_hand_offset = np.array([0.000, 0.030, -0.210])

    # Desired Pose
    q_des = np.zeros(model.q_size)  # *np.deg2rad(10)
    q_des[18] = np.deg2rad(0)
    desired_pose = fk(model, end_effector1, q=q_des,
                      body_offset=l_soft_hand_offset)
    input('wuuu')

    rot = np.zeros((3, 3))
    rot[2, 0] = 1
    rot[1, 1] = 1
    rot[0, 2] = -1
    des_orient = homogeneous_matrix(rot=rot)
    des_orient = tf_transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0])
    des_orient = des_orient.dot(
        tf_transformations.rotation_matrix(np.deg2rad(-8), [1, 0, 0]))
    des_orient = des_orient.dot(
        tf_transformations.rotation_matrix(np.deg2rad(5), [0, 0, 1]))
    desired_pose[:4] = tf_transformations.quaternion_from_matrix(des_orient)
    box_position = [0.75, 0, 0.0184]
    box_size = [0.4, 0.5, 0.3]
    desired_pose[4] = box_position[0] + 0.05
    desired_pose[5] = box_position[1] + box_size[
        1] / 2. - 0.02  # + 0.1  # - 0.03
    desired_pose[6] = box_position[2] + 0.3

    # desired_pose = np.concatenate((rpy, pos))

    # Actual Pose
    q_init = np.zeros(model.q_size)
    q_init[16] = np.deg2rad(30)
    left_sign = np.array([1, 1, 1, 1, 1, 1, 1])
    right_sign = np.array([1, -1, -1, 1, -1, 1, -1])
    value = [-0.1858, 0.256, 0.0451, -1.3449, 0.256, -0.0691, 0.2332]
    q_init[bigman_params['joint_ids']['LA']] = np.array(value) * left_sign
    q_init[bigman_params['joint_ids']['RA']] = np.array(value) * right_sign
    # q_init = touch_box_config
    q = q_init.copy()
    actual_pose = fk(model, end_effector1, q=q_init.copy(),
                     body_offset=l_soft_hand_offset)
    print(actual_pose)

    print("Calculating kinematics")
    # ##################### #
    # IK Iterative solution #
    # ##################### #
    cartesian_error = compute_cartesian_error(desired_pose, actual_pose)
    gamma = 0.1  # 0.1
    J = np.zeros((6, model.qdot_size))
    nm = np.inf
    start = time.time()
    while nm > 1e-6:
        # while False:
        cartesian_error = compute_cartesian_error(desired_pose, actual_pose,
                                                  rotation_rep='quat')
        nm = np.linalg.norm(cartesian_error)
        xdot = cartesian_error.copy()

        # Compute the jacobian matrix
        rbdl.CalcPointJacobian6D(model, q, model.GetBodyId(end_effector1),
                                 np.zeros(0), J, True)
        J[:, 12:15] = 0

        qdot = np.linalg.lstsq(J, xdot)[0]
        # qdot = np.linalg.pinv(J).dot(xdot)
        # qdot = J.T.dot(xdot)
        # print(qdot)

        # Integrate the computed velocities
        q = q + qdot * gamma
        actual_pose = fk(model, end_effector1, q=q,
                         body_offset=l_soft_hand_offset)

    print(repr(q[15:22]))
    print("Time ITER: %s" % str(time.time() - start))

    # #################### #
    # IK with Optimization #
    # #################### #
    q = q_init.copy()


    def optimize_target(q):
        # squared_distance = np.linalg.norm(chain.forward_kinematics(x) - target)
        q[12:15] = 0
        actual_pose = fk(model, end_effector1, q=q,
                         body_offset=l_soft_hand_offset)
        squared_distance = compute_cartesian_error(desired_pose, actual_pose,
                                                   rotation_rep='quat')
        squared_distance = np.linalg.norm(squared_distance)
        return squared_distance


    # If a regularization is selected
    regularization_parameter = None
    if regularization_parameter is not None:
        def optimize_total(x):
            regularization = np.linalg.norm(x - q)
            return optimize_target(
                x) + regularization_parameter * regularization
    else:
        def optimize_total(x):
            return optimize_target(x)
    real_bounds = [(None, None) for _ in range(model.q_size)]
    options = {}
    max_iter = None
    if max_iter is not None:
        options['maxiter'] = max_iter
    start = time.time()
    q_sol = scipy.optimize.minimize(optimize_total, q.copy(),
                                    method='L-BFGS-B', bounds=real_bounds,
                                    options=options).x
    print(repr(q_sol[15:22]))
    print("Time OPT: %s" % str(time.time() - start))

    robot_model = RobotModel(urdf_file)
    q_init = np.zeros(model.q_size)
    # q_init[16] = np.deg2rad(30)
    # q_init = touch_box_config
    q = q_init

    robot_model.set_joint_position(q)
    torso_joints = bigman_params['joint_ids']['TO']
    start = time.time()
    q_sol = robot_model.ik(
        end_effector1,
        desired_pose,
        mask_joints=torso_joints,
        method='optimization'
    )
    print(repr(q_sol[15:22]))
    print("Time OPT: %s" % str(time.time() - start))
    start = time.time()
    q_sol = robot_model.ik(end_effector1, desired_pose,
                           mask_joints=torso_joints, method='iterative')
    print(repr(q_sol[15:22]))
    print("Time ITER: %s" % str(time.time() - start))
