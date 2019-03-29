import numpy as np
import rbdl
import pybullet as pb
from robolearn_envs.robot_models import get_urdf_path
from robolearn_envs.utils.robot_model import (
    RobotModel,
    fk
)
from robolearn_envs.pybullet.cogimon.cogimon import Cogimon

np.set_printoptions(precision=4, suppress=True, linewidth=1000)

# Instantiate robot in pybullet
init_pos = [-5.68993626e-02, 1.17457386e-03, 9.61556668e-01]
init_ori = [4.01970997e-05, -2.97989642e-02, 3.99505371e-05,
            9.99555911e-01]

# pb_client = pb.connect(pb.DIRECT)
pb_client = pb.connect(pb.GUI)

robot = Cogimon(
    init_pos=init_pos,
    init_ori=init_ori,
)
urdf_file = robot.model_xml.encode()

# Reset robot
robot_state = robot.reset(physicsClientId=pb_client)

# Robot Model
robot_model = RobotModel(urdf_file=robot.model_xml.encode(),
                         floating_base=not robot.is_fixed_base)

# Rbdl Model
rbdl_model = rbdl.loadModel(urdf_file, verbose=False,
                            floating_base=not robot.is_fixed_base)

# ######## #
# Check FK #
# ######## #
q_des = robot.n_ordered_joints
print('q_robot', robot.n_ordered_joints)
print('q_model', robot_model.q_size)
print('q_rbdl', rbdl_model.q_size, rbdl_model.qdot_size)

print('\n****'*3)
print('Body names:')
print('-----------')
for name in robot.get_body_parts_names():
    print(name)
print('\n****'*3)

print('Joint names:')
print('-----------')
for nn, name in enumerate(robot.get_joints_names()):
    print(nn, name)
print(dir(rbdl_model.mJoints))

print(len(robot.initial_configuration))

q_des = np.zeros(rbdl_model.q_size)#*np.deg2rad(10)

end_effector1 = 'ROOT'
# l_soft_hand_offset = np.array([0.000, -0.030, -0.210])
# r_soft_hand_offset = np.array([0.000, 0.030, -0.210])
l_soft_hand_offset = np.array([0.0, 0.0, 0.0])
r_soft_hand_offset = np.array([0.0, 0.0, 0.0])

# Desired Pose
desired_pose = fk(rbdl_model, end_effector1, q=q_des,
                  body_offset=l_soft_hand_offset)

body_fk = lambda body_name: \
    fk(rbdl_model, body_name, q=q_des, body_offset=np.array([0., 0., 0.]))

# for q in range(rbdl_model.q_size):
for q in range(4):
    name = rbdl_model.GetBodyName(q)
    if len(name) > 0:
        print(q, name.decode(), body_fk(name.decode()))

print('des_pose', desired_pose)
input('wuuu')

rot = np.zeros((3, 3))
rot[2, 0] = 1
rot[1, 1] = 1
rot[0, 2] = -1
des_orient = homogeneous_matrix(rot=rot)
des_orient = tf_transformations.rotation_matrix(np.deg2rad(-90), [0, 1, 0])
des_orient = des_orient.dot(tf_transformations.rotation_matrix(np.deg2rad(-8), [1, 0, 0]))
des_orient = des_orient.dot(tf_transformations.rotation_matrix(np.deg2rad(5), [0, 0, 1]))
desired_pose[:4] = tf_transformations.quaternion_from_matrix(des_orient)
box_position = [0.75, 0, 0.0184]
box_size = [0.4, 0.5, 0.3]
desired_pose[4] = box_position[0] + 0.05
desired_pose[5] = box_position[1] + box_size[1]/2. - 0.02  #  + 0.1  # - 0.03
desired_pose[6] = box_position[2] + 0.3

#desired_pose = np.concatenate((rpy, pos))

#Actual Pose
q_init = np.zeros(model.q_size)
q_init[16] = np.deg2rad(30)
left_sign = np.array([1, 1, 1, 1, 1, 1, 1])
right_sign = np.array([1, -1, -1, 1, -1, 1, -1])
value = [-0.1858,  0.256 ,  0.0451, -1.3449,  0.256 , -0.0691,  0.2332]
q_init[bigman_params['joint_ids']['LA']] = np.array(value)*left_sign
q_init[bigman_params['joint_ids']['RA']] = np.array(value)*right_sign
#q_init = touch_box_config
q = q_init.copy()
actual_pose = fk(model, end_effector1, q=q_init.copy(), body_offset=l_soft_hand_offset)
print(actual_pose)

print("Calculating kinematics")
# ##################### #
# IK Iterative solution #
# ##################### #
cartesian_error = compute_cartesian_error(desired_pose, actual_pose)
gamma = 0.1#0.1
J = np.zeros((6, model.qdot_size))
nm = np.inf
start = time.time()
while nm > 1e-6:
    #while False:
    cartesian_error = compute_cartesian_error(desired_pose, actual_pose, rotation_rep='quat')
    nm = np.linalg.norm(cartesian_error)
    xdot = cartesian_error.copy()

    # Compute the jacobian matrix
    rbdl.CalcPointJacobian6D(model, q, model.GetBodyId(end_effector1), np.zeros(0), J, True)
    J[:, 12:15] = 0

    qdot = np.linalg.lstsq(J, xdot)[0]
    #qdot = np.linalg.pinv(J).dot(xdot)
    #qdot = J.T.dot(xdot)
    #print(qdot)

    # Integrate the computed velocities
    q = q + qdot * gamma
    actual_pose = fk(model, end_effector1, q=q, body_offset=l_soft_hand_offset)

print(repr(q[15:22]))
print("Time ITER: %s" % str(time.time() - start))


# #################### #
# IK with Optimization #
# #################### #
q = q_init.copy()

def optimize_target(q):
    #squared_distance = np.linalg.norm(chain.forward_kinematics(x) - target)
    q[12:15] = 0
    actual_pose = fk(model, end_effector1, q=q, body_offset=l_soft_hand_offset)
    squared_distance = compute_cartesian_error(desired_pose, actual_pose, rotation_rep='quat')
    squared_distance = np.linalg.norm(squared_distance)
    return squared_distance
# If a regularization is selected
regularization_parameter = None
if regularization_parameter is not None:
    def optimize_total(x):
        regularization = np.linalg.norm(x - q)
        return optimize_target(x) + regularization_parameter * regularization
else:
    def optimize_total(x):
        return optimize_target(x)
real_bounds = [(None, None) for _ in range(model.q_size)]
options = {}
max_iter = None
if max_iter is not None:
    options['maxiter'] = max_iter
start = time.time()
q_sol = scipy.optimize.minimize(optimize_total, q.copy(), method='L-BFGS-B', bounds=real_bounds, options=options).x
print(repr(q_sol[15:22]))
print("Time OPT: %s" % str(time.time() - start))

robot_model = RobotModel(urdf_file)
q_init = np.zeros(model.q_size)
#q_init[16] = np.deg2rad(30)
#q_init = touch_box_config
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
q_sol = robot_model.ik(end_effector1, desired_pose, mask_joints=torso_joints, method='iterative')
print(repr(q_sol[15:22]))
print("Time ITER: %s" % str(time.time() - start))
