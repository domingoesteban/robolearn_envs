import numpy as np
import pybullet as pb

from robolearn_envs.core.robolearn_sensors import RoboLearnSensor

# ###################### #
# PROPRIOCEPTIVE SENSORS #
# ###################### #


class Encoder(RoboLearnSensor):
    def __init__(self, joint_list, position=True, velocity=True, torque=False,
                 noise=None):
        self._joint_list = joint_list
        self._n_joints = len(joint_list)
        self._noise = noise

        self._state = np.zeros(self._n_joints*sum([position, velocity, torque]))

        # Calculate sensor idxs
        temp_idx = 0
        if position:
            self._pos_idx = temp_idx
            temp_idx += self._n_joints
        else:
            self._pos_idx = None
        if velocity:
            self._vel_idx = temp_idx
            temp_idx += self._n_joints
        else:
            self._vel_idx = None
        if torque:
            self._tor_idx = temp_idx
            temp_idx += self._n_joints
        else:
            self._tor_idx = None
        if temp_idx == 0:
            raise AttributeError('Neither position, velocity nor torque data'
                                 'has been selected.')

    def read(self):
        for jj, joint in enumerate(self._joint_list):
            if self._pos_idx is not None:
                self._state[self._pos_idx + jj] = \
                    joint.get_position()
            if self._vel_idx is not None:
                self._state[self._vel_idx + jj] = \
                    joint.get_velocity()
            if self._tor_idx is not None:
                self._state[self._tor_idx + jj] = \
                    joint.get_torque()

        return self._state.copy()


# ##################### #
# EXTEROCEPTIVE SENSORS #
# ##################### #

class Camera(RoboLearnSensor):
    def __init__(self, body, dist=3, width=320, height=320, noise=False):
        self._cam_body = body
        self._cam_dist = dist
        self._render_width = width
        self._render_height = height
        self._noise = noise
        self._state = np.zeros((width, height, 3))

    def read(self):
        # view_matrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=self._cam_position,
        #     distance=self._cam_dist,
        #     yaw=self._cam_yaw,
        #     pitch=self._cam_pitch,
        #     roll=self._cam_roll,
        #     upAxisIndex=2)
        eye_pose = self._cam_body.get_pose()
        target_pose = pb.multiplyTransforms(
            eye_pose[:3],
            eye_pose[3:],
            [0, 0, self._cam_dist],
            [0, 0, 0, 1]
        )
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=eye_pose[:3],
            cameraTargetPosition=target_pose[0],
            cameraUpVector=[0, 0, 1],
        )

        proj_matrix = pb.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self._render_width)/self._render_height,
            nearVal=0.1,
            farVal=100.0
        )

        (_, _, px, _, _) = pb.getCameraImage(
            width=self._render_width,
            height=self._render_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(px)
        self._state = rgb_array[:, :, :3]

        return self._state.copy()

