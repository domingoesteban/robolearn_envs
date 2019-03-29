from __future__ import print_function
import numpy as np
import os.path as osp
from robolearn_envs.pybullet.core.bullet_utils import Link
from robolearn_envs.pybullet.core.bullet_sensors import Camera
from robolearn_envs.pybullet.core.bullet_multibody import BulletMultibody


class BulletObject(BulletMultibody):
    def __init__(self, model_file, base_name,
                 init_pos=None, init_ori=None,
                 self_collision=True, use_file_inertia=True,
                 fixed_base=False, pybullet_client=None):
        """Non controllable rigid multi-body.

        Args:
            model_file:
            base_name:
            init_pos:
            init_ori:
            self_collision:
            use_file_inertia:
            fixed_base:
            pybullet_client:
        """
        BulletMultibody.__init__(self, pybullet_client)

        model_type = osp.splitext(model_file)[1][1:]
        if model_type not in ['urdf', 'mjcf', 'sdf']:
            raise NotImplemented("Wrong model_type."
                                 "Only URDF and MJCF are supported")

        self.model_xml = model_file
        self.base_name = base_name
        self.self_collision = self_collision
        self.use_file_intertia = use_file_inertia

        self._model_type = model_type


        if init_pos is None:
            init_pos = [0, 0, 0]
        self.init_base_pos = np.array(init_pos)

        if init_ori is None:
            init_ori = [0, 0, 0, 1]
        self.init_base_ori = np.array(init_ori)

        self.links = None
        self.object_body = None

        self._is_fixed_base = fixed_base

    def spawn_in_pb(self, model_uid=None):
        if model_uid is None:
            if self._model_type == 'urdf':
                load_fcn = self.pbc.loadURDF
            elif self._model_type == 'mjcf':
                load_fcn = self.pbc.loadMJCF
            elif self._model_type == 'sdf':
                load_fcn = self.pbc.loadSDF

            # Spawn the robot again
            if self._model_type == 'urdf':
                # TODO: Use self.use_file_inertia for urdf
                if self.self_collision:
                    flags = self.pbc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT + \
                            self.pbc.URDF_USE_INERTIA_FROM_FILE
                else:
                    flags = self.pbc.URDF_USE_INERTIA_FROM_FILE

                model_uid = load_fcn(
                    self.model_xml,
                    basePosition=self.init_base_pos,
                    baseOrientation=self.init_base_ori,
                    flags=flags,
                    useFixedBase=self._is_fixed_base,
                )
            else:
                model_uid = load_fcn(
                    self.model_xml,
                )
        self.links = {}

        if not isinstance(model_uid, tuple):
            model_uid = [model_uid]

        self._bodies_uids = model_uid

        for i, bb in enumerate(model_uid):
            # if self.pbc.getNumJoints(bb) == 0:
            part_name, object_name = self.pbc.getBodyInfo(bb)
            object_name = object_name.decode("utf8")
            part_name = part_name.decode("utf8")
            self.links[part_name] = Link(part_name, bb, -1,
                                         pbc=self.pbc)
            if len(model_uid) == 1:
                self.object_body = self.links[part_name]
                self.id = bb
            else:
                if i == 0:
                    self.object_body = self.links[part_name]
                    self.id = bb

    def reset(self):
        self.spawn_in_pb()

        return self.get_pose()

    def get_total_bodies(self):
        return len(self.links)

    def get_body_pose(self, body_name):
        return self.links[body_name].get_pose()

    def reset_base_pos(self):
        self.pbc.resetBasePositionAndOrientation()

    def add_camera(self, body_name, dist=3, width=320, height=320):
        return Camera(self.links[body_name], dist=dist,
                      width=width, height=height)
