from __future__ import print_function
import numpy as np
import pybullet as pb
from robolearn_envs.pybullet.core.bullet_utils import Link
from robolearn_envs.pybullet.core.bullet_colors import get_pb_color
from robolearn_envs.pybullet.core.bullet_multibody import BulletMultibody


class BulletVisual(BulletMultibody):
    def __init__(self, visuals, init_pos=None, init_ori=None,
                 pybullet_client=None):
        BulletMultibody.__init__(self, pybullet_client)
        self._visual_kwargs = update_arg_visual_fcns(visuals)

        if init_pos is None:
            init_pos = [0, 0, 0]
        self._init_pos = np.array(init_pos)

        if init_ori is None:
            init_ori = [0, 0, 0, 1]
        self._init_ori = np.array(init_ori)

        self.parts = None
        self.object_body = None

    def spawn_in_pb(self):
        visual_shapes = [self.pbc.createVisualShape(**kwa)
                         for kwa in self._visual_kwargs]

        n_visual_shapes = len(visual_shapes)
        if n_visual_shapes == 1:
            model_uid = self.pbc.createMultiBody(
                baseVisualShapeIndex=visual_shapes[0],
                basePosition=self._init_pos,
                baseOrientation=self._init_ori,
                useMaximalCoordinates=False,
            )
        else:
            link_iterator = range(n_visual_shapes-1)
            model_uid = self.pbc.createMultiBody(
                baseVisualShapeIndex=visual_shapes[0],
                basePosition=self._init_pos,
                baseOrientation=self._init_ori,
                linkMasses=[0],
                linkCollisionShapeIndices=[-1],
                linkVisualShapeIndices=visual_shapes[1:],
                linkPositions=[(0, 0, 0) for _ in link_iterator],
                linkOrientations=[(0, 0, 0, 1) for _ in link_iterator],
                linkInertialFramePositions=[(0, 0, 0) for _ in link_iterator],
                linkInertialFrameOrientations=[(0, 0, 0, 1) for _ in link_iterator],
                linkParentIndices=[0 for _ in link_iterator],
                linkJointTypes=[self.pbc.JOINT_FIXED for _ in link_iterator],
                linkJointAxis=[(0, 0, 0, 1) for _ in link_iterator],
            )
        part_name, _ = self.pbc.getBodyInfo(model_uid)
        part_name = part_name.decode("utf8")
        self.object_body = Link(part_name, model_uid, -1, self.pbc)
        self.id = model_uid

    def reset(self, pose=None):
        self.spawn_in_pb()

        if pose is not None:
            if len(pose) > 3:
                ori = pose[3:]
            else:
                ori = (0, 0, 0, 1)
            self.set_pose(pose[:3], ori)

        return self.get_pose()


def update_arg_visual_fcns(visuals):
    kwargs_list = []
    for key, kwargs in visuals.items():
        if key == 'cylinder':
            shape = pb.GEOM_CYLINDER
        elif key == 'sphere':
            shape = pb.GEOM_SPHERE
        elif key == 'box':
            shape = pb.GEOM_BOX
        elif key == 'mesh':
            shape = pb.GEOM_MESH
        else:
            raise ValueError("Shape %s not supported in BulletVisual!!"
                             % key)
        kwargs['shapeType'] = shape
        if 'rgbaColor' in kwargs and issubclass(type(kwargs['rgbaColor']), str):
            kwargs['rgbaColor'] = get_pb_color(kwargs['rgbaColor'])

        kwargs_list.append(kwargs)

    return kwargs_list
