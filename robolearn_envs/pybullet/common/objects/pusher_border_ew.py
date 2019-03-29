from robolearn_envs.pybullet.core.bullet_object import BulletObject
from robolearn_envs.pybullet.core.bullet_multibody import BulletMultibody
import pybullet as pb
from robolearn_envs.pybullet.core.bullet_colors import pb_colors
from robolearn_envs.utils.transformations import create_quat
import numpy as np


class PusherBorderEW(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), init_ori=(0., 0., 0., 1.),
                 color='green', pybullet_client=None):
        BulletMultibody.__init__(self, pybullet_client)

        # mjcf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                         'models/goals/'+'goal_sphere'+'.xml')
        #                         # 'models/goals/'+'goal_sphere'+'.urdf')
        #
        # init_pos = (-0.3, 0.0, 0.01)

        self._init_pos = init_pos
        self._init_ori = init_ori

        self.color = color

        # # super(GoalSphere, self).__init__(model_type='urdf',
        # super(GoalSphere, self).__init__(model_type='mjcf',
        #                                  model_xml=mjcf_xml,
        #                                  base_name='goal_circle',
        #                                  init_pos=init_pos,
        #                                  self_collision=True)

    def spawn_in_pb(self):
        # BulletObject.reset(self)

        self.parts, self.object_body = None, None

        # Border West
        visualShapeId = self.pbc.createVisualShape(
            rgbaColor=pb_colors[self.color],
            # shapeType=pb.GEOM_CYLINDER,
            # radius=0.02,
            # length=2,
            shapeType=pb.GEOM_BOX,
            halfExtents=[0.02, 0.02, 1.0],
        )
        collisionShapeId = self.pbc.createCollisionShape(
            # shapeType=pb.GEOM_CYLINDER,
            # radius=0.02,
            # height=2,
            shapeType=pb.GEOM_BOX,
            halfExtents=[0.02, 0.02, 1.0],
        )
        model_uid = self.pbc.createMultiBody(
            baseCollisionShapeIndex=collisionShapeId,
            baseVisualShapeIndex=visualShapeId,
            basePosition=self._init_pos,
            baseOrientation=create_quat(rot_roll=np.deg2rad(90))
        )
        BulletObject.spawn_in_pb(self, model_uid)

    def reset(self, pose=None):

        self.spawn_in_pb()

        if pose is not None:
            if len(pose) > 3:
                ori = pose[3:]
            else:
                ori = (0, 0, 0, 1)
            self.set_pose(pose[:3], ori)

        self.set_color(self.color)
