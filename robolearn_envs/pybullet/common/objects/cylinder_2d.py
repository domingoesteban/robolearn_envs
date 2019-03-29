import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class Cylinder2d(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='white'):

        urdf_file = 'cylinder_2d'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/cylinders/'+urdf_file+'.urdf')

        self.color = color

        super(Cylinder2d, self).__init__(
            model_file=urdf_xml,
            base_name='target',
            init_pos=init_pos,
            self_collision=True
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

        self.set_color(self.color)



