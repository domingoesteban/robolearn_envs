import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class Bottle(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='white', bottle_type='C',
                 fixed_base=False):

        urdf_file = 'wine_bottle'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/bottle/'+urdf_file+'.urdf')

        self.color = color

        super(Bottle, self).__init__(
            model_file=urdf_xml,
            base_name='target',
            init_pos=init_pos,
            self_collision=True,
            fixed_base=fixed_base
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

        self.set_color(self.color)



