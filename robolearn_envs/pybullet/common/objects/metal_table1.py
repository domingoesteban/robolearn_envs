import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class MetalTable1(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='gray',
                 pybullet_client=None):

        urdf_file = 'metal_table1'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models',
                                'tables',
                                urdf_file+'.urdf'
                                )

        self.color = color

        super(MetalTable1, self).__init__(
            model_file=urdf_xml,
            base_name='target',
            init_pos=init_pos,
            self_collision=True,
            pybullet_client=pybullet_client,
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

        self.set_color(self.color)



