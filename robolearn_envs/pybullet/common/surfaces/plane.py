import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class Plane(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color=None,
                 plane_type='checker_blue', pybullet_client=None):

        if plane_type.lower() == 'stone':
            urdf_file = 'plane_stone'
        elif plane_type.lower() == 'simple':
            urdf_file = 'plane_simple'
        else:
            urdf_file = 'plane_checker_blue'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models',
                                'plane',
                                urdf_file+'.urdf'
                                )

        self.color = color

        super(Plane, self).__init__(
            model_file=urdf_xml,
            base_name='plane',
            init_pos=init_pos,
            self_collision=True,
            pybullet_client=pybullet_client,
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

        if self.color is not None:
            self.set_color(self.color)



