import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class Cylinder(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='white', cylinder_type='C',
                 fixed_base=False, pybullet_client=None):

        if cylinder_type.upper() == 'C':
            urdf_file = 'target_cylinder'
        elif cylinder_type.upper() == 'CS':
            urdf_file = 'target_cylinderS'
        elif cylinder_type.upper() == 'HEAVY':
            urdf_file = 'cylinder1_heavy'
        elif cylinder_type.upper() == 'T':  # Tall
            urdf_file = 'cylinder2'
        else:
            urdf_file = 'target_sphere'

        urdf_xml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/cylinders/'+urdf_file+'.urdf')

        self.color = color

        super(Cylinder, self).__init__(
            model_file=urdf_xml,
            base_name='target',
            init_pos=init_pos,
            self_collision=True,
            fixed_base=fixed_base,
            pybullet_client=pybullet_client,
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

        self.set_color(self.color)



