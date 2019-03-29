import os
from robolearn_envs.pybullet.core.bullet_object import BulletObject


class Drill(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='blue',
                 fixed_base=False, pybullet_client=None):

        file_name = 'drill'

        xml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models/cordless_drill/'+file_name+'.sdf')

        self.color = color

        super(Drill, self).__init__(
            model_file=xml_file,
            base_name='drill',
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



