import os
import numpy as np
from robolearn_envs.pybullet.core.bullet_object import BulletObject
import pybullet as pb
from robolearn_envs.pybullet.common.objects.pusher_border_ew \
    import PusherBorderEW


class PusherBorder(BulletObject):
    def __init__(self, init_pos=(0., 0., 0.), color='white', half_env=False,
                 pybullet_client=None):

        if half_env:
            mjcf_xml = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models/pusher_border/pusher_border_small.xml'
            )
        else:
            mjcf_xml = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'models/pusher_border/pusher_border.xml'
            )

        self.color = color
        self._half_env = half_env

        super(PusherBorder, self).__init__(
            model_file=mjcf_xml,
            base_name='reacher_border',
            init_pos=init_pos,
            self_collision=True,
            pybullet_client=pybullet_client,
        )

        # Border West
        borderW_pos = np.array(init_pos)
        borderW_pos[2] += 0.01
        if half_env:
            borderW_pos[0] -= 0.3
        else:
            borderW_pos[0] -= 1.00
        self._borderW = PusherBorderEW(
            init_pos=borderW_pos,
            color=color,
            pybullet_client=pybullet_client
        )

    def reset(self, pose=None):
        BulletObject.reset(self)

        self._borderW.pbc = self.pbc
        self._borderW.reset()

        if pose is not None:
            self.set_pose(pose[:3], pose[3:])

            borderW_pos = np.array(pose[:3])
            borderW_pos[2] += 0.01
            if self._half_env:
                borderW_pos[0] -= 0.3
            else:
                borderW_pos[0] -= 0.5
            self._borderW.set_pose(pose[:3], pose[3:])

        self.set_color(self.color)



