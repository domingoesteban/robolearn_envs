import os
from robolearn_envs.pybullet.core.bullet_visual import BulletVisual


class GoalSphere(BulletVisual):
    def __init__(self, init_pos=(0., 0., 0.), init_ori=(0., 0., 0., 1.),
                 color='green', tolerance=0.01):

        visuals = {
            'spherer': {
                'rgbaColor': color,
                'radius': tolerance,
            }
        }

        super(GoalSphere, self).__init__(
            init_pos=init_pos,
            init_ori=init_ori,
            visuals=visuals,
        )
