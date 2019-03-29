import os
from robolearn_envs.pybullet.core.bullet_visual import BulletVisual


class PusherGoalLine(BulletVisual):
    def __init__(self, init_pos=(0., 0., 0.), axis='x', color='blue',
                 border_half=True, tolerance=0.01, height=0.01):

        if axis.lower() == 'x':
            x_size = tolerance
            y_size = 1.00
        else:
            y_size = tolerance
            if border_half:
                x_size = 0.65
            else:
                x_size = 1.00

        visuals = {
            'box': {
                'rgbaColor': color,
                'halfExtents': [x_size, y_size, height],
            }
        }

        super(PusherGoalLine, self).__init__(
            init_pos=init_pos,
            visuals=visuals,
        )
