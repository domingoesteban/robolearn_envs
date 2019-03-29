import os
from robolearn_envs.pybullet.core.bullet_visual import BulletVisual


class GoalArrow(BulletVisual):
    def __init__(self, init_pos=(0., 0., 0.), init_ori=(0., 0., 0., 1.),
                 color='green', tolerance=0.1):

        mesh_file = \
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'models',
                         'goals',
                         'arrow_obj',
                         'arrow.obj'
                         )
        mesh_scale = (tolerance*1.5, 0.1, 0.1)

        visuals = {
            'sphere': {
                'rgbaColor': color,
                'radius': tolerance,
            },
            'mesh': {
                'fileName': mesh_file,
                'meshScale': mesh_scale,
                'rgbaColor': color,
            },
        }

        super(GoalArrow, self).__init__(
            init_pos=init_pos,
            init_ori=init_ori,
            visuals=visuals,
        )
