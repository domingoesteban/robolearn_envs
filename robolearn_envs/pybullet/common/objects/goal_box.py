from robolearn_envs.pybullet.core.bullet_visual import BulletVisual


class GoalBox(BulletVisual):
    def __init__(self, init_pos=(0., 0., 0.), init_ori=(0., 0., 0., 1.),
                 color='green', size=None, tolerance=0.01):

        if size is None:
            size = (tolerance, tolerance, tolerance)

        visuals = {
            'box': {
                'rgbaColor': color,
                'halfExtents': size,
            }
        }

        super(GoalBox, self).__init__(
            init_pos=init_pos,
            init_ori=init_ori,
            visuals=visuals,
        )

