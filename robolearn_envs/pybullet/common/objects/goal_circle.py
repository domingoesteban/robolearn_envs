from robolearn_envs.pybullet.core.bullet_visual import BulletVisual


class GoalCircle(BulletVisual):
    def __init__(self, init_pos=(0., 0., 0.), init_ori=(0., 0., 0., 1.),
                 color='green', tolerance=0.05, length=0.025):

        visuals = {
            'cylinder': {
                'rgbaColor':color,
                'radius': tolerance,
                'length': length,
            }
        }

        super(GoalCircle, self).__init__(
            init_pos=init_pos,
            init_ori=init_ori,
            visuals=visuals,
        )
