import numpy as np
from robolearn_envs.pybullet.planarmanipulators.reacher2d_env import Reacher2DEnv
from robolearn_envs.pybullet.common.objects.pusher_border_ew import PusherBorderEW
from robolearn_envs.pybullet.common import PusherGoalLine
from robolearn_envs.reward_functions.action_rewards import quadratic_action_cost
from robolearn_envs.reward_functions.state_rewards \
    import exponential_distance_reward

# TODO: THE FOLLOWING SHOULD BE CONSIDERED BY REACHER2d3dof_env
# HALF_GOAL_RANGE_X = (0.40, 0.70)
# HALF_GOAL_RANGE_Y = (-0.2, 0.70)
HALF_GOAL_RANGE_X = (0.63, 0.68)
HALF_GOAL_RANGE_Y = (0.33, 0.38)
FULL_GOAL_RANGE_X = (-0.10, 0.70)
FULL_GOAL_RANGE_Y = (-0.7, 0.70)
MAX_GOAL_NORM = 0.8
MIN_GOAL_NORM = 0.4
# ROBOT_CONFIG_RANGE = ((-2.18, 2.18),
#                       (-2.18, 2.18),
#                       (-2.18, 2.18))

# ROBOT_CONFIG_RANGE = ((-1.45, -1.25),
#                       (1.50, 2.00),
#                       (0.25, 1.15))

ROBOT_CONFIG_RANGE = ((-1.40, -1.20),
                      (1.70, 1.90),
                      (0.25, 1.15))


class Reacher2DGoalCompoEnv(Reacher2DEnv):
    def __init__(
            self,
            robot_dof=3,
            is_render=False,
            obs_distances=False,
            only_position=True,  # TODO: Scale yaw cost if only_position=False
            obs_with_goal=False,  # TODO: Check if we want this
            goal_pose=None,
            rdn_goal_pos=True,
            robot_config=None,
            rdn_robot_config=True,
            goal_cost_weight=1.0,
            goal_tolerance=0.05,
            ctrl_cost_weight=1.0e-2,
            use_log_distances=False,
            log_alpha=1e-6,
            max_time=None,  # seconds
            half_env=True,
            sim_timestep=1/240.,
            frame_skip=1,
            subtask=None,
            seed=None,
    ):
        """ Reacher
        """
        super(Reacher2DGoalCompoEnv, self).__init__(
            robot_dof=robot_dof,
            is_render=is_render,
            obs_distances=obs_distances,
            only_position=only_position,
            rdn_tgt_pos=rdn_goal_pos,
            tgt_pose=goal_pose,
            robot_config=robot_config,
            rdn_robot_config=rdn_robot_config,
            tgt_cost_weight=goal_cost_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            use_log_distances=use_log_distances,
            log_alpha=log_alpha,
            tgt_tolerance=goal_tolerance,
            max_time=max_time,
            half_env=half_env,
            sim_timestep=sim_timestep,
            frame_skip=frame_skip,
            seed=seed,
        )

        # Border2
        self._border2 = PusherBorderEW(color='green')  # For collision
        self.add_to_sim(self._border2)

        self.subgoals = []
        # Subgoal X
        subgoal_x = PusherGoalLine(
            init_pos=[0., 0., 0.01],
            axis='x',
            color='orange',
            border_half=self._half_env,
        )
        self.subgoals.append(subgoal_x)
        self.add_to_sim(subgoal_x)
        # Subgoal Y
        subgoal_y = PusherGoalLine(
            init_pos=[0., 0., 0.01],
            axis='y',
            color='blue',
            border_half=self._half_env,
        )
        self.subgoals.append(subgoal_y)
        self.add_to_sim(subgoal_y)

        self.active_subtask = subtask

    def _reset_env(self, *args, **kwargs):
        super(Reacher2DGoalCompoEnv, self)._reset_env(*args, **kwargs)

        goal_pose = self.goal.get_pose()
        if self._is_env_instantiation_complete:
            # Subgoal X
            self.subgoals[0].reset(
                pose=(goal_pose[0], 0, -0.0055)
            )
            # Subgoal Y
            self.subgoals[1].reset(
                pose=(0.35 if self._half_env else 0., goal_pose[1], -0.0055)
            )

    def _check_termination(self):
        if self.max_time is None:
            # Done when EE is far from limit
            done = self.get_ee_pose()[0] <= -0.10
            if done:
                done_trigger = 'failure'
            else:
                done_trigger = False
        else:
            if self.env_time >= self.max_time:
                done = True
                done_trigger = 'time'
            else:
                done = False
                done_trigger = False

        done_multigoal = [done for _ in range(self.n_subgoals)]

        if self.active_subtask is not None:
            done = done_multigoal[self.active_subtask].item()

        info = {
            'done_trigger': done_trigger,
            'done_multigoal': done_multigoal,
        }

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        # Consider only X-Y coordinates
        ee_pos = self.get_ee_pose()[:self._pose_dof]
        goal_pos = self.get_goal_pose()[:self._pose_dof]

        # Distance ee to goal
        goal_masks = np.array([
            [True, True, True],  # X and Y
            [True, False, True],  # X
            [False, True, True]   # Y
        ])
        goal_masks = goal_masks[:, :self._pose_dof]
        goal_ee_rewards = np.array([
            exponential_distance_reward(goal_pos[mask] - ee_pos[mask],
                                        # temperature=0.6)[0]  # 27/02 8pm
                                        temperature=0.4)[0].item()
            for mask in goal_masks
        ])
        goal_ee_rewards = self._tgt_cost_weight * goal_ee_rewards

        # Control
        # ctrl_cost = np.square(action).sum()
        lb = self.low_action_bounds
        ub = self.high_action_bounds
        scaling = (ub - lb) * 0.5
        cost, _ = quadratic_action_cost(action, weights=1. / scaling)
        ctrl_cost = self._ctrl_cost_weight * cost

        reward_composition = np.array([
            goal_ee_rewards[0],
            -ctrl_cost,
            # -self._max_rewards[0],
        ])

        reward_subtasks = goal_ee_rewards[1:]

        # Add common reward_functions:
        reward_subtasks -= ctrl_cost
        # reward -= ctrl_cost

        reward = reward_composition.sum().item()

        if self.active_subtask is not None:
            reward = reward_subtasks[self.active_subtask].item()

        info = {
            'reward_composition': reward_composition,
            'reward_multigoal': reward_subtasks,
        }

        return reward, info

    @property
    def n_subgoals(self):
        return len(self.subgoals)

    @property
    def active_subtask(self):
        return self._active_subtask

    @active_subtask.setter
    def active_subtask(self, subtask):
        if subtask not in list(range(0, self.n_subgoals)) + [None]:
            print("Wrong option '%s'!" % subtask)
        self._active_subtask = None if subtask == -1 else subtask

    def get_active_subtask(self):
        return self.active_subtask

    def set_active_subtask(self, subtask):
        self.active_subtask = subtask


if __name__ == "__main__":
    render = True
    # render = False

    H = 500

    env = Reacher2DGoalCompoEnv(robot_dof=3,
                                is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        next_obs, rew, done, info = env.step(env.action_space.sample())

        if done:
            print("The environment is done!")
            print(info.get('done_trigger', 'The env does not say why!'))
            break

        if render:
            env.render()

    print("Correct!")
    input('fsd')
