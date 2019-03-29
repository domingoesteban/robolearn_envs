from __future__ import print_function
import numpy as np
from robolearn_envs.pybullet.planarmanipulators.pusher2d_env import Pusher2DBulletEnv
from robolearn_envs.pybullet.common.objects.pusher_border_ew import PusherBorderEW
from robolearn_envs.pybullet.common import PusherGoalLine

from robolearn_envs.reward_functions.state_rewards \
    import exponential_distance_reward


class Pusher2DGoalCompoEnv(Pusher2DBulletEnv):
    def __init__(
            self,
            robot_dof=3,
            is_render=False,
            obs_distances=False,
            only_position=True,  # TODO: Scale yaw cost if only_position=False
            obs_with_goal=False,  # TODO: Check if we want this
            goal_pose=None,
            rdn_goal_pose=True,
            tgt_pose=None,
            rdn_tgt_object_pose=True,
            robot_config=None,
            rdn_robot_config=True,
            tgt_cost_weight=1.0,
            goal_cost_weight=3.0,
            goal_tolerance=0.05,
            ctrl_cost_weight=1.0e-2,
            max_time=None,  # seconds
            sim_timestep=1/240.,
            frame_skip=1,
            half_env=True,
            subtask=None,
            seed=None,
    ):
        """Pusher
        """
        super(Pusher2DGoalCompoEnv, self).__init__(
            robot_dof=robot_dof,
            is_render=is_render,
            obs_distances=obs_distances,
            only_position=only_position,
            rdn_goal_pose=rdn_goal_pose,
            goal_pose=goal_pose,
            tgt_pose=tgt_pose,
            rdn_tgt_object_pose=rdn_tgt_object_pose,
            robot_config=robot_config,
            rdn_robot_config=rdn_robot_config,
            tgt_cost_weight=tgt_cost_weight,
            goal_cost_weight=goal_cost_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            goal_tolerance=goal_tolerance,
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
        super(Pusher2DGoalCompoEnv, self)._reset_env(*args, **kwargs)

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
            done1 = self.get_ee_pose()[0] <= -0.10
            if done1:
                done_trigger = 'failure-robot-pose'
            else:
                done_trigger = False

            done2 = self.get_object_pose()[0] <= -0.10
            if done2:
                # TODO: Find a better way to show this
                done_trigger = [done_trigger, 'failure-object-pose'] \
                    if done_trigger else 'failure-object-pose'

            else:
                done_trigger = False
            done = done1 or done2
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
        goal_pose = self.get_goal_pose()[:self._pose_dof]
        object_pose = self.get_object_pose()[:self._pose_dof]
        ee_pose = self.get_ee_pose()[:self._pose_dof]

        # Distance object to goal
        goal_masks = np.array([
            [True, True, True],  # X and Y
            [True, False, True],  # X
            [False, True, True]   # Y
        ])
        goal_masks = goal_masks[:, :self._pose_dof]
        goal_obj_rewards = np.array([
            exponential_distance_reward(
                goal_pose[mask] - object_pose[mask],
                # temperature=.4)[0].item()
                # temperature=.7)[0].item()  # Experimentos 25-02
                # temperature=.1)[0].item()  # Experimentos 07-02
                # temperature=0.6)[0].item()  # Experimentos 26-02
                # temperature=0.9)[0].item()  # Experimentos 26-02
                temperature=0.1)[0].item()  # Experimentos 27-02  9.2pm
            for mask in goal_masks
        ])
        goal_obj_rewards = self._goal_cost_weight * goal_obj_rewards

        # Give distance object to goal reward only if touching it and inside gripper
        # goal_obj_rewards *= (self.is_robot_touching_cylinder(only_with_ee=True)
        #                      and self.is_cylinder_inside_gripper())
        goal_obj_rewards *= self.is_cylinder_inside_gripper()

        # Distance ee to object (Same for all subgoals)
        obj_ee_reward = exponential_distance_reward(object_pose - ee_pose,
                # temperature=1.)[0].item() * np.ones(3)  # Experimentos antes 26-02
                # temperature=.15)[0].item() * np.ones(3)
                temperature=0.6)[0].item() * np.ones(3)
        ee_obj_rewards = self._tgt_cost_weight * obj_ee_reward

        # Control
        ctrl_cost = self._ctrl_cost_weight * np.square(action).sum()

        reward_subtasks = np.array([
            + ee_obj_reward
            + goal_obj_reward
            - ctrl_cost
            for ee_obj_reward, goal_obj_reward
            in zip(ee_obj_rewards[1:], goal_obj_rewards[1:])
        ]).squeeze()

        reward_composition = np.array([
            ee_obj_rewards[0],
            goal_obj_rewards[0],
            -ctrl_cost,
        ])
        # print(reward_composition, self.is_robot_touching_cylinder(only_with_ee=True), self.is_cylinder_inside_gripper())
        # input('wuuu')
        # reward = reward_subtasks.mean().item()
        # reward = reward_subtasks.sum().item()
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

    env = Pusher2DGoalCompoEnv(robot_dof=4,
                               is_render=render)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        # next_obs, rew, done, info = env.step(env.action_space.sample())
        next_obs, rew, done, info = env.step(np.zeros_like(env.action_space.sample()))

        if done:
            print("The environment is done!")
            print(info.get('done_trigger', 'The env does not say why!'))
            break

        if render:
            env.render()

    print("Correct!")
