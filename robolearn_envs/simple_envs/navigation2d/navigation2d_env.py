import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.colors as mcolors

from gym import spaces
from robolearn_envs.core.robolearn_env import RobolearnEnv
from robolearn_envs.utils.plt_pause import plt_pause

COLOR_DICT = dict(zip(mcolors.CSS4_COLORS.keys(),
                  [mcolors.hex2color(color)
                   for color in mcolors.CSS4_COLORS.values()
                   ]))


class Navigation2dEnv(RobolearnEnv):
    """A 2D point mass has to reach a goal position.

    State: position.
    Action: velocity.
    """

    def __init__(
            self,
            goal_reward=10,
            actuation_cost_coeff=30,
            distance_cost_coeff=1,
            log_distance_cost_coeff=1,
            alpha=1e-6,
            init_position=None,
            init_sigma=0.1,
            goal_position=None,
            goal_threshold=0.1,
            dynamics_sigma=0,
            horizon=None,
            subtask=None,
            seed=None,
    ):
        super(Navigation2dEnv, self).__init__(seed=seed)

        # Bounds
        self._xlim = (-7, 7)
        self._ylim = (-7, 7)
        self._vel_bound = 1.

        # Point-mass Dynamics
        self._dynamics = PointMassDynamics(dim=2, sigma=dynamics_sigma)

        # State/Observation Space
        self.state_space = spaces.Box(
                low=np.array((self._xlim[0], self._ylim[0])),
                high=np.array((self._xlim[1], self._ylim[1])),
                shape=None,
                dtype=np.float32
        )
        self.observation_space = self.state_space

        # Action Space
        self.action_space = spaces.Box(
            low=-self._vel_bound,
            high=self._vel_bound,
            shape=(self._dynamics.a_dim,),
            dtype=np.float32
        )

        # Initial Position
        if init_position is None:
            init_position = (0, 0)
        self.init_mu = np.array(init_position, dtype=np.float32)
        self.init_sigma = init_sigma

        self.current_state = np.array(init_position, dtype=np.float32)

        # Goal Position
        if goal_position is None:
            self.goal_position = np.array([5, 5], dtype=np.float32)
        else:
            self.goal_position = np.array(goal_position, dtype=np.float32)

        # Masks
        self.goal_masks = [[True, True],
                           [True, False],
                           [False, True]]

        # Active subtask
        self.active_subtask = subtask

        # Reward-related Variables
        self._goal_threshold = goal_threshold
        self._goal_reward = goal_reward
        self._action_cost_coeff = actuation_cost_coeff
        self._distance_cost_coeff = distance_cost_coeff
        self._alpha = alpha
        self._log_distance_cost_coeff = log_distance_cost_coeff

        # Maximum Reward
        self._max_rewards = [0, 0, 0]
        reward, rew_info = self._compute_reward(self.goal_position, np.zeros(2))
        self._max_rewards[0] = reward
        self._max_rewards[1] = rew_info['reward_multigoal'][0]
        self._max_rewards[2] = rew_info['reward_multigoal'][1]

        # Main Rendering
        self._main_fig = None
        self._main_ax = None
        self._dynamic_line = None
        self._main_marker = None
        self._env_lines = []

        # Subgoals rendering
        self._subgoals_fig = None
        self._subgoals_ax = None
        self._dynamic_goals_lines = []
        self._subgoal_markers = [None for _ in range(self.n_subgoals)]

        # Time-related variables
        self._horizon = horizon

    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state):
        self._current_state = state

    @property
    def goal_position(self):
        return self._goal_position

    @goal_position.setter
    def goal_position(self, position):
        self._goal_position = position

    def _set_action(self, action):
        next_obs = self._dynamics.forward(self.current_state, action)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.current_state = np.clip(next_obs, o_lb, o_ub)

    def _update_env_state(self):
        return self.current_state

    _update_env_obs = _update_env_state

    def _reset_env(self, *args, **kwargs):
        self.clear_all_plots()

        # Sample an initial state
        unclipped_observation = self.init_mu + \
            self.init_sigma * self.np_random.normal(size=self._dynamics.s_dim)
        o_lb, o_ub = self.observation_space.low, self.observation_space.high
        self.current_state = np.clip(unclipped_observation, o_lb, o_ub)

        observation = self._update_env_obs()

        return observation

    def _check_termination(self):
        if self._horizon is None:
            goal_pos = self.goal_position
            tgt_pos = self.current_state

            goal_poses_masked = [goal_pos[mask] for mask in self.goal_masks]
            tgt_poses_masked = [tgt_pos[mask] for mask in self.goal_masks]

            goal_tgt_dists = [np.linalg.norm(goal_mask - tgt_mask)
                              for goal_mask, tgt_mask in zip(goal_poses_masked,
                                                             tgt_poses_masked)]

            dones = [goal_tgt_dist <= self._goal_threshold
                     for goal_tgt_dist in goal_tgt_dists]

            done = dones[0].item()
            done_multigoal = dones[1:]
        else:
            if self.step_counter >= self._horizon:
                done = True
            else:
                done = False

            done_multigoal = [done for _ in range(self.n_subgoals)]

        if self.active_subtask is not None:
            done = done_multigoal[self.active_subtask].item()

        info = {
            'done_multigoal': done_multigoal,
        }

        return done, info

    def _compute_reward(self, state, action, next_state=None):
        # Penalize the L2 norm of the action (velocity)
        action_cost = self._action_cost_coeff * np.sum(action ** 2, axis=-1)

        # Get goal and subgoals obs
        goal_position_mask = [self.goal_position[mask]
                              for mask in self.goal_masks]

        # Get current obs
        cur_position = state
        if cur_position.ndim == 1:
            cur_position_mask = [cur_position[mask] for mask in self.goal_masks]
        elif cur_position.ndim == 2:
            cur_position_mask = [cur_position[:, mask] for mask in self.goal_masks]
        else:
            raise NotImplementedError

        # Penalize squared dist to goal
        dist_all_goals = np.array([
            # np.sum((goal_pos - cur_pos) ** 2, axis=-1)
            np.linalg.norm(goal_pos - cur_pos, axis=-1)
            for goal_pos, cur_pos in zip(goal_position_mask,
                                         cur_position_mask)
        ])
        goal_costs = self._distance_cost_coeff * dist_all_goals

        # Penalize log dist to goal
        log_goal_costs = self._log_distance_cost_coeff * \
            np.log(dist_all_goals + self._alpha)

        # Bonus for being inside threshold area
        dist_all_goals = np.array(
            [np.linalg.norm(goal_pos - cur_pos, axis=-1)
             for cur_pos, goal_pos in zip(cur_position_mask,
                                          goal_position_mask)]
        )
        any_done = dist_all_goals < self._goal_threshold
        bonus_goal_rewards = self._goal_reward * any_done

        # Compute Multigoal reward
        reward_subtasks = np.array([
            sum([
                -goal_cost,
                -log_goal_cost,
                -action_cost*0.5,
                bonus_goal_reward,
                -max_reward,
            ])
            for goal_cost, log_goal_cost, bonus_goal_reward, max_reward
            in zip(goal_costs[1:], log_goal_costs[1:], bonus_goal_rewards[1:],
                   self._max_rewards[1:])
        ])

        # Compute Main-Task Reward
        reward_composition = np.array([
            -goal_costs[0],
            -log_goal_costs[0],
            -action_cost,
            bonus_goal_rewards[0],
            -self._max_rewards[0]
        ])

        reward = reward_composition.sum(axis=-1)
        if np.prod(reward.shape) == 1:
            reward = reward.item()

        if self.active_subtask is not None:
            reward = reward_subtasks[self.active_subtask].item()

        info = {
            'reward_composition': reward_composition,
            'reward_multigoal': reward_subtasks,
        }

        return reward, info

    def close(self):
        if self._main_fig is not None:
            plt.close(self._main_fig)
            # Main Rendering
            self._main_fig = None
            self._main_ax = None
            self._dynamic_line = None
            self._main_marker = None
            self._env_lines = []

        if self._subgoals_fig is not None:
            plt.close(self._subgoals_fig)
            # Subgoals rendering
            self._subgoals_fig = None
            self._subgoals_ax = None
            self._dynamic_goals_lines = []
            self._subgoal_markers = [None for _ in range(self.n_subgoals)]

    def render(self, mode='human', paths=None):
        if self._main_ax is None:
            self._init_main_plot()

        if self._subgoals_ax is None:
            self._init_subgoals_plot()

        # Close the figures makes the environment to close
        if not self._fig_exist and not self._goal_fig_exist:  # Figures closed
            sys.exit(-1)

        if paths is not None:
            # noinspection PyArgumentList
            [line.remove() for line in self._env_lines]
            self._env_lines = []
            for path in paths:
                positions = path["env_infos"]["pos"]
                xx = positions[:, 0]
                yy = positions[:, 1]
                self._env_lines += self._main_ax.plot(xx, yy, 'b0')
        else:
            if self.current_state is not None:
                if self._dynamic_line is None:
                    self._dynamic_line, = \
                        self._main_ax.plot(self.current_state[0],
                                           self.current_state[1],
                                           color='b',
                                           marker='o',
                                           markersize=2
                                           )

                else:
                    line = self._dynamic_line
                    line.set_xdata(
                        np.append(line.get_xdata(), self.current_state[0])
                    )
                    line.set_ydata(
                        np.append(line.get_ydata(), self.current_state[1])
                    )
                self._main_fig.canvas.set_window_title(
                    'Main Goal | t=%03d' % self.step_counter
                )
                if self._main_marker is not None:
                    self._main_marker.remove()
                self._main_marker = self.draw_robot_marker(self._main_ax,
                                                           self.current_state[0],
                                                           self.current_state[1],
                                                           color='black',
                                                           zoom=0.03)

                if not self._dynamic_goals_lines:
                    n_cols = 2
                    for aa in range(self.n_subgoals):
                        row = aa // n_cols
                        col = aa % n_cols
                        # ax = self._subgoals_ax[row, col]  # ESTO PARA 2 SUBFIG
                        ax = self._subgoals_ax[aa]  # ESTO PARA VARIOS FIG
                        line, = ax.plot(self.current_state[0], self.current_state[1],
                                        color='b', marker='o', markersize=2)
                        self._dynamic_goals_lines.append(line)
                else:
                    for aa in range(self.n_subgoals):
                        line = self._dynamic_goals_lines[aa]
                        line.set_xdata(
                            np.append(line.get_xdata(), self.current_state[0])
                        )
                        line.set_ydata(
                            np.append(line.get_ydata(), self.current_state[1])
                        )
                for aa, fig in enumerate(self._subgoals_figs):
                    fig.canvas.set_window_title(
                        'Subgoals %02d | t=%03d' % (aa, self.step_counter)
                    )
                for aa in range(self.n_subgoals):
                    n_cols = 2
                    row = aa // n_cols
                    col = aa % n_cols
                    # ax = self._subgoals_ax[row, col]
                    ax = self._subgoals_ax[aa]
                    if self._subgoal_markers[aa] is not None:
                        self._subgoal_markers[aa].remove()
                    self._subgoal_markers[aa] = \
                        self.draw_robot_marker(ax, self.current_state[0],
                                               self.current_state[1],
                                               color='black',
                                               zoom=0.03)
                                           # zoom=0.015)

            self._main_fig.canvas.draw()
            # self._subgoals_fig.canvas.draw()
            for fig in self._subgoals_figs:
                fig.canvas.draw()
            plt_pause(0.01)

        return None

    def clear_all_plots(self):
        if self._dynamic_line is not None:
            self._dynamic_line.remove()
            self._main_fig.canvas.draw()
            plt_pause(0.01)
            self._dynamic_line = None

        if self._dynamic_goals_lines:
            for ll in self._dynamic_goals_lines:
                ll.remove()
            if self._subgoals_fig:
                self._subgoals_fig.canvas.draw()
            plt_pause(0.01)
            self._dynamic_goals_lines = []

    def _init_main_plot(self):
        plt.ion()
        self._main_fig = plt.figure(figsize=(7, 7))
        self._main_fig.canvas.set_window_title('Navigation2D Goal Composition'
                                               'Environment | '
                                               't=%002d' % self.step_counter)
        self._main_ax = self._main_fig.add_subplot(111)
        self._main_ax.axis('equal')
        self._main_ax.set_aspect('equal', 'box')

        self._env_lines = []
        self._main_ax.set_xlim(left=self._xlim[0], right=self._xlim[1])
        self._main_ax.set_ylim(bottom=self._ylim[0], top=self._ylim[1])

        # self._main_ax.set_title('Navigation2D Goal Composition Environment')
        self._main_ax.set_xlabel('X', fontweight='bold', fontsize=18)
        self._main_ax.set_ylabel('Y', fontweight='bold', fontsize=18)

        self._plot_main_cost(self._main_ax)

        plt.show(block=False)

    def _init_subgoals_plot(self):
        plt.ion()
        n_cols = 2
        n_rows = int(np.ceil(self.n_subgoals/n_cols))
        # self._subgoals_fig, self._subgoals_ax = plt.subplots(n_rows, n_cols)
        #
        # self._subgoals_ax = np.atleast_2d(self._subgoals_ax)
        #
        # for aa in range(self.n_subgoals):
        #     row = aa // n_cols
        #     col = aa % n_cols
        #     self._subgoals_ax[row, col].set_xlim(self._xlim)
        #     self._subgoals_ax[row, col].set_ylim(self._ylim)
        #     self._subgoals_ax[row, col].set_title('Navigation2D GoalCompo '
        #                                           'Env. | '
        #                                           'Sub-goal %d' % aa)
        #     self._subgoals_ax[row, col].set_xlabel('X')
        #     self._subgoals_ax[row, col].set_ylabel('Y')
        #     self._subgoals_ax[row, col].set_aspect('equal', 'box')

        self._subgoals_figs = []
        self._subgoals_ax = []

        for aa in range(self.n_subgoals):
            self._subgoals_figs.append(plt.figure(figsize=(7, 7)))
            self._subgoals_ax.append(self._subgoals_figs[-1].add_subplot(111))

            self._subgoals_ax[-1].set_xlim(self._xlim)
            self._subgoals_ax[-1].set_ylim(self._ylim)
            # self._subgoals_ax[-1].set_title('Navigation2D GoalCompo '
            #                                 'Env. | '
            #                                 'Sub-goal %d' % aa)
            self._subgoals_ax[-1].set_xlabel('X', fontweight='bold', fontsize=14)
            self._subgoals_ax[-1].set_ylabel('Y', fontweight='bold', fontsize=14)
            self._subgoals_ax[-1].set_aspect('equal', 'box')

        self._plot_subgoals_cost(self._subgoals_ax)

    def _plot_main_cost(self, ax):
        # Create a mesh with X-Y coordinates
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self._xlim))
        y_min, y_max = tuple(1.1 * np.array(self._ylim))
        all_x = np.arange(x_min, x_max, delta)
        all_y = np.arange(y_min, y_max, delta)
        xy_mesh = np.meshgrid(all_x, all_y)

        # Compute sub-goal costs
        all_obs = np.array(xy_mesh).transpose(1, 2, 0).reshape(-1, 2)
        costs = self._compute_reward(all_obs,
                                     action=np.zeros_like(all_obs))[0]
        costs = costs.reshape(len(all_x), len(all_y))

        # Plot cost contour
        contours = ax.contour(xy_mesh[0], xy_mesh[1], costs, 20,
                              colors='dimgray')
        ax.clabel(contours, inline=1, fontsize=10, fmt='%.0f')
        ax.imshow(costs, extent=(x_min, x_max, y_min, y_max), origin='lower',
                  alpha=0.5)

        # Plot Goal coordinates
        x_line = ax.plot([self.goal_position[0], self.goal_position[0]],
                         [y_min, y_max], 'r', alpha=0.3)
        y_line = ax.plot([x_min, x_max],
                         [self.goal_position[1], self.goal_position[1]], 'r',
                         alpha=0.3)

        # Plot Goal
        # goal = ax.plot(self.goal_position[0],
        #                self.goal_position[1], 'ro', alpha=0.1)
        goal = None
        goal_threshold = plt.Circle(self.goal_position, self._goal_threshold,
                                    color='r', alpha=0.35)
        ax.add_artist(goal_threshold)
        # goal_threshold = None

        # Sub-plot appearance
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        return [contours, goal, goal_threshold]

    def _plot_subgoals_cost(self, axs):
        # Create a mesh with X-Y coordinates
        delta = 0.01
        x_min, x_max = tuple(1.1 * np.array(self._xlim))
        y_min, y_max = tuple(1.1 * np.array(self._ylim))
        all_x = np.arange(x_min, x_max, delta)
        all_y = np.arange(y_min, y_max, delta)
        xy_mesh = np.meshgrid(all_x, all_y)

        # Compute sub-goal costs
        all_obs = np.array(xy_mesh).transpose(1, 2, 0).reshape(-1, 2)
        subgoal_costs = \
            self._compute_reward(all_obs, action=np.zeros_like(all_obs))[1]['reward_multigoal']

        n_cols = 2
        goals = [None for _ in range(self.n_subgoals)]
        contours = [None for _ in range(self.n_subgoals)]
        goal_thresholds = [None for _ in range(self.n_subgoals)]

        for aa in range(self.n_subgoals):
            # Get axis
            row = aa // n_cols
            col = aa % n_cols

            # ax = axs[row, col]  # ESTO PARA 2 SUBFIG
            ax = axs[aa]  # ESTO PARA VARIOS FIG

            # Get sub-goal costs
            costs = subgoal_costs[aa].reshape(len(all_x), len(all_y))

            # Plot sub-goal contour
            contours[aa] = ax.contour(xy_mesh[0], xy_mesh[1], costs, 20,
                                      colors='dimgray')
            ax.clabel(contours[aa], inline=1, fontsize=10, fmt='%.0f')
            ax.imshow(costs, extent=(x_min, x_max, y_min, y_max),
                      origin='lower', alpha=0.5)

            # Plot Goal coordinates
            x_line = ax.plot([self.goal_position[0], self.goal_position[0]],
                             [y_min, y_max], 'r', alpha=0.3)
            y_line = ax.plot([x_min, x_max],
                             [self.goal_position[1], self.goal_position[1]],
                             'r', alpha=0.3)

            # Plot sub-goal
            # goals[aa] = ax.plot(self.goal_position[0],
            #                     self.goal_position[1], 'ro', alpha=0.1)
            goals[aa] = None
            goal_thresholds[aa] = \
                plt.Circle(self.goal_position, self._goal_threshold,
                           color='r', alpha=0.35)
            ax.add_artist(goal_thresholds[aa])

            # Sub-plot appearance
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_min, y_max])

        return [contours, goals, goal_thresholds]

    @property
    def _fig_exist(self):
        if self._main_fig is None:
            return False
        else:
            return plt.fignum_exists(self._main_fig.number)

    @property
    def _goal_fig_exist(self):
        if self._subgoals_fig is None:
            return False
        else:
            return plt.fignum_exists(self._subgoals_fig.number)

    @staticmethod
    def draw_robot_marker(axis, x, y, color='red', zoom=0.03):

        image = plt.imread(os.path.join(os.path.dirname(__file__), 'figures',
                                        'robotio.png'))

        if isinstance(color, str):
            for cc in range(3):
                image[:, :, cc] = COLOR_DICT[color][cc]
        else:
            image[:, :, :] = color

        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        return axis.add_artist(ab)

    @property
    def n_subgoals(self):
        return len(self.goal_masks)-1

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


class PointMassDynamics(object):
    """
    State: position
    Action: velocity
    """
    def __init__(self, dim, sigma, np_random=None):
        self.dim = dim
        self.sigma = sigma
        self.s_dim = dim
        self.a_dim = dim

        if np_random is None:
            self.np_random = np.random

    def forward(self, state, action):
        mu_next = state + action
        state_next = mu_next + self.sigma * \
            self.np_random.normal(size=self.s_dim)
        return state_next


if __name__ == "__main__":
    # render = False
    render = True

    H = 20

    env = Navigation2dEnv(horizon=H)
    print('env_reset')
    env.reset()
    if render:
        env.render()

    for tt in range(H):
        print('env_step: %02d' % tt)
        obs, reward, done, info = env.step(env.action_space.sample())
        if render:
            env.render()

        if done:
            print("Environment is done!")
            break

    print("Correct!")
