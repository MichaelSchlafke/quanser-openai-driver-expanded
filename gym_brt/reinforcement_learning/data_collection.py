# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# TODO: TEST
# TODO: Add energy logging


class Log:
    """
    Log class for logging data from the environment
    Dataframes:
        - log contains all sates, actions, and rewards for each time step in the current episode
        - episode_log contains the average and max values for each state, the sum of all actions, and whether the episode
            ended due to time constraints or hitting the constraints
    """
    def __init__(self):
        self.ended_early = False
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        self.episode_log = pd.DataFrame(columns=['avg_theta', 'avg_alpha', 'avg_theta_dot', 'avg_alpha_dot',
                                                 'max_theta', 'max_alpha', 'max_theta_dot', 'max_alpha_dot',
                                                 'sum_action', 'time_up', 'hit constraints', 'successful'])

    def update(self, state, action, reward, terminated):
        self.log.append({'theta': state[0], 'alpha': state[1], 'theta_dot': state[2], 'alpha_dot': state[3],
                         'action': action, 'reward': reward}, ignore_index=True)
        self.ended_early = terminated

    def calc(self):
        """
        calculates the average and max values for each state, the sum of all actions, and whether the episode
            ended due to time constraints or hitting the constraints and appends them to episode_log while resetting
            log.
        """
        alpha = self.log['alpha']
        time_up = (np.pi/2. > np.absolute(alpha)).sum()  # TODO verify
        successfull = np.pi/2. > np.absolute(np.max(alpha)) # TODO verify
        self.episode_log.append({'avg_theta': np.mean(self.log['theta']), 'avg_alpha': np.mean(self.log['alpha']),
                                    'avg_theta_dot': np.mean(self.log['theta_dot']),
                                    'avg_alpha_dot': np.mean(self.log['alpha_dot']),
                                    'max_theta': np.max(self.log['theta']), 'max_alpha': np.max(self.log['alpha']),
                                    'max_theta_dot': np.max(self.log['theta_dot']),
                                    'max_alpha_dot': np.max(self.log['alpha_dot']),
                                    'sum_action': np.sum(self.log['action']),
                                    'time_up': time_up,
                                    'hit constraints': self.ended_early,  # TODO verify
                                    'successful': self.log['theta'].size - self.log['theta'].count() == 0},
                                    ignore_index=True)
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        self.ended_early = False

    def plot_episode(self):
        plt.plot(self.log['states'])
        plt.plot(self.log['action'])
        plt.plot(self.log['reward'])
        plt.show()

    def plot_all(self, path):
        # histograms
        plt.hist(self.episode_log['avg_theta'])
        plt.hist(self.episode_log['avg_alpha'])
        plt.hist(self.episode_log['avg_theta_dot'])
        plt.hist(self.episode_log['avg_alpha_dot'])
        plt.hist(self.episode_log['max_theta'])
        plt.hist(self.episode_log['max_alpha'])
        plt.hist(self.episode_log['max_theta_dot'])
        plt.hist(self.episode_log['max_alpha_dot'])
        plt.hist(self.episode_log['sum_action'])
        plt.hist(self.episode_log['time_up'])
        # pie charts
        number_successful = self.episode_log['successful'].sum()
        number_hit_constraints = self.episode_log['hit constraints'].sum()
        plt.bar(['successful', 'hit constraints'], [number_successful, number_hit_constraints])
        # save
        plt.savefig(path) # TODO: verify and improve
        # show
        plt.show()

    def save(self, path):
        self.episode_log.to_csv(path)
