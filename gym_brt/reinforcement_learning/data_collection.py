# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# TODO: TEST
# TODO: Add energy logging
# TODO: Add "Einschwingzeit" + "Überschwingen" + "Ausregelzeit" logging


class Log:
    """
    Log class for logging data from the environment
    Dataframes:
        - log contains all sates, actions, and rewards for each time step in the current episode
        - episode_log contains the average and max values for each state, the sum of all actions, and whether the episode
            ended due to time constraints or hitting the constraints
    """
    def __init__(self, safe_episodes=False):
        self.ended_early = False
        self.safe_episodes = safe_episodes
        self.i = 0
        self.t = 0
        self.rise_time = np.nan
        self.overshoot = np.nan
        self.settling_time = np.nan
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        self.episode_log = pd.DataFrame(columns=['avg_theta', 'avg_alpha', 'avg_theta_dot', 'avg_alpha_dot',
                                                 'max_theta', 'max_alpha', 'max_theta_dot', 'max_alpha_dot',
                                                 'sum_action', 'time_up', 'hit constraints', 'successful'])

    def update(self, state, action, reward, terminated):
        self.t += 1
        self.log.append({'theta': state[0], 'alpha': state[1], 'theta_dot': state[2], 'alpha_dot': state[3],
                         'action': action, 'reward': reward}, ignore_index=True)
        # detect characteristics
        if abs(state[1]) < np.pi * 0.1 and np.isnan(self.rise_time):
            self.rise_time = self.t  # TODO verify
        self.ended_early = terminated

    def calc(self):
        """
        calculates the average and max values for each state, the sum of all actions, and whether the episode
            ended due to time constraints or hitting the constraints and appends them to episode_log while resetting
            log.
        """
        self.i += 1
        # calculation
        if not np.isnan(self.rise_time):  # values only make sense if the pendulum has risen within target range (10%)
            alphas = self.log['alpha']
            self.overshoot = np.max(np.absolute(alphas[self.rise_time:]))  # given as a percentage  # TODO verify
            if np.absolute(alphas[-1]) < 0.05 * np.pi:  # ensures stationary target value reached
                target_met = np.argwhere(np.absolute(alphas[self.rise_time:]) < 0.05 * np.pi)[0][0]  # TODO verify
                for index in range(target_met.size):
                    # traverses array to find the earliest time target value was reached continuously  # TODO test rigerously
                    if target_met[target_met.size - 1 - index] != target_met[target_met.size - 2 - index] and index > 5:
                        # TODO is 5 a good value?
                        self.settling_time = target_met.size - index
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
                                    'successful': successfull,
                                    'rise_time': self.rise_time,
                                    'overshoot': self.overshoot,
                                    'settling_time': self.settling_time},
                                    ignore_index=True)
        # safe to csv
        if safe_episodes:
            self.log.to_csv(path+f'/episode{i}.csv')
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        # reset values
        self.t = 0
        self.rise_time = np.nan
        self.overshoot = np.nan
        self.settling_time = np.nan
        self.ended_early = False

    def plot_episode(self):
        plt.plot(self.log['states'])
        plt.plot(self.log['action'])
        plt.plot(self.log['reward'])
        plt.show()

    def plot_all(self, path):
        # histograms
        plt.hist(self.episode_log['avg_theta'])
        plt.title('avg_theta')
        plt.hist(self.episode_log['avg_alpha'])
        plt.title('avg_alpha')
        plt.hist(self.episode_log['avg_theta_dot'])
        plt.title('avg_theta_dot')
        plt.hist(self.episode_log['avg_alpha_dot'])
        plt.title('avg_alpha_dot')
        plt.hist(self.episode_log['max_theta'])
        plt.title('max_theta')
        plt.hist(self.episode_log['max_alpha'])
        plt.title('max_alpha')
        plt.hist(self.episode_log['max_theta_dot'])
        plt.title('max_theta_dot')
        plt.hist(self.episode_log['max_alpha_dot'])
        plt.title('max_alpha_dot')
        plt.hist(self.episode_log['sum_action'])
        plt.title('sum_action')
        plt.hist(self.episode_log['time_up'])
        plt.title('time_up')
        plt.hist(self.episode_log['overshoot'])
        plt.title('overshoot')
        plt.hist(self.episode_log['settling_time'])
        plt.title('settling_time')
        plt.hist(self.episode_log['rise_time'])
        plt.title('rise_time')
        # pie charts
        number_successful = self.episode_log['successful'].sum()
        number_hit_constraints = self.episode_log['hit constraints'].sum()
        plt.bar(['successful', 'hit constraints'], [number_successful, number_hit_constraints])
        plt.title('success rates')
        # save
        plt.savefig(path) # TODO: verify and improve
        # show
        plt.show()

    def save(self, path):
        self.episode_log.to_csv(path+'/collected_data.csv')
