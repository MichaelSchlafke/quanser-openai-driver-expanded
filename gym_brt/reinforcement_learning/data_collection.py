# temp fix for pandas dep. warning:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch

# TODO: TEST
# TODO: Add "Einschwingzeit" + "Überschwingen" + "Ausregelzeit" logging


def energy(state):  # TODO: TEST
    """
    calculates the energy of the system
    :param state: state of the system
    :return: kinetic energy, potential energy, total energy
    """
    g = 9.81  # acceleration due to gravity in m/s^2
    # constants taken from the Qube Servo 2 user manual
    J = 4e-6  # inertia of the pendulum in kg*m^2  # TODO correct value is unknown!!!
    m_p = 0.024  # mass of the pendulum in kg
    l = 0.129  # length of the pendulum in m
    E_kin = 0.5 * J * state[3] ** 2
    E_pot = m_p * g * l * (1 - np.cos(state[1] - np.pi))
    return E_kin, E_pot, E_kin + E_pot


class Log:
    """
    Log class for logging data from the environment
    Dataframes:
        - log contains all sates, actions, and rewards for each time step in the current episode
        - episode_log contains the average and max values for each state, the sum of all actions, and whether the episode
            ended due to time constraints or hitting the constraints
    """
    def __init__(self, save_episodes=False, track_energy=True, path=""):
        self.ended_early = False
        self.save_episodes = save_episodes
        self.track_energy = track_energy
        if path == "":
            self.path = os.getcwd()
            print(f"path set to {self.path}")
        else:
            self.path = path
        self.i = 0
        self.t = 0
        self.rise_time = np.nan
        self.overshoot = np.nan
        self.settling_time = np.nan
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        self.episode_log = pd.DataFrame(columns=['avg_theta', 'avg_alpha', 'avg_theta_dot', 'avg_alpha_dot',
                                                 'max_theta', 'max_alpha', 'max_theta_dot', 'max_alpha_dot',
                                                 'sum_action', 'time_up', 'hit constraints', 'reward', 'successful'])

    def update(self, state, action, reward, terminated):
        self.t += 1  # tracks time
        # convert from tensor to numpy array
        action = action.cpu()
        action = action.detach().numpy()
        reward = reward.cpu()
        reward = reward.detach().numpy()
        state = state.cpu()
        state = state.detach().numpy()
        state = state[0, :]  # converts from 2D array to 1D array

        # calculate and save energy over time
        if self.track_energy:
            E_kin, E_pot, E_tot = energy(state)
            self.log = self.log.append(
                {'theta': state[0], 'alpha': state[1], 'theta_dot': state[2], 'alpha_dot': state[3],
                 'action': action, 'reward': reward, 'E_kin': E_kin, 'E_pot': E_pot, 'E_tot': E_tot}, ignore_index=True)
        else:
            self.log = self.log.append(
                {'theta': state[0], 'alpha': state[1], 'theta_dot': state[2], 'alpha_dot': state[3],
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
        try:
            if not np.isnan(self.rise_time):  # values only make sense if the pendulum has risen within target range (10%)
                alphas = self.log['alpha'].to_numpy()
                self.overshoot = np.max(np.absolute(alphas[self.rise_time:]))  # given as a percentage  # TODO verify
                if np.absolute(alphas[-1]) < 0.05 * np.pi:  # ensures stationary target value reached
                    target_met = np.argwhere(np.absolute(alphas[self.rise_time:]) < 0.05 * np.pi)[0][0]  # TODO verify
                    for index in range(target_met.size):
                        # traverses array to find the earliest time target value was reached continuously  # TODO test rigerously
                        if target_met[target_met.size - 1 - index] != target_met[target_met.size - 2 - index] and index > 5:
                            # TODO is 5 a good value?
                            self.settling_time = target_met.size - index
        except:  # TODO: improve
            self.overshoot = np.nan
            self.settling_time = np.nan
            print("calculation of overshoot and settling time failed critically!")
        alpha = self.log['alpha'].to_numpy()
        time_up = (np.pi/2. > np.absolute(alpha)).sum()  # TODO verify
        successfull = np.pi/2. > np.absolute(np.max(alpha)) # TODO verify
        self.episode_log = self.episode_log.append({'avg_theta': np.mean(self.log['theta']), 'avg_alpha': np.mean(self.log['alpha']),
                                    'avg_theta_dot': np.mean(self.log['theta_dot']),
                                    'avg_alpha_dot': np.mean(self.log['alpha_dot']),
                                    'max_theta': np.max(self.log['theta']), 'max_alpha': np.max(self.log['alpha']),
                                    'max_theta_dot': np.max(self.log['theta_dot']),
                                    'max_alpha_dot': np.max(self.log['alpha_dot']),
                                    'sum_action': np.sum(self.log['action']),
                                    'time_up': time_up,
                                    'hit constraints': self.ended_early,  # TODO FIX THIS, ALWAYS TRUE!
                                    'reward': self.log['reward'].sum(),
                                    'successful': successfull,
                                    'rise_time': self.rise_time,
                                    'overshoot': self.overshoot,
                                    'settling_time': self.settling_time},
                                    ignore_index=True)
        # safe to csv
        if self.save_episodes:
            self.log.to_csv(os.path.join(self.path, f'episode{self.i}.csv'))
        self.log = pd.DataFrame(columns=['theta', 'alpha', 'theta_dot', 'alpha_dot', 'action', 'reward'],)
        # reset values
        self.t = 0
        self.rise_time = np.nan
        self.overshoot = np.nan
        self.settling_time = np.nan
        self.ended_early = False

    def plot_episode(self):
        plt.plot(self.log['theta'], label='theta')
        plt.plot(abs(self.log['alpha']), label='|alpha|')
        plt.plot(self.log['theta_dot'], label='theta_dot')
        plt.plot(self.log['alpha_dot'], label='alpha_dot')
        plt.legend()
        plt.ylabel('states in rad or rad/s')
        plt.xlabel('time in ms')
        plt.title(f'states over time in episode {self.i}')
        plt.show()  # TODO add dual axis for angle and angular velocity
        if self.track_energy:
            plt.plot(self.log['E_kin'], label='E_kin')
            plt.plot(self.log['E_pot'], label='E_pot')
            plt.plot(self.log['E_tot'], label='E_tot')
            plt.legend()
            plt.ylabel('energy in J')
            plt.xlabel('time in ms')
            plt.title(f'energy over time in episode {self.i}')
            plt.show()
        # TODO add options to save
        # plt.scatter(np.linspace(0, self.log['action'].size, self.log['action'].size), self.log['action'])
        # plt.title('actions')
        # plt.show()
        # reward loging only sensible if variable with time
        # plt.plot(self.log['reward'])
        # plt.title('rewards')
        # plt.show()

    def plot_all(self):
        # TODO fix only fist histogram is shown!
        print(f"plotting statistical data collected over {self.i} episodes")
        plt.ioff()
        # histograms
        plt.hist(self.episode_log['avg_theta'])
        plt.title('avg_theta')
        plt.figure(1)
        plt.show()
        plt.hist(self.episode_log['avg_alpha'])
        plt.title('avg_alpha')
        plt.figure(2)
        plt.show()
        plt.hist(self.episode_log['avg_theta_dot'])
        plt.title('avg_theta_dot')
        plt.figure(3)
        plt.show()
        plt.hist(self.episode_log['avg_alpha_dot'])
        plt.title('avg_alpha_dot')
        plt.figure(4)
        plt.show()
        plt.hist(self.episode_log['max_theta'])
        plt.title('max_theta')
        plt.figure(5)
        plt.show()
        plt.hist(self.episode_log['max_alpha'])
        plt.title('max_alpha')
        plt.figure(6)
        plt.show()
        plt.hist(self.episode_log['max_theta_dot'])
        plt.title('max_theta_dot')
        plt.figure(7)
        plt.show()
        plt.hist(self.episode_log['max_alpha_dot'])
        plt.title('max_alpha_dot')
        plt.figure(8)
        plt.show()
        plt.hist(self.episode_log['sum_action'])
        plt.title('sum_action')
        plt.figure(9)
        plt.show()
        # if self.episode_log['time_up'].nonzero()[0].size > 0:
        plt.hist(self.episode_log['time_up'])
        plt.title('time_up')
        plt.figure(10)
        plt.show()
        # TODO: handle NaN values properly
        # if np.isfinite(self.episode_log['overshoot'].to_numpy()):
        #     plt.hist(self.episode_log['overshoot'])
        #     plt.title('overshoot')
        # if np.isfinite(self.episode_log['settling_time']):
        #     plt.hist(self.episode_log['settling_time'].to_numpy())
        #     plt.title('settling_time')
        # if np.isfinite(self.episode_log['rise_time']):
        #     plt.hist(self.episode_log['rise_time'].to_numpy())
        #     plt.title('rise_time')
        # pie charts
        number_successful = self.episode_log['successful'].sum()
        number_hit_constraints = self.episode_log['hit constraints'].sum()
        plt.bar(['successful', 'hit constraints'], [number_successful, number_hit_constraints])
        plt.title('success rates')
        plt.figure(11)
        plt.show()
        print("finished plotting")
        # TODO save figs

    def save(self):
        self.episode_log.to_csv(os.path.join(self.path, 'collected_data.csv'))
