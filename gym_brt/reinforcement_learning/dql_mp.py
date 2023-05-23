""""
adapted from the deep q learning tutorial at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
changed: - the environment to the QubeSwingupEnv
"""
import time

# import gym_brt.envs.qube_swingup_env as qse
import gym
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv
from gym_brt.reinforcement_learning.data_collection import Log

# import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# quality of life
from tqdm import tqdm
import logging

# Multicore
import multiprocessing
from multiprocessing import Process

print("running torch version: ", torch.__version__)

parser = argparse.ArgumentParser()  # sets up argument parsing

# Hyperparameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1  # originally 0.05
EPS_DECAY = 10000  # originally 1000
TAU = 0.005  # for soft update of target parameters
LR = 1e-4

# settings
runtime_duration_tracking = False
# parsing of arguments
# TODO add sub arguments for Log
parser.add_argument(
    "-r", "--render",
    default=False, type=bool,
    help="toggles rendering of the simulation",
)
parser.add_argument(
    "-t", "--track",
    default=True, type=bool,
    help="toggles tracking simulated states",
)
parser.add_argument(
    "-l", "--load",
    default="", type=str,
    help="if a path is given, the model will be loaded from the given .pt file",
)
parser.add_argument(
    "-e", "--episodes",
    default=200, type=int,
    help="number of episodes to train the model for",
)
parser.add_argument(
    "-s", "--simulation",
    default=True, type=bool,
    help="toggles between simulation and hardware",
)
parser.add_argument(
    "-le", "--learn",
    default=True, type=bool,
    help="toggles learning",
)
parser.add_argument(
    "-se", "--save_episodes",
    default=False, type=bool,
    help="toggles saving of episodes as csv",
)

args, _ = parser.parse_known_args()

renderer = False  # args.render  # render each episode?
track = args.track
load = args.load != ""
path = args.load
num_episodes = args.episodes
simulation = False  # args.simulation
learn = args.learn
save_episodes = args.save_episodes
track_energy = track  # replace with own parameter?

if track:
    log = Log(save_episodes=save_episodes, track_energy=track_energy)

env = QubeSwingupDescActEnv(use_simulator=simulation)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# Get number of actions from gym action space
n_actions = env.action_space.n  # TODO: ensure compatability with discrete action spaces
# Get the number of state observations
state = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
if load:
    try:
        if torch.cuda.is_available():  # checks if running on gpu
            policy_net.load_state_dict(torch.load(path))
            target_net.load_state_dict(torch.load(path))
        else:  # ensures that state dict is loaded to cpu
            policy_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            target_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"File not found. Please check your Path: {path}.")
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold or not learn:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train(state_mp, reward_mp, done_mp, action_mp, t_mp, reward_episode_mp, reset_flag, stop_flag, state_prev_mp, step_flag):
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name()}, number of episodes: {num_episodes}")
    else:
        print(f"Running on CPU, number of episodes: {num_episodes}")

    # main training loop

    if track:
        min_alphas = []
        rewards = []
    # tracking best reward to continuously save the best model as a countermeasure to catastrophic forgetting
    top_reward = 0
    for i_episode in tqdm(range(num_episodes)):
        # initialize the environment and get it's state
        t_current = 0
        t_last = 0
        while True:
            if not step_flag.value:
                continue
            time_start = time.time()
            t_current = t_mp.value
            if t_current <= t_last:  # checks if in sync with hardware controller
                logging.warning(f"{t_current} steps info episode {i_episode}: {t_current - t_last - 1} samples have been missed!")

            action = select_action(state)
            action_mp.value = action.item()
            # # observation, reward, terminated, truncated, _ = env.step(action.item())
            # observation, reward, terminated, _ = env.step(action.item())  # match def of qube base env
            # # ~ state, reward, done, _ = env.step(action) from:
            # # https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/docs/alternatives.md#usage
            # reward = torch.tensor([reward_mp.value], device=device)
            # # qube base only uses done instead of differentiating between terminated and truncated
            # done = terminated  # or truncated
            #
            # if terminated:
            #     next_state = None
            #     print(f"finished after {t + 1} steps with x = {state}")  # print alpha at end of episode
            #
            # # renderer used in test.py
            # if renderer:
            #     env.render()
            # if track:
            #     # total_reward += reward.item()
            #     alpha.append(abs(observation[1]))

            next_state = torch.tensor(state_prev_mp.value, dtype=torch.float32, device=device).unsqueeze(0)

            # Todo: check if memory push is sensible here or in the hardware controller!!!
            # # Store the transition in memory
            # memory.push(state, action, next_state, reward)

            if learn:
                # Perform one step of the optimization (on the policy network)
                try:
                    # Perform one step of the optimization (on the policy network)
                    optimize_model()

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = target_net.state_dict()
                    policy_net_state_dict = policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                                1 - TAU)
                    target_net.load_state_dict(target_net_state_dict)

                except RuntimeError as e:
                    logging.error(e)
                    logging.warning("skipping optimization step due to error!")

            t_last = t_current  # update time step tracking

            time_end = time.time()
            logging.debug(f"hardware controller step took {(time_end - time_start) * 1000} ms")

            if done_mp.value:
                logging.info(f"Episode {i_episode} finished:\ttotal reward: {reward_episode_mp.value},\tduration: {t_mp.value}")
                break

        if (i_episode % int(num_episodes / 10) == 0 and i_episode != 0) or runtime_duration_tracking or i_episode == num_episodes - 1:

            print("plotting episode...")

            if i_episode == num_episodes - 1:

                print(f'finished training after {i_episode + 1} episodes')
                if learn:
                    torch.save(policy_net.state_dict(), 'trained_models/model.pt')
                log.save()
                stop_flag.value = True
            elif learn:
                # save backup of net:
                torch.save(policy_net.state_dict(), f'trained_models/model_in_e={i_episode}.pt')

            if track:
                log.plot_episode()  # TODO: test

        if reward_episode_mp.value > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
            logging.info(f"new best performance: {reward_episode_mp.value}, saving model...")
            top_reward = reward_episode_mp.value
            torch.save(policy_net.state_dict(), f'trained_models/dql_best_performance.pt')


def hardware_controller(state_mp, reward_mp, done_mp, action_mp, t_mp, reward_episode_mp, reset_flag, stop_flag, state_prev_mp, step_flag):
    try:
        while not stop_flag.value:
            time_start = time.time()
            state_prev_mp.value = state_mp.value
            state_mp.value, reward_mp.value, done_mp.value, _ = env.step(action_mp.value)
            step_flag.value = True
            # tracking using datalogger
            if track:
                log.update(state_mp.value, action_mp.value, reward_mp.value, done_mp.value)
            # Store the transition in memory
            # TODO: check if this is sensible here or in the training loop!!!
            # this way no state action paired are lost due to missed samples
            memory.push(state_prev_mp.value, action_mp.value, state_mp.value, reward_mp.value)
            # running totals
            reward_episode_mp.value += reward_mp.value
            t_mp.value += 1
            time_end = time.time()
            logging.debug(f"hardware controller step took {(time_end - time_start) * 1000} ms")
            if done_mp.value or reset_flag.value:  # TODO: is reset_flag redundant?
                # Todo: maybe replace with separate reset function?
                t_mp.value = 0
                reward_episode_mp.value = 0
                state_mp.value = env.reset()
                reset_flag.value = False
                if track:
                    log.calc()
                break
        logging.info("hardware controller stopped")
    finally:
        env.close()


def control():
    if simulation:
        train()
    else:
        # shared variables
        state_mp = multiprocessing.Array('d', [0, 0, 0, 0])  # TODO: add alternative start state for balance
        state_prev_mp = multiprocessing.Array('d', [0, 0, 0, 0])
        # ensure that state and state_prev are correctly initialized
        state_prev_mp.value = env.reset()
        state_mp.value = state_prev_mp.value
        reward_mp = multiprocessing.Value('d', 0)
        done_mp = multiprocessing.Value('b', False)
        action_mp = multiprocessing.Value('d', 0)
        t_mp = multiprocessing.Value('i', 0)
        # cumulative
        reward_episode_mp = multiprocessing.Value('d', 0)
        # flags
        stop_flag = multiprocessing.Value('b', False)  # tells hardware controller to stop, after training done
        reset_flag = multiprocessing.Value('b', False)  # tells hardware controller to reset, after episode done
        step_flag = multiprocessing.Value('b', True)  # tells training loop that new step is ready

        # individual processes, initialized with shared variables
        training_process = Process(target=train, args=(state_mp, reward_mp, done_mp, action_mp, t_mp,
                                                       reset_flag, stop_flag, state_prev_mp, step_flag))
        controller_process = Process(target=hardware_controller, args=(state_mp, reward_mp, done_mp, action_mp, t_mp,
                                                                       reset_flag, stop_flag, state_prev_mp, step_flag))
        # start processes
        training_process.start()
        controller_process.start()
        # wait for processes to finish
        training_process.join()
        controller_process.join()


if __name__ == '__main__':
    control()
    logging.info("all processes finished, exiting...")
