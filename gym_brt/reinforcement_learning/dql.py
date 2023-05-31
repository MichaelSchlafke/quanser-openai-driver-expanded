""""
adapted from the deep q learning tutorial at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
changed: - the environment to the QubeSwingupEnv
"""

# import gym_brt.envs.qube_swingup_env as qse
import gym
from gym_brt.envs.qube_swingup_env import QubeSwingupEnv
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv, QubeSwingupStatesSquaredEnvDesc, QubeOnlySwingupDescActEnv
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
import time as timer
import logging

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
TAU = 0.005 # for soft update of target parameters
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
parser.add_argument(
    "-R", "--Reward",
    default="original", type=str,
    help="choose reward function",
    choices=["original", "state_diff", "lqr", "original+onlySwingUp"],
)

args, _ = parser.parse_known_args()

renderer = False # args.render  # render each episode?
track = args.track
load = args.load != ""
path = args.load
num_episodes = args.episodes
simulation = True # args.simulation
learn = args.learn
learn = True
save_episodes = args.save_episodes
save_episodes = False
track_energy = track  # replace with own parameter?
reward_f = args.Reward

if track:
    log = Log(save_episodes=save_episodes, track_energy=track_energy)

if reward_f == "original":
    env = QubeSwingupDescActEnv(use_simulator=simulation)
elif reward_f == "state_diff":
    env = QubeSwingupStatesSquaredEnvDesc(use_simulator=simulation)
elif reward_f == "original+onlySwingUp":
    env = QubeOnlySwingupDescActEnv(use_simulator=simulation)
else:
    logging.error(f"reward function {reward_f} not implemented")

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


def train():
    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name()}, number of episodes: {num_episodes}")
    else:
        print(f"Running on CPU, number of episodes: {num_episodes}")

    # main training loop
    try:  # ensures environment closes to not brick the board
        if track:
            min_alphas = []
            rewards = []
        # tracking best reward to continuously save the best model as a countermeasure to catastrophic forgetting
        top_reward = 0
        for i_episode in tqdm(range(num_episodes)):
            # initialize the environment and get it's state
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if track:
                total_reward = 0
                alpha = []
            for t in count():
                timings = f"timings in step {t}:"
                t1 = timer.time()
                action = select_action(state)
                t2 = timer.time()
                timings += f"\n\t- model evaluation for action selection took {(t2 - t1) * 1000}ms"
                # observation, reward, terminated, truncated, _ = env.step(action.item())
                observation, reward, terminated, _ = env.step(action.item())  # match def of qube base env
                t3 = timer.time()
                timings += f"\n\t- environment step took {(t3 - t2) * 1000}ms"
                # ~ state, reward, done, _ = env.step(action) from:
                # https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/docs/alternatives.md#usage
                reward = torch.tensor([reward], device=device)
                # qube base only uses done instead of differentiating between terminated and truncated
                done = terminated  # or truncated

                if terminated:
                    next_state = None
                    timings += f"\n\t- finished after {t + 1} steps with x = {state}"  # print alpha at end of episode

                # renderer used in test.py
                if renderer:
                    env.render()
                if track:
                    total_reward += reward.item()
                    alpha.append(abs(observation[1]))

                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory

                memory.push(state, action, next_state, reward)
                t4 = timer.time()
                timings += f"\n\t- saving to replay buffer took {(t4 - t3) * 1000}ms"

                if track:
                    log.update(state, action[0, 0], reward[0], done)

                # Move to the next state
                state = next_state

                if learn:
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

                    t5 = timer.time()
                    timings += f"\n\t- optimizing the model {(t5 - t4) * 1000}ms"

                t6 = timer.time()
                if (t6 - t1) * 1000 > 4 and not simulation:
                    logging.warning(f"total time of episode: {(t6 - t1) * 1000 }ms exeeds 4ms budget!\n" + timings)
                else:
                    logging.debug(timings)

                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()  # only sensible if episode limit is reached
                    break

            if (i_episode % int(
                    num_episodes / 10) == 0 and i_episode != 0) or runtime_duration_tracking or i_episode == num_episodes - 1:

                print("plotting episode...")
                # plot_durations(show_result=False)  # made redundant by log.plot_episode()
                if i_episode == num_episodes - 1:
                    # plt.savefig('result.png')
                    print('finished training')
                    if learn:
                        torch.save(policy_net.state_dict(), 'trained_models/model.pt')
                    log.save()
                elif learn:
                    # save backup of net:
                    torch.save(policy_net.state_dict(), f'trained_models/model_in_e={i_episode}.pt')
                # plt.ioff()
                # plt.show()
                if track:
                    log.plot_episode()  # TODO: test

            if total_reward > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
                top_reward = total_reward
                torch.save(policy_net.state_dict(), f'trained_models/dql_best_performance.pt')
                torch.save(target_net.state_dict(), f'trained_models/dql_best_performance_state_dict.pt')
            if track:
                print(f"total reward: {total_reward}")
                log.calc()
                # rewards.append(total_reward)
                # min_alphas.append(min(alpha))

        if track:
            log.plot_all()

    finally:
        env.close()


if __name__ == '__main__':
    train()
