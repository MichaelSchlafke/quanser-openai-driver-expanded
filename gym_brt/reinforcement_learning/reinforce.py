import argparse
import logging

import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# custom classes
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv, QubeSwingupStatesSquaredEnvDesc, \
    QubeOnlySwingupDescActEnv
from gym_brt.reinforcement_learning.data_collection import Log


parser = argparse.ArgumentParser(description='PyTorch REINFORCE')

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# settings
runtime_duration_tracking = False
# in cartpole this equates to the number if frames the pole has to be raised for the task to be completed
# with cube base environment, this is not applicable as the reward is not sparse
reward_threshold = 5000  # TODO: find and verify sensible value, maybe replace altogether?
# parsing of arguments
#TODO add sub arguments for Log
parser.add_argument(
    "-r", "--render",
    default=False, type=bool,
    help="toggles rendering of the simulation",
)
parser.add_argument(
    '--gamma', type=float, default=0.99, metavar='G',
    help='discount factor (default: 0.99)'
)
parser.add_argument(
    '--seed', type=int, default=543, metavar='N',
    help='random seed (default: 543)'
)
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
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

renderer = args.render  # render each episode?
track = args.track
load = args.load != ""
path = args.load
num_episodes = args.episodes
simulation = args.simulation
learn = args.learn
save_episodes = args.save_episodes
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

# env.reset(seed=args.seed)  # TODO: keep or remove?
env.reset()
torch.manual_seed(args.seed)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 512)
        # self.affine2 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(512, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = x.to(device)
        x = self.affine1(x)
        # x = self.affine2(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine3(x)
        print(x)
        return F.softmax(action_scores, dim=1), x


policy = Policy().to(device)
policy = policy.float()
if path != "":
    try:
        if torch.cuda.is_available():  # checks if running on gpu
            policy.load_state_dict(torch.load(path))
        else:  # ensures that state dict is loaded to cpu
            policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print("File not found. Training new model.")
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    if type(state) == np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0)
    probs, _ = policy(state.float())
    m = Categorical(probs)
    action = m.sample().to(device)
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def train():
    # tracking best reward to continuously save the best model as a countermeasure to catastrophic forgetting
    top_reward = 0
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if track:
                log.update(state, action, reward, done)
            state = torch.tensor(state, device=device)
            reward = torch.tensor(reward, device=device)
            done = torch.tensor(done, device=device)
            policy.rewards.append(reward)
            ep_reward += reward
            if renderer:
                env.render()
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if track:
            log.calc()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if i_episode % 100 == 0:
            torch.save(policy.state_dict(), f'trained_models/reinforce_e={i_episode}.pt')
            log.save()
        elif ep_reward > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
            top_reward = ep_reward
            torch.save(policy.state_dict(), f'trained_models/resolve_best_performance.pt')
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    train()
