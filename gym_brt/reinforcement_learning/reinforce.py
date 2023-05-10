import argparse
import gym
import numpy as np
from itertools import count
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# settings
runtime_duration_tracking = False
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

if track:
    log = Log(save_episodes=save_episodes, track_energy=track_energy)

env = QubeSwingupDescActEnv(use_simulator=simulation)

env.reset(seed=args.seed)  # TODO: keep or remove?
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 512)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(512, 3)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
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


def main():
    running_reward = 10
    for i_episode in count(1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            policy.rewards.append(reward)
            ep_reward += reward
            if renderer:
                env.render()
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
