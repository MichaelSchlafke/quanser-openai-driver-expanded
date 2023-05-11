import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# custom classes
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv
from gym_brt.reinforcement_learning.data_collection import Log


# settings
runtime_duration_tracking = False
# in cartpole this equates to the number if frames the pole has to be raised for the task to be completed
# with cube base environment, this is not applicable as the reward is not sparse
reward_threshold = 5000  # TODO: find and verify sensible value, maybe replace altogether?
# parsing of arguments
parser = argparse.ArgumentParser(description='PyTorch actor-critic')
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
    default=False, type=bool,
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
    '--log-interval', type=int, default=10, metavar='N',
    help='interval between training status logs (default: 10)'
)

args, _ = parser.parse_known_args()

renderer = args.render  # render each episode?
track = args.track
load = args.load != ""
path = args.load
num_episodes = args.episodes
simulation = args.simulation
simulation = False
learn = args.learn
save_episodes = args.save_episodes
track_energy = track  # replace with own parameter?

if track:
    log = Log(save_episodes=save_episodes, track_energy=track_energy)

# set up environment
env = QubeSwingupDescActEnv(use_simulator=simulation)
# env.reset(seed=args.seed)  # TODO: keep or remove?
env.reset()
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 512)

        # actor's layer
        self.action_head = nn.Linear(512, 3)

        # critic's layer
        self.value_head = nn.Linear(512, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
if path != "":
    try:
        model.load_state_dict(torch.load(path))
    except FileNotFoundError:
        print("File not found. Training new model.")
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    try:
        # tracking best reward to continuously save the best model as a countermeasure to catastrophic forgetting
        top_reward = 0
        running_reward = 10

        # run infinitely many episodes
        for i_episode in count(1):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in range(1, 10000):

                # select action from policy
                action = select_action(state)

                # take the action
                state, reward, done, _ = env.step(action)

                if args.render:
                    env.render()

                model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            finish_episode()

            # log results
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                      i_episode, ep_reward, running_reward))

            if i_episode % 100 == 0:
                torch.save(model.state_dict(), f'trained_models/actor-critic_e={i_episode}.pt')
            elif ep_reward > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
                top_reward = ep_reward
                torch.save(model.state_dict(), f'trained_models/actor-critic_best_performance.pt')

            # check if we have "solved" the cart pole problem
            if running_reward > reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break
    finally:
        env.close()

if __name__ == '__main__':
    main()
