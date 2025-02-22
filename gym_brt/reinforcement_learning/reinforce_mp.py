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
# custom classes
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv
from gym_brt.reinforcement_learning.data_collection import Log


parser = argparse.ArgumentParser(description='PyTorch REINFORCE')

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

# env.reset(seed=args.seed)  # TODO: keep or remove?
env.reset()
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
if path != "":
    try:
        policy.load_state_dict(torch.load(path))
    except FileNotFoundError:
        print("File not found. Training new model.")
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


def train(state_mp, reward_mp, done_mp, action_mp, t_mp, reward_episode_mp, reset_flag, stop_flag, state_prev_mp, step_flag):
    # tracking best reward to continuously save the best model as a countermeasure to catastrophic forgetting
    top_reward = 0
    running_reward = 10
    for i_episode in count(1):
        t_last = 0
        # state = env.reset()
        # ep_reward = 0
        while True:
            if not step_flag.value:
                continue
            time_start = time.time()
            t_current = t_mp.value
            if t_current <= t_last:  # checks if in sync with hardware controller
                logging.warning(f"{t_current} steps info episode {i_episode}: {t_current - t_last - 1} samples have been missed!")

            action_mp.value = select_action(state_mp.value)
            # state, reward, done, _ = env.step(action)
            policy.rewards.append(reward_mp.value)

            t_last = t_current  # update time step tracking

            time_end = time.time()
            logging.debug(f"hardware controller step took {(time_end - time_start) * 1000} ms")
            if renderer:
                env.render()
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if i_episode % 100 == 0:
            torch.save(policy.state_dict(), f'trained_models/reinforce_e={i_episode}.pt')
        elif ep_reward > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
            top_reward = ep_reward
            torch.save(policy.state_dict(), f'trained_models/resolve_best_performance.pt')
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


def hardware_controller(state_mp, reward_mp, done_mp, action_mp, t_mp, reward_episode_mp, reset_flag, stop_flag,
                        state_prev_mp, step_flag):
    try:
        while not stop_flag.value:
            time_start = time.time()
            state_prev_mp.value = state_mp.value
            state_mp.value, reward_mp.value, done_mp.value, _ = env.step(action_mp.value)
            step_flag.value = True
            # tracking using datalogger
            if track:
                log.update(state_mp.value, action_mp.value, reward_mp.value, done_mp.value)

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
