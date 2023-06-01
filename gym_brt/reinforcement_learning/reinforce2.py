import sys
import torch
import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
# custom classes
from gym_brt.envs.qube_swingup_custom_env import QubeSwingupDescActEnv, QubeSwingupStatesSquaredEnvDesc
from gym_brt.reinforcement_learning.data_collection import Log
# QoL imports
from tqdm import tqdm

# Constants
GAMMA = 0.9

# settings
track = True
save_episodes = False
track_energy = False
render = False

if track:
    log = Log(save_episodes=save_episodes, track_energy=track_energy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state = state.to(device)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.cpu().detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def main():
    env = QubeSwingupDescActEnv(use_simulator=True, )
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 512).to(device)

    max_episode_num = 10000
    max_steps = 10000
    numsteps = []
    avg_numsteps = []
    all_rewards = []
    best_reward = 1

    for episode in tqdm(range(max_episode_num)):
        state = env.reset()
        log_probs = []
        rewards = []
        ep_reward = 0

        for steps in range(max_steps):
            if render:
                env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log.update(new_state, action, reward, done)
            log_probs.append(log_prob)
            rewards.append(reward)
            ep_reward += reward

            if done:
                update_policy(policy_net, rewards, log_probs)
                numsteps.append(steps)
                avg_numsteps.append(np.mean(numsteps[-10:]))
                all_rewards.append(np.sum(rewards))
                log.calc()
                if ep_reward > best_reward * 1.05:
                    print(f"new best reward of {ep_reward} in episode {episode}, beating {best_reward}")
                    best_reward = ep_reward
                    torch.save(policy_net.state_dict(), f'./trained_models/REINFORCE_best_rew.pth')
                if episode % 1 == 0:
                    sys.stdout.write(
                        "episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(
                            np.sum(rewards), decimals=3), np.round(np.mean(all_rewards[-10:]), decimals=3), steps))
                if episode % 1000 == 0:
                    torch.save(policy_net.state_dict(), f'./trained_models/REINFORCE_e={episode}.pth')
                    log.save()
                if episode == max_episode_num - 1:
                    torch.save(policy_net.state_dict(), './trained_models/REINFORCE.pth')
                break

            state = new_state

    plt.plot(numsteps)
    plt.plot(avg_numsteps)
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f"Running REINFORCE on GPU: {torch.cuda.get_device_name()}")
    else:
        print(f"Running REINFORCE on CPU")
    main()