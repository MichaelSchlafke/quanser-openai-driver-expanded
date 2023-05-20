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

# usable implementations from non multiprocessing version
from dql import *


def dql(state, reward, action, sender, stop_flag):
    """
    Deep Q-Learning, modified to use multiprocessing for communication between the training process and the hardware.
    """
    # hyperparameters
    TAU = 0.005  # for soft update of target parameters

    if torch.cuda.is_available():
        print(f"Running on GPU: {torch.cuda.get_device_name()}, number of episodes: {num_episodes}")
    else:
        print(f"Running on CPU, number of episodes: {num_episodes}")

    # main training loop
    try:  # ensures environment closes to not brick the board
        if track:
            min_alphas = []
            rewards = []
        for i_episode in tqdm(range(num_episodes)):
            # initialize the environment and get it's state
            # state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if track:
                total_reward = 0
                alpha = []
            for t in count():
                action = select_action(state)
                sender.send(action)
                # observation, reward, terminated, truncated, _ = env.step(action.item())
                # observation, reward, terminated, _ = env.step(action.item())  # match def of qube base env
                # ~ state, reward, done, _ = env.step(action) from:
                # https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/docs/alternatives.md#usage
                # reward = torch.tensor([reward], device=device)
                # qube base only uses done instead of differentiating between terminated and truncated
                # done = terminated  # or truncated

                if stop_flag.value:
                    next_state = None
                    print(f"finished after {t + 1} steps with x = {state}")  # print alpha at end of episode

                # renderer used in test.py
                if renderer:
                    env.render()
                # if track:
                #     # total_reward += reward.item()
                #     alpha.append(abs(observation[1]))

                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # if track:
                #     log.update(state, action[0, 0], reward[0], done)

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

                if done:
                    episode_durations.append(t + 1)
                    # plot_durations()  # only sensible if episode limit is reached
                    break

            if (i_episode % int(
                    num_episodes / 10) == 0 and i_episode != 0) or runtime_duration_tracking or i_episode == num_episodes - 1:

                # print("plotting episode...")
                # plot_durations(show_result=False)  # made redundant by log.plot_episode()
                if i_episode == num_episodes - 1:
                    # plt.savefig('result.png')
                    print('finished training')
                    if learn:
                        torch.save(policy_net.state_dict(), 'trained_models/model.pt')
                    # log.save()
                elif learn:
                    # save backup of net:
                    torch.save(policy_net.state_dict(), f'trained_models/model_in_e={i_episode}.pt')
                # plt.ioff()
                # plt.show()
                # if track:
                #     log.plot_episode()  # TODO: test

            if total_reward > top_reward * 1.1:  # extra 10% to reduce unnecessary saving overhead
                top_reward = total_reward
                torch.save(policy_net.state_dict(), f'trained_models/dql_best_performance.pt')
            if track:
                print(f"total reward: {total_reward}")
                log.calc()
                # rewards.append(total_reward)
                # min_alphas.append(min(alpha))

        if track:
            log.plot_all()

    finally:
        env.close()


def reinforce():
    raise NotImplementedError


def actor_critic():
    raise NotImplementedError
