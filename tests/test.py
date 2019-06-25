from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import argparse
import numpy as np

from gym_brt.envs import QubeBeginDownEnv, QubeBeginUprightEnv

from gym_brt.control import (
    zero_policy,
    constant_policy,
    random_policy,
    square_wave_policy,
    energy_control_policy,
    pd_control_policy,
    flip_and_hold_policy,
    square_wave_flip_and_hold_policy,
    dampen_policy,
)


def print_info(state, action, reward):
    theta_x, theta_y = state[0], state[1]
    alpha_x, alpha_y = state[2], state[3]
    theta = np.arctan2(theta_y, theta_x)
    alpha = np.arctan2(alpha_y, alpha_x)
    theta_dot, alpha_dot = state[4], state[5]
    print(
        "State: theta={:06.3f}, alpha={:06.3f}, theta_dot={:06.3f}, alpha_dot={:06.3f}".format(
            theta, alpha, theta_dot, alpha_dot
        )
    )
    print("Action={}, Reward={}".format(action, reward))


def test_env(
    env_name,
    policy,
    num_episodes=10,
    num_steps=250,
    frequency=250,
    state_keys=None,
    verbose=False,
    use_simulator=False,
    render=False,
):

    with env_name(use_simulator=use_simulator, frequency=frequency) as env:
        for episode in range(num_episodes):
            state = env.reset()
            # print("Started episode {}".format(episode))
            for step in range(num_steps):
                # print("step {}.{}".format(episode, step))
                action = policy(state, step=step, frequency=frequency)
                state, reward, done, _ = env.step(action)
                if done:
                    pass  # break
                if verbose:
                    print_info(state, action, reward)
                if render:
                    env.render()


def main():
    envs = {"down": QubeBeginDownEnv, "up": QubeBeginUprightEnv}
    policies = {
        "none": zero_policy,
        "zero": zero_policy,
        "const": constant_policy,
        "rand": random_policy,
        "random": random_policy,
        "sw": square_wave_policy,
        "energy": energy_control_policy,
        "energy": energy_control_policy,
        "pd": pd_control_policy,
        "hold": pd_control_policy,
        "flip": flip_and_hold_policy,
        "sw-hold": square_wave_flip_and_hold_policy,
        "damp": dampen_policy,
    }

    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        default="down",
        choices=list(envs.keys()),
        help="Enviroment to run.",
    )
    parser.add_argument(
        "-c",
        "--control",
        default="random",
        choices=list(policies.keys()),
        help="Select what type of action to take.",
    )
    parser.add_argument(
        "-ne",
        "--num-episodes",
        default="10",
        type=int,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "-ns",
        "--num-steps",
        default="10000",
        type=int,
        help="Number of step to run per episode.",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        "--sample-frequency",
        default="250",
        type=float,
        help="The frequency of samples on the Quanser hardware.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-r", "--render", action="store_true")
    parser.add_argument("-s", "--use_simulator", action="store_true")
    args, _ = parser.parse_known_args()

    print("Testing Env:  {}".format(args.env))
    print("Controller:   {}".format(args.control))
    print("{} steps over {} episodes".format(args.num_steps, args.num_episodes))
    print("Samples freq: {}".format(args.frequency))
    test_env(
        envs[args.env],
        policies[args.control],
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        frequency=args.frequency,
        verbose=args.verbose,
        use_simulator=args.use_simulator,
        render=args.render,
    )


if __name__ == "__main__":
    main()
