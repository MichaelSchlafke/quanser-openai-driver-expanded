from gym_brt.reinforcement_learning.dql import DQN

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--path",
    default="", type=str,
    help="the model will be loaded from the given .pt file",
)
parser.add_argument(
    "-r", "--render",
    default=False, type=bool,
    help="toggles rendering of the simulation",
)
parser.add_argument(
    "-a", "--algortihm",
    default="dqn", type=str,
    chooices=["dqn"],  # extend this list as more algorithms are added
    help="the algorithm to use for training",
)

args, _ = parser.parse_known_args()

if args.algorithm == "dqn":
    env = BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.2  # originally 0.05
    EPS_DECAY = 10000  # originally 1000
    TAU = 0.005
    LR = 1e-4

# TODO: finish this
