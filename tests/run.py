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

# TODO: finish this
