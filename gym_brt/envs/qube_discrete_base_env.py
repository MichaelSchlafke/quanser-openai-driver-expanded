from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gym
import time
import math
import numpy as np

from gym import spaces
from gym.utils import seeding

# For other platforms where it's impossible to install the HIL SDK
try:
    from gym_brt.quanser import QubeHardware
except ImportError:
    print("Warning: Can not import QubeHardware in qube_base_env.py")

from gym_brt.quanser import QubeSimulator
from gym_brt.envs.rendering import QubeRenderer
from gym_brt.envs.qube_base_env import QubeBaseEnv


class QubeDiscBaseEnv(QubeBaseEnv):
    """A modification of qube_base_env.
     This class is for discrete action spaces and allows for the actions left, right and no action.
     To compensate the sampling frequency is increased to 1kHz."""
    def __init__(
            self,
            frequency=250,
            batch_size=2048,
            use_simulator=False,
            encoder_reset_steps=int(1e8),
    ):
        super().__init__()
        self.action_space = spaces.Discrete(3)

