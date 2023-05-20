from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import multiprocessing
import threading
from multiprocessing import Process
import time
import numpy as np
# Qube libraries
from gym_brt.quanser import QubeSimulator
from gym_brt.envs.rendering import QubeRenderer
try:
    from gym_brt.quanser import QubeHardware
except ImportError:
    print("Warning: Can not import QubeHardware in MotorProcess.py")

# settings
use_simulator = True  # todo: clean up
MAX_MOTOR_VOLTAGE = 3

class MotorProcess:
    def __init__(self, frequency):
        self.qube = None
        self.frequency = frequency

    def loop_state_check(self, state, stop_flag, action, reward):
        # selects between Hardware and Simulator
        if use_simulator:
            # TODO: Check assumption: ODE integration should be ~ once per ms
            integration_steps = int(np.ceil(1000 / self.frequency))
            self.qube = QubeSimulator(
                forward_model="ode",
                frequency=self.frequency,
                integration_steps=integration_steps,
                max_voltage=MAX_MOTOR_VOLTAGE,
            )
        else:
            # print("DEBUG: initializing qube hardware")  # TODO: clean up
            self.qube = QubeHardware(
                frequency=self.frequency, max_voltage=MAX_MOTOR_VOLTAGE
            )

        while True:
            actionIn = np.array([-action.value], dtype=np.float64)
            state.get_obj()[:], reward.get_obj()[:], stop_flag.get_obj, _ = self.qube.step(actionIn)
            # TODO: replace with  observation, reward, terminated, _ = env.step(action.item()) while keeping .get_obj()
            if stop_flag.value:
                break
        self.qube.close()
