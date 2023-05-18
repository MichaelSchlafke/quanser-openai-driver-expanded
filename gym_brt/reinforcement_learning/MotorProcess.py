from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import multiprocessing
import threading
from multiprocessing import Process
import time
import numpy as np
from my_qube_interface import QubeHardware


class MotorProcess:
    def __init__(self, frequency):
        self.qube = None
        self.frequency = frequency

    def loop_state_check(self, state, stop_flag, action, currents):
        self.qube = QubeHardware(frequency=self.frequency)
        while True:
            actionIn = np.array([-action.value], dtype=np.float64)
            state.get_obj()[:], currents.get_obj()[:] = self.qube.step_dt(actionIn)
            # TODO: replace with  observation, reward, terminated, _ = env.step(action.item()) while keeping .get_obj()
            if stop_flag.value:
                break
        self.qube.close()
