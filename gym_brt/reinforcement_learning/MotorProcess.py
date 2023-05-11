from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import multiprocessing
import threading
from multiprocessing import Process
import time
import numpy as np

try:
    from gym_brt.quanser import QubeHardware
except ImportError:
    print("Warning: Can not import QubeHardware in qube_base_env.py")


class MotorProcess:
    def __init__(self, frequency=250):
        self.qube = None
        self.frequency = frequency

    def loop_state_check(self, state, stop_flag, action, currents):
        self.qube = QubeHardware(frequency=self.frequency)
        while True:
            actionIn = np.array([-action.value], dtype=np.float64)
            state.get_obj()[:], currents.get_obj()[:] = self.qube.step_dt(actionIn)
            if stop_flag.value:
                break
        self.qube.close()
