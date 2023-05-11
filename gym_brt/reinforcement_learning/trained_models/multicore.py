import multiprocessing
import numpy as np
import time
from multiprocessing import Process

try:
    from gym_brt.quanser.quanser_wrapper.quanser_wrapper import QubeServo2
except ImportError:
    print("Warning: Can not import QubeServo2 in qube_interface.py")

from gym_brt.reinforcement_learning.MotorProcess import MotorProcess
from gym_brt.reinforcement_learning.reinforce import main as reinforce

"""shared variables for multiprocessing"""
Qube = MotorProcess()

state0 = [0, 0, 0, 0]  # TODO: verify

stop_flag = multiprocessing.Value('b', False)
state = multiprocessing.Array('d', [state0[0], state0[1], state0[2], state0[3]])  # TODO: check if [0,0,0,0] sensible
action = multiprocessing.Value('d', 0)
currents = multiprocessing.Array('d', [0, 0])
receiver, sender = multiprocessing.Pipe()
"""Timespan of Calculation"""
end_loop_time = 20
"""initialize the Processes"""
# TODO: rewrite main function so that they get their args from here to facilitate multicore
# p1 = Process(target=reinforce, args=(state, action, sol_states0, sol_input0, step_size, end_loop_time, sender,))
p1 = Process(target=reinforce, args=(state, action, end_loop_time, sender,))
p2 = Process(target=Qube.loop_state_check, args=(state, stop_flag, action, currents,))
# multicore plotting not implemented but sensible
# p3 = Process(target=Plotting.plot_loop, args=(state, step_size, receiver,))
"""start the Processes"""
p1.start()
p2.start()
# p3.start()
"""wait Processes to finish"""
p1.join()
stop_flag.value = True
p2.join()
# p3.join()
