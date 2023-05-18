import multiprocessing
import numpy as np
import time
import Controller_NLP
from multiprocessing import Process
from Controller_NLP import Controller
from MotorProcess import MotorProcess
from PlotProcess import PlotProcess
from my_qube_interface import QubeHardware
# import dql
# import reinforce as rf
# import actor_critic as ac
import RL_processes as rl
import argparse


def control():
    """manages separate processes for communicating with the Qube Servo 2 hardware,
     the reinforcement learning algorithm and logging as well as plotting of the results"""

    """argument parsing"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default='dql'
                        , help='choose the algorithm to train the agent with',
                        choices=['dql', 'reinforce', 'actor_critic'])
    arg = parser.parse_args()
    training_loops = {'dql': rl.dql(), 'reinforce': rl.reinforce(), 'actor_critic': rl.actor_critic()}
    """Processes"""
    Qube = MotorProcess(frequency=250)
    # Plotting = PlotProcess()  # not implemented yet
    rl_loop = training_loops[arg.algorithm]

    # is state0 the initial state?
    state0 = np.array([0, 0, 0, 0])
    """shared variables for multiprocessing"""
    stop_flag = multiprocessing.Value('b', False)
    state = multiprocessing.Array('d', [state0[0], state0[1], state0[2], state0[3]])
    action = multiprocessing.Value('d', 0)
    currents = multiprocessing.Array('d', [0, 0])  # what's this for?
    receiver, sender = multiprocessing.Pipe()
    """Timespan of Calculation"""
    end_loop_time = 20
    """initialize the Processes"""
    p1 = Process(target=rl_loop, args=(state, currents, action, sender, stop_flag))
    p2 = Process(target=Qube.loop_state_check, args=(state, stop_flag, action, currents,))
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


if __name__ == "__main__":
    control()
