from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from gym import spaces
from gym_brt.envs.qube_base_env import QubeBaseEnv
from gym_brt.envs.qube_discrete_base_env import QubeDiscBaseEnv

"""
    Description:
        A pendulum is attached to an un-actuated joint to a horizontal arm,
        which is actuated by a rotary motor. The pendulum begins
        downwards and the goal is flip the pendulum up and then to keep it from
        falling by applying a voltage on the motor which causes a torque on the
        horizontal arm.

    Source:
        This is a modified version of Blue River Technology's QubeSwingupEnv from their whitepaper:
        https://arxiv.org/abs/2001.02254

    Observation:
        Type: Box(4)
        Num Observation                   Min         Max
        0   Rotary arm angle (theta)     -90 deg      90 deg
        1   Pendulum angle (alpha)       -180 deg     180 deg
        2   Cart Velocity                -Inf         Inf
        3   Pole Velocity                -Inf         Inf
        Note: the velocities are limited by the physical system.

    Actions:
        Type: Real number (1-D Continuous) (voltage applied to motor)

    Starting State:
        Theta = 0 + noise, alpha = pi + noise

    Episode Termination:
        When theta is greater than ±90° or after 2048 steps
"""


class QubeSwingupStatesSquaredEnv(QubeBaseEnv):
    """"
        Reward:
        r(s_t, a_t) = 1 - (0.75 * alpha^2 + 0.15 * theta^2 + 0.05 * alpha_dot^2 + 0.05 * theta_dot^2)
    """
    def _reward(self):
        # only modification from QubeSwingupEnv
        alpha_sqrd = np.square(self._alpha / np.pi)
        theta_sqrd = np.square((self._target_angle - self._theta) / np.pi)
        alpha_dot_sqrd = np.square(self._alpha_dot / np.pi)
        theta_dot_sqrd = np.square(self._theta_dot / np.pi)
        reward = 1 - (
            (0.75 * alpha_sqrd + 0.15 * theta_sqrd + 0.05 * alpha_dot_sqrd + 0.05 * theta_dot_sqrd)
        )
        return max(reward, 0)  # Clip for the follow env case

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(QubeSwingupStatesSquaredEnv, self).reset()
        state = self._reset_down()
        return state


class QubeSwingupDescActEnv(QubeDiscBaseEnv):
    def _reward(self):
        reward = 1 - (
                (0.8 * np.abs(self._alpha) + 0.2 * np.abs(self._target_angle - self._theta))
                / np.pi
        )
        return max(reward, 0)  # Clip for the follow env case

    def _isdone(self):
        done = False
        done |= self._episode_steps >= self._max_episode_steps
        done |= abs(self._theta) > (90 * np.pi / 180)
        return done

    def reset(self):
        super(QubeSwingupDescActEnv, self).reset()
        state = self._reset_down()
        return state


# Integral doesn't make sense because the reward is not sparse and thus is already taken into account for each step

# class QubeSwingupStateIntegralEnv(QubeSwingupStatesSquaredEnv):
#     """"
#         Reward:
#         r(s_t, a_t) = 1 - (0.75 * alpha^2 + 0.15 * theta^2 + 0.05 * alpha_dot^2 + 0.05 * theta_dot^2)
#     """
#     def __init__(self, *args, **kwargs):
#         super(QubeSwingupStateIntegralEnv, self).__init__(*args, **kwargs)
#         self._alpha_integral = 0
#         self._theta_integral = 0
#         self._alpha_dot_integral = 0
#         self._theta_dot_integral = 0
#
#     def _reward(self):
#         # identical to QubeSwingupStatesSquaredEnv
#         alpha_sqrd = np.square(self._alpha / np.pi)
#         theta_sqrd = np.square((self._target_angle - self._theta) / np.pi)
#         alpha_dot_sqrd = np.square(self._alpha_dot / np.pi)
#         theta_dot_sqrd = np.square(self._theta_dot / np.pi)
#         state_sqrd = (0.75 * alpha_sqrd + 0.15 * theta_sqrd + 0.05 * alpha_dot_sqrd + 0.05 * theta_dot_sqrd)
#         # summation of previous states
#         self._alpha_integral += alpha_sqrd
#         self._theta_integral += theta_sqrd
#         self._alpha_dot_integral += alpha_dot_sqrd
#         self._theta_dot_integral += theta_dot_sqrd
#         state_integral = (0.75 * self._alpha_integral + 0.15 * self._theta_integral + 0.05 * self._alpha_dot_integral + 0.05 * self._theta_dot_integral)
#         # reward calculation
#         reward = 1 - 0.7 * state_sqrd - 0.3 * min(state_integral / 100, 1)  # TODO: replace 100 with max_episode_steps
#         return max(reward, 0)  # Clip for the follow env case
