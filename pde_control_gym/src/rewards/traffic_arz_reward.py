from pde_control_gym.src.rewards.base_reward import BaseReward
from typing import Optional
import numpy as np

class TrafficARZReward(BaseReward):
    """
    TrafficARZReward

    This is a custom reward used for evaluating the action taken by traffic controller with respect to steady velocity and density
    """

    def reward(self, v_desired: float, r_desired: float, v: np.ndarray, r: np.ndarray):
        """ 
        reward

        :param v_desired: (required) Desired steady state velocity
        :param r_desired: (required) Desired steady state density
        :param v: (required) Current state velocity
        :param r: (required) Current state density
        """

        return -(np.linalg.norm(v - v_desired, ord=None) / (v_desired) + np.linalg.norm(r - r_desired, ord=None) / (r_desired))