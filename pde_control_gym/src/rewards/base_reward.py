from abc import ABC, abstractmethod
import numpy as np
from typing import Optional

class BaseReward(ABC):
    """
    Reward (Abstract base class)

    This class is to be inherited by any custom reward functions. It has one abstract method, namely reward which is required to be overridden. 

    """

    @abstractmethod
    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None):
        r""" 
        reward

        :param uVec: (required) This is the solution vector of the PDE of which to compute the reward on.
        :param time_index: (required) This is the time at which to compute the reward. (Given in terms of index of uVec).
        :param terminate: States whether the episode is the terminal episode.
        :param truncate: States whether the epsiode is truncated, or ending early.
        :param action: Ignored in this reward - needed to inherit from base reward class.

        """

    def reset(self):
        r"""
        reset function

        This function is called anytime the environment resets. For the base reward func, it does nothing, but this can be(not required) overridden for handling custom reward functions with state
        """
        pass
