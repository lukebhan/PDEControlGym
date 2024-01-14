from abc import ABC, abstractmethod
import numpy as np


class Reward(ABC):
    """
    Reward (Abstract base class)

    This class is to be inherited by any custom reward functions. It has one abstract method, namely reward which is required to be overridden. 

    """

    @abstractmethod
    def reward(self):
        pass
