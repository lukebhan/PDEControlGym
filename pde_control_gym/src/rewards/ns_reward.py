from pde_control_gym.src.rewards.base_reward import BaseReward
from typing import Optional, Union
import numpy as np

class NSReward(BaseReward):
    """
    NSReward

    This is a reward aiming at tracking the desired trajectory while minimizing the action cost :math: `-\frac{1}{2} \|s'-s_{ref}\|_{L_2} - \frac{1}{2} \|a-a_{ref}\|_{L_2}

    :param gamma: coefficient for the action cost
    """
    
    def __init__(self, gamma: float=0.1):
        self.gamma = gamma

    def reward(self, uVec: np.ndarray = None, time_index: int = None, U_ref: Union[float, np.ndarray]=None, action: Union[float, np.ndarray] = None, action_ref:  Union[float, np.ndarray] = None):
        """ 
        reward

        :param uVec: (required) This is the difference of the vector of PDE and tracking trajectory. 
        :param action: (required) control actions
        """
        return - 1/2 * np.linalg.norm(uVec[time_index]-U_ref[time_index])**2/uVec.shape[1]/uVec.shape[2] - self.gamma/2 * np.linalg.norm(action - action_ref[time_index])**2