from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional

class NormReward(BaseReward):
    r""" 
    NormReward
        
    This reward offers :math:`L_1, L_2, L_\infty` norms that can be implemented in a vairety of different ways according to the parameters. 
    
    :param nt: The number of maximum timesteps for the episode simulation. No default: Error is thrown if not specified.
    :param norm: Norm to use. Accepts "1", "2", "inf" corresponding to the norm to use. Default is "2".
    :param horizon: Designates how to compute the norm. "temporal" indicates to indicate the norm of the current state. "differential" indicates to compute the normed difference between this state and the last state. "t-horizon" allows the norm to be computed for the average of last ``t-horion-length`` steps. Default is "temporal".
    :param truncate_penalty: Allows the user to specify a penalty reward for each remaining timestep in case the episode is ended early. Default is :math:`-1e4`.
    :param terminate_reward: Allows the user to add a reward for reaching the full length of the episode: Default is :math:`1e2`.
    :param t_horizon_length: Allows the user to set the length to be average over in the case of ``t-horizon`` approach for the ``horizon`` parameter. Default is :math:`5`.
    """

    def __init__(self, nt: int = None, norm: str = "2", horizon: str = "temporal", truncate_penalty: float=  -1e-4, terminate_reward: float = 1e2, t_horizon_length: int = 5, *extras):
        if nt is None:
            raise Exception("Number of simulation steps must be specified in the NormReward class.")
        self.nt = nt
        self.norm = norm
        self.horizon = horizon
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward
        self.t_hoizon_length = t_horizon_length

    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None):
        r""" 
        reward

        :param uVec: (required) This is the solution vector of the PDE of which to compute the reward on.
        :param time_index: (required) This is the time at which to compute the reward. (Given in terms of index of uVec).
        :param terminate: States whether the episode is the terminal episode.
        :param truncate: States whether the epsiode is truncated, or ending early.
        :param action: Ignored in this reward - needed to inherit from base reward class.

        """
        # Exception Handling
        if uVec == None:
            raise Exception("Class NormReward attempted to call reward function and recieved a None vector to compute on")
        if time_index == None:
            raise Exception("Class NormReward attempted to call reward fucntion and recieved a None time_index parameter to identify the reward step")

        # Check terminate and truncate conditions 
        if terminate:
            return self.terminate_reward
        if truncate:
            return self.truncate_penalty*(self.nt-time_index)

        match self.horizon:
            case "temporal":
                return -np.linalg.norm(uVec[time_index], ord=self.norm) / norm_coeff
            case "differential":
                if time_index > 0:
                    return np.linalg.norm(
                        uVec[time_index] - uVec[time_index - 1], ord=self.norm
                    )  / norm_coeff
                else:
                    return -np.linalg.norm(uVec[time_index], ord=self.norm) / norm_coeff
            case "t-horizon":
                # Handles cases where time_index < self.t_horizon_length
                result = 0
                if time_index > self.t_horizon_length:
                    for i in range(0, self.t_horizon_length):
                        result += np.linalg.norm(uVec[time_index-1 * i], ord=self.norm)
                    result /= self.t_horizon_length
                else:
                    for i in range(0, time_index):
                        result += np.linalg.norm(uVec[time_index-1 * i], ord=self.norm)
                    result /= time_index
                return -result / norm_coeff
