# Class for creating reward functions for PDEs
# All reward functions inherit the general class Reward
from abc import ABC, abstractmethod
import numpy as np


class Reward(ABC):
    @abstractmethod
    def reward(self):
        pass


class NormReward(Reward):
    # self.norm is a number: "1", "2", or "inf"
    # self.horizon can be "temporal", "differential", "t-horizon". See details in Docs website
    def __init__(self, nt, norm=2, horizon="temporal", reward_average_length=5, truncate_penalty=-1e4, terminate_reward=1e2):
        self.norm = norm
        self.horizon = horizon
        self.reward_average_length = reward_average_length
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward
        self.nt = nt

    # uVec is in the form
    # temporal: u(t, x)
    # differential: (u(t, x), u(t-dt, x))
    # t-horizon: (u(t-t_avg*dt, x), u(t-(t_avg-1)*dt, x), ..., u(t, x))
    def reward(self, uVec, time_index, terminate, truncate):
        if terminate:
            return self.terminate_reward
        if truncate:
            return self.truncate_penalty*(self.nt-time_index)
        norm_coeff = len(uVec[time_index])
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
                result = 0
                if time_index > self.reward_average_length:
                    for i in range(0, self.reward_average_length):
                        result += np.linalg.norm(uVec[time_index-1 * i], ord=self.norm)
                    result /= self.reward_average_length
                else:
                    for i in range(0, time_index):
                        result += np.linalg.norm(uVec[time_index-1 * i], ord=self.norm)
                    result /= time_index
                return -result / norm_coeff
