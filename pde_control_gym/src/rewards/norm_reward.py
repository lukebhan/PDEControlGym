from pde_control_gym.src.rewards.base_reward import Reward

class NormReward(Reward):
    """
    NormReward
        
    This reward offers :math:`L_1, L_2, L_\infty` norms that can be implemented in a vairety of different ways according to the parameters. 

    :param rewardParameters: Takes in a dictionary consisting of the parameters listed below
    :param norm: Norm to use. Accepts "1", "2", "inf" corresponding to the norm to use. Default is "2".
    :param horizon: Designates how to compute the norm. "temporal" indicates to indicate the norm of the current state. "differential" indicates to compute the normed difference between this state and the last state. "t-horizon" allows the norm to be computed for the average of last ``t-horion-length`` steps. Default is "temporal"
    :param truncate_penalty: Allows the user to specify a penalty reward for each remaining timestep in case the episode is ended early. Default is :math:`-1e4`
    :param terminate_reward: Allows the user to add a reward for reaching the full length of the episode: Default is :math:`1e2`
    :param t-horizon-length: Allows the user to set the length to be average over in the case of ``t-horizon`` approach for the ``horizon`` parameter. Default is :math:`5`

    """

    defaultParams = {
        "norm": 2,
        "horizon": "temporal", 
        "truncate_penalty": -1e-4, 
        "terminate_reward": 1e2, 
        "t-horizon-length": 5
    }

    def __init__(self, rewardParameters=defaultParams):
        self.norm = rewardParameters["norm"]
        self.horizon = rewardParameters["horizon"]
        self.truncate_penalty = rewardParameters["truncate_penalty"]
        self.terminate_reward = rewardParameters["terminate_reward"]
        self.t_horizon_length = rewardParameters["t-horizon-length"]

    def reward(self, uVec, time_index, terminate, truncate, _):
        norm_coeff = len(uVec[time_index])
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
