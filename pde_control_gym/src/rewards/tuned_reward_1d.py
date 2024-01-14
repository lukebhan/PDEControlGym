from pde_control_gym.src.rewards.base_reward import Reward

class TunedReward1D(Reward):
    def __init__(self, rewardParams):
        self.truncate_penalty = rewardParams["truncate_penalty"]
        self.terminate_reward = rewardParams["terminal_reward"]
        self.nt = rewardParams["nt"]

    def shapedReward(self, uVec, time_index, terminate, truncate, action):

        if terminate and np.linalg.norm(uVec[time_index]) < 20:
            return (self.terminate_reward - np.sum(abs(uVec[:, -1]))/1000 - np.linalg.norm(uVec[time_index]))
        if truncate:
            return self.truncate_penalty*(self.nt-time_index)
        return np.linalg.norm(uVec[time_index-100])-np.linalg.norm(uVec[time_index])
