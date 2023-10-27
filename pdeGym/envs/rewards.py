# Class for creating reward functions for PDEs
# All reward functions inherit the general class Reward

class Reward(ABC):
    @abstractmethod
    def reward(self):
        pass

class NormReward(Reward):
    def __init__(self, norm, horizon):
        self.norm = norm
        self.horizon = horizon
    
    # uVec is in the form 
    # temporal: u(t, x)
    # differential: (u(t, x), u(t-dt, x))
    # t-horizon: (u(t-t_avg*dt, x), u(t-(t_avg-1)*dt, x), ..., u(t, x))
    def reward(self, uVec):
        match self.horizon:
            case "temporal":
                return self.mse(uVec, ord=self.norm)
            case "differential":
                return self.linalg.norm(uVec[0]-uVec[1], ord=self.norm)
            case "t-horizon":
                # t_avg is handled in the env. For cases t<t_avg, take all values up to t
                result = 0
                for i in range(1, len(uVec)+1):
                    result += self.mse(uVec[-1*i], ord=self.norm)
                return result
