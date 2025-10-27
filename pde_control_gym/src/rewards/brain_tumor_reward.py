from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional

class BrainTumorReward(BaseReward):
    
    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None, verbose = True, **kwargs):
        # Episode reward
        t_benchmark = kwargs["t_benchmark"]
        if t_benchmark is None:
            if verbose:
                print(f"Warning: t_benchmark is not yet set -> returned reward of 0\n")
            return 0

        if (terminate or truncate):
            if verbose:
                print(f"Reward Class: time_index - t_benchmark = {time_index} - {t_benchmark}")
            return time_index - t_benchmark
          
        # Treatment reward
        def dmaxsafe(treatment_radius: int):
            return 116 * (treatment_radius ** -0.685)
        
        treatment_radius = kwargs["treatment_radius"]
        applied_dosage = kwargs["applied_dosage"]
        total_dosage = kwargs["total_dosage"]

        lambda_toxic = 50
        
        maxsafe = dmaxsafe(treatment_radius)
        if applied_dosage <= maxsafe:
            r_toxic = 0.0
        elif applied_dosage >= total_dosage:
            r_toxic = 1.0
        else:
            r_toxic = ((applied_dosage - maxsafe) / (total_dosage - maxsafe)) ** (1/3)

        if verbose:
            print(f"Reward Class: - l_t*r_toxic = {- lambda_toxic * r_toxic}")
            print(f"\tParams: treatment_radius={treatment_radius} applied_dosage={applied_dosage} dmaxsafe(treatment_radius)={dmaxsafe(treatment_radius)}")
        return - lambda_toxic * r_toxic

