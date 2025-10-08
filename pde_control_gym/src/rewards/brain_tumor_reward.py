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
        
        tumor_radius = kwargs["tumor_radius"]
        treatment_radius = kwargs["treatment_radius"]
        applied_dosage = kwargs["applied_dosage"]

        lambda_control = 1
        lambda_toxic = 0.2
        if (tumor_radius is None):
            r_control = 1
        else:
            r_control = 1 / (1 + tumor_radius)
        r_toxic = max(0, (applied_dosage / dmaxsafe(treatment_radius)) - 1) ** 2

        if verbose:
            print(f"Reward Class: l_c*r_control - l_t*r_toxic = {lambda_control * r_control} - {lambda_toxic * r_toxic}")
            print(f"\tParams: tumor_radius={tumor_radius} treatment_radius={treatment_radius} applied_dosage={applied_dosage} dmaxsafe(treatment_radius)={dmaxsafe(treatment_radius)}")
        return lambda_control * r_control - lambda_toxic * r_toxic

