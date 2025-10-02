from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional

class BrainTumorReward(BaseReward):
    
    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None, **kwargs):
        # Episode reward
        t_benchmark = kwargs["t_benchmark"]
        if t_benchmark is None:
            print(f"Warning: t_benchmark is not yet set -> returned reward of 0\n")
            return 0

        if (terminate or truncate):
            print(f"Reward Class: time_index - t_benchmark = {time_index} - {t_benchmark}")
            return time_index - t_benchmark
          
        # Treatment reward
        def dmaxsafe(treatment_radius: int):
            return 116 * (treatment_radius ** -0.685)
        
        tumor_radius = kwargs["tumor_radius"]
        treatment_radius = kwargs["treatment_radius"]
        applied_dosage = kwargs["applied_dosage"]

        r_control = 1 / (1 + tumor_radius)
        r_toxic = 0.1 * (max(0, (applied_dosage / dmaxsafe(treatment_radius)) - 1) ** 2)

        print(f"Reward Class: r_control - r_toxic = {r_control} - {r_toxic}")
        print(f"\tParams: tumor_radius={tumor_radius} treatment_radius={treatment_radius} applied_dosage={applied_dosage} dmaxsafe(treatment_radius)={dmaxsafe(treatment_radius)}")
        return r_control - r_toxic
    '''

    def reward():
        return 0
    '''  

