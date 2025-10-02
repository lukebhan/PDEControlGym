from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional

class BrainTumorReward(BaseReward):
    '''
    def reward(self, uVec: np.ndarray =None, time_index: int = None, terminate: Optional[bool] =None, truncate: Optional[bool] =None, action: Optional[float] =None, **kwargs):
        # Episode reward
        t_benchmark = kwargs["t_benchmark"]
        if (terminate or truncate):
            return time_index - t_benchmark
          
        # Treatment reward
        def dmaxsafe(treatment_radius: int):
            return 116 * (treatment_radius ** -0.685)
        
        tumor_radius = kwargs["tumor_radius"]
        treatment_radius = kwargs["treatment_radius"]
        applied_dose = kwargs["applied_dose"]

        r_control = 1 / (1 + tumor_radius)
        r_toxic = 0.1 * (max(0, (applied_dose / dmaxsafe(treatment_radius)) - 1) ** 2)

        return r_control - r_toxic
    '''

    def reward():
        return 0
        

