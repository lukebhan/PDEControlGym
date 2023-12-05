import numpy as np
import pygame

from .base_env_1d import PDEEnv1D

import gymnasium as gym
from gymnasium import spaces

class HyperbolicPDEEnv(PDEEnv1D):
    def __init__(self, hyperbolicParams):
        super(HyperbolicPDEEnv, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.parameters = hyperbolicParams
        self.parameters["nt"] = int(round(self.parameters["T"] / self.parameters["dt"]))
        self.parameters["nx"] = int(round(self.parameters["X"] / self.parameters["dx"]))

        # Observation space changes depending on sensing 
        match self.parameters.sensing_loc:
            case "full":
                self.observation_space = spaces.Box(np.full(nx, -np.inf, dtype="float32"), np.full(nx, np.inf, dtype="float32"))
            case "collocated" | "opposite":
                self.observation_space = spaces.Box(np.full(1, -np.inf, dtype="float32"), np.full(1, np.inf, dtype="float32"))
            case _:
                raise Exception("Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details.")

        # Action space is always just boundary control
        self.action_space = spaces.Box(np.full(1, -np.inf, dtype="float32"), np.full(1, np.inf, dtype="float32"))
        # Holds entire system state
        self.u = np.zeros((self.parameters["nt"], self.parameters["nx"]))
        self.time_index = 0

    def step(self):
        
    
    # Resets the system state
    def reset(self, init_condition=None):
        if init_condition is None:
            init_condition = np.ones((self.parameters["nx"))
        self.u = np.zeros((self.parameters["nt"], self.parameters["nx"]))
        self.u[0] = init_condition
        self.time_index = 0


            

    
