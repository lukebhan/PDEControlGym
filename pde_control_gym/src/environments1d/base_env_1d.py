import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

class PDEEnv1D(gym.Env):
    # The entire enviornment and PDE problem gets specified here, so after initialization of the environment each step size is extremely quick
    # PDE Settings (given as a dictionary for the first argument)
    #   T - time horizon. Defaults to 1.
    #   dt - temporal step size. Defaults to 1e-3.
    #   X - spatial length. Defaults to 1.
    #   dx - spatial step size. Defaults to 1e-1.
    #   control_loc:
    #       control will always be at position 0 in the system. There are two options for control:
    #           'Neumann': control is applied at du/dx|_{x=0}.
    #           'Dirchilet': control is applied at x=0.
    #           Defaults to 'Dirchilet'.
    #   sensing_loc:
    #       sensing depends on the control values and takes two parameters in the form of a tuple.
    #           The first argument can take one of two forms:
    #               'full': full state measurement is returned at every step.
    #               'collocated': sensing is given at the point x=0 for Neumann control or its derivative du/dx|_{x=0} for Dirchilet control.
    #               'opposite': sensing is given at the opposite point x=X.
    #               Default is 'full'
    #           The second argument only applies to whether the sensing is Dirchilet or Neumann for the 'opposite'.
    #               'Neumann': sensing is given for the point at du/dx|_{x=X}
    #               'Dirchilet': sensing is given for the point at x=X
    #               Defaults to Dirchilet.
    #   sensing_noise_func - function to be called as the sample noise. It takes no parameters can can be invoked additively or multiplicatively according to sensing_noise_mode. Defaults to None.
    #   limit_pde_state_size: Used to end the epsidoe early if a PDE state is above a certain value specified by max_state_value (See below). Defaults to True.
    #   max_state_value: Only used if limit_pde_state_size is set to True. Sets the maximum value for the PDE at a time step to continue the environment. If over the max value, the episode ends and the reward is given as remaining time steps * max value. Default is 1e10
    def __init__(self, params):
        super(PDEEnv1D, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.parameters = params
        self.parameters["nt"] = int(round(self.parameters["T"] / self.parameters["dt"]))
        self.parameters["nx"] = int(round(self.parameters["X"] / self.parameters["dx"]))

      	# Observation Space handled in individual environment due to sensing modifications

        # Action space is always just boundary control. Normalized to -1 and 1 but gets expanded according to max_control_value
        self.action_space = spaces.Box(
            np.full(1, -1, dtype="float32"), np.full(1, 1, dtype="float32")
        )
        if self.parameters["normalize"]:
            self.normalize = lambda action, max_value : (action + 1)*max_value - max_value
        else:
            self.normalize = lambda action, max_value : action
        # Holds entire system state. Modified with ghost points for parabolic PDE
        self.u = np.zeros((self.parameters["nt"], self.parameters["nx"]))
        self.time_index = 0

        # Setup reward function. Use custom reward with shaping
        #self.reward = NormReward(self.parameters["nt"],
        #    self.parameters["reward_norm"], self.parameters["reward_horizon"], self.parameters["max_state_value"], self.parameters["reward_average_length"], self.parameters["truncate_penalty"], self.parameters["terminate_reward"])
        self.reward = self.parameters["reward_class"](self.parameters["reward_parameters"])
        

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, init_cond, recirculation_func):
        pass
