import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod

class PDECEnv(gym.Env):
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
    #       sensing_noise_func - function to be called as the sample noise. It takes no parameters can can be invoked additively or multiplicatively according to sensing_noise_mode. Defaults to None. 
    #       sensing_noise_mode:
    #           "additive": sensing is computed as u_measurement+sensing_noise_func()
    #           "multiplicative": sensing is computed as u_measurement*sensing_noise_func() 
    #           Defaults to additive.
    #           
    # Rendering Modes (Given as a dicitonary for the second argument):
    #   Rendering is implemented with the vision for only being of control of a single PDE problem. This will cause LARGE overhead if modified for training.It is called once the episode terminates and has the following arguments that are passed as a dictionary:
    #       'animation_render': boolean. True means a matplotlib animation file with be produced with a step by step rendering. Defaults to False. 
    #       'animation_file_name': string. File name to save animation. Defaults to 'PDECGAnimate'
    #       'animation_time': int. Number in times in seconds for the entire animation to complete. Defaults to 10.
    #       'animation_show_error_figure': boolean. Adds a figure with episodes cumulative reward. Defaults to False.
    #       'animation_reward_func': function. Pass the reward function to be plotted if animation_show_error_figure is True. Defaults to tradition L2 norm.

    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def close(self):
        pass

    # Implements a series of plotting and animation modes for rendering results
    # Includes matplotlib animations as well as point rendering. 
    # args to the __init__ env function call initial set the rendering arguments. They can then be further updated using the render function call here. 
    def render(self):
        pass


