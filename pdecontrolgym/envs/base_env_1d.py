import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from .rewards import NormReward


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
    def __init__(self, hyperbolicParams):
        super(PDEEnv1D, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.parameters = hyperbolicParams
        self.parameters["nt"] = int(round(self.parameters["T"] / self.parameters["dt"]))
        self.parameters["nx"] = int(round(self.parameters["X"] / self.parameters["dx"]))

        # Observation space changes depending on sensing
        match self.parameters["sensing_loc"]:
            case "full":
                self.observation_space = spaces.Box(
                    np.full(self.parameters["nx"], -self.parameters["max_state_value"], dtype="float32"),
                    np.full(self.parameters["nx"], self.parameters["max_state_value"], dtype="float32"),
                )
            case "collocated" | "opposite":
                self.observation_space = spaces.Box(
                    np.full(1, -self.parameters["max_state_value"], dtype="float32"),
                    np.full(1, self.parameters["max_state_value"], dtype="float32"),
                )
            case _:
                raise Exception(
                    "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                )

        # Setup configurations for control and sensing. Messy, but done once, explicitly before runtime to setup return and control functions
        # There is a trick here where noise is a function call itself. Important that noise is a single argument function that returns a single argument
        match self.parameters["control_type"]:
            case "Neumann":
                self.control_update = lambda control, state, dx: control * dx + state
                match self.parameters["sensing_loc"]:
                    # Neumann control u_x(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Neumann control u_x(1), Dirchilet sensing u(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(state[-1])
                    case "opposite":
                        match "sensing_type":
                            # Neumann control u_x(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Neumann control u_x(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    state[0]
                                )
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
                    case _:
                        raise Exception(
                            "Invalid sensing_loc parameter. Please use 'full', 'collocated', or 'opposite'. See documentation for details."
                        )
            case "Dirchilet":
                self.control_update = lambda control, state, dt: control
                match self.parameters["sensing_loc"]:
                    # Dichilet control u(1), full state measurement
                    case "full":
                        self.sensing_update = lambda state, dx, noise: noise(state)
                    # Dichilet control u(1), Neumann sensing u_x(1)
                    case "collocated":
                        self.sensing_update = lambda state, dx, noise: noise(
                            (state[-1] - state[-2]) / dx
                        )
                    case "opposite":
                        match "sensing_type":
                            # Dichilet control u(1), Neumann sensing u_x(0)
                            case "Neumann":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    (state[1] - state[0]) / dx
                                )
                            # Dirchilet control u(1), Dirchilet sensing u(0)
                            case "Dirchilet":
                                self.sensing_update = lambda state, dx, noise: noise(
                                    state[0]
                                )
                            case _:
                                raise Exception(
                                    "Invalid sensing_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                                )
            case _:
                raise Exception(
                    "Invalid control_type parameter. Please use 'Neumann' or 'Dirchilet'. See documentation for details."
                )

        # Action space is always just boundary control. Normalized to -1 and 1 but gets expanded according to max_control_value
        self.action_space = spaces.Box(
            np.full(1, -1, dtype="float32"), np.full(1, 1, dtype="float32")
        )
        if self.parameters["normalize"]:
            self.normalize = lambda action, max_value : (action + 1)*max_value - max_value
        else:
            self.normalize = lambda action, max_value : action
        # Holds entire system state
        self.u = np.zeros((self.parameters["nt"], self.parameters["nx"]))
        self.time_index = 0

        # Setup reward function
        self.reward = NormReward(self.parameters["nt"],
            self.parameters["reward_norm"], self.parameters["reward_horizon"], self.parameters["reward_average_length"], self.parameters["truncate_penalty"], self.parameters["terminate_reward"])
        

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, init_cond, recirculation_func):
        pass

    # @abstractmethod
    # def close(self):
    #    pass

    # Implements a series of plotting and animation modes for rendering results
    # Includes matplotlib animations as well as point rendering.
    # args to the __init__ env function call initial set the rendering arguments. They can then be further updated using the render function call here.
    #
    # Rendering Modes (Given as a dicitonary for the second argument):
    #   Rendering is implemented with the vision for only being of control of a single PDE problem. This will cause LARGE overhead if modified for training.It is called once the episode terminates and has the following arguments that are passed as a dictionary:
    #       'animation_render': boolean. True means a matplotlib animation file with be produced with a step by step rendering. Defaults to False.
    #       'animation_file_name': string. File name to save animation. Defaults to 'PDECGAnimate'
    #       'animation_time': int. Amount of time in seconds for the entire animation to complete. Defaults to 10.
    #       'animation_show_error_figure': boolean. Adds a figure with episodes cumulative reward. Defaults to False.
    #       'animation_reward_func': function. Pass the reward function to be plotted if animation_show_error_figure is True. Defaults to traditional L2 norm.
    #       'animation_highlight_control': boolean. The control will be highlighted in red if Dirchilet conditions. Defaults to True.
    #       'animation_highlight_sensing': boolean. The sensing will be highlighted in blue if Dirchilet conditions. Defaults to False.
    #       'figure_generation': boolean. True means a matplotlib figure will be saved as well. Same exact properties as animations.
    #       'figure_file_name': string. File name to save figure. Defaults to 'PDECGFfigure'
    #       'figure_show_error_figure': boolean. Adds a figure with the episodes cumulative reward. Defaults to False.
    #       'figure_reward_func': function. Pass the reward function to be plotted if figure_show_error_figure is True. Defaults to traditional L2 norm.
    #       'figure_highlight_control': boolean. The control will be highlighted in red if Dirchilet conditions. Defaults to True.
    #       'figure_highlight_sensing': boolean. The sensing will be highlighted in blue if Dirchilet conditions. Defaults to False.
    def render(self):
        pass
