import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Type
from pde_control_gym.src.rewards import BaseReward


class PDEEnv2D(gym.Env):
    """
    This is the base env for all 2D PDE problems. All 2D custom environments should inherit this environment and implement the eccording methods

    :param T: The end time of the simulation.
    :param dt: The temporal timestep of the simulation.
    :param X: The first dimension of spatial length of the simulation.
    :param dx: The first dimension of spatial timestep of the simulation.
    :param Y: The second dimension of spatial length of the simulation.
    :param dy: The second dimension of spatial timestep of the simulation.
    :param action_dim: the dimension of the action space
    :param reward_class: An instance of the reward class to specify user reward for each simulation step. Must inherit BaseReward class. See `reward documentation <../../utils/rewards.html>`_ for detials.
    :param normalize: Chooses whether to take action inputs between -1 and 1 and normalize them to betwen (``-max_control_value``, ``max_control_value``) or to leave inputs unaltered. ``max_control_value`` is environment specific so please see the environment for details. 
    """
    def __init__(self, T: float, dt: float, X: float, dx: float, Y: float, dy: float, action_dim: int, reward_class: Type[BaseReward], normalize: bool = False):
        super(PDEEnv2D, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.nt = int(round(T / dt))
        self.nx = int(round(X / dx + 1))
        self.ny = int(round(Y / dy + 1))
        self.dx = dx
        self.dy = dy
        self.dt = dt
      	# Observation Space is always full
        self.x = np.linspace(0, X, self.nx)
        self.y = np.linspace(0, Y, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.observation_space = spaces.Box(
                    np.full((self.nx, self.ny, 2), -np.inf, dtype="float32"),
                    np.full((self.nx, self.ny, 2), np.inf,  dtype="float32"),
                )
        
        # Action space is always just boundary control. Normalized to -1 and 1 but gets expanded according to max_control_value
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim, ), dtype=np.float32)
        if normalize:
            self.normalize = lambda action, max_value : (action + 1)*max_value - max_value
        else:
            self.normalize = lambda action, max_value : action
        # Holds entire system state. Modified with ghost points for parabolic PDE
        self.U = np.zeros((self.nt, self.nx, self.ny, 2))
        self.time_index = 0

        # Setup reward function. 
        self.reward_class = reward_class
        

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self, init_cond, recirculation_func):
        pass
