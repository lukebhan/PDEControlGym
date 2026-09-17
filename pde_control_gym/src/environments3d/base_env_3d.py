import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing import Type
from pde_control_gym.src.rewards import BaseReward


class PDEEnv3D(gym.Env):
    """
    This is the base env for all 3D PDE problems. All 3D custom environments should inherit this environment and implement the according methods.
    The grids/state use ``'ij'`` (matrix) indexing so index 0 is the ``x`` axis.

    :param T: The end time of the simulation.
    :param dt: The temporal timestep of the simulation.
    :param X: The first (length) spatial dimension of the simulation.
    :param dx: The first dimension spatial step of the simulation.
    :param Y: The second (width) spatial dimension of the simulation.
    :param dy: The second dimension spatial step of the simulation.
    :param Z: The third (height) spatial dimension of the simulation.
    :param dz: The third dimension spatial step of the simulation.
    :param action_dim: The dimension of the action space.
    :param reward_class: An instance of the reward class to specify user reward for each simulation step. Must inherit BaseReward class. See `reward documentation <../../utils/rewards.html>`_ for details.
    :param normalize: Chooses whether to take action inputs between -1 and 1 and normalize them to between (``-max_control_value``, ``max_control_value``) or to leave inputs unaltered. ``max_control_value`` is environment specific so please see the environment for details.
    :param state_dim: The number of state components stored per grid cell in the observation (e.g. 3 for velocity ``(u, v, w)``, or 5 to also carry pressure and temperature). Defaults to 3.
    """
    def __init__(self, T: float, dt: float, X: float, dx: float, Y: float, dy: float, Z: float, dz: float, action_dim: int, reward_class: Type[BaseReward], normalize: bool = False, state_dim: int = 3):
        super(PDEEnv3D, self).__init__()
        # Build parameters for number of time steps and number of spatial steps
        self.nt = int(round(T / dt))
        self.nx = int(round(X / dx + 1))
        self.ny = int(round(Y / dy + 1))
        self.nz = int(round(Z / dz + 1))
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.state_dim = state_dim

        # Spatial grids. 'ij' indexing keeps axis 0 aligned with x (length),
        # matching the staggered FFD solver's convention.
        self.x = np.linspace(0, X, self.nx)
        self.y = np.linspace(0, Y, self.ny)
        self.z = np.linspace(0, Z, self.nz)
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")

        # Observation Space is always full: (nx, ny, nz, state_dim).
        self.observation_space = spaces.Box(
            np.full((self.nx, self.ny, self.nz, state_dim), -np.inf, dtype="float32"),
            np.full((self.nx, self.ny, self.nz, state_dim), np.inf, dtype="float32"),
        )

        # Continuous action space of dimension action_dim, normalized to [-1, 1].
        # Interpretation (boundary control, setpoints, etc.) is environment-specific;
        # pass max_value as a scalar or per-component array to normalize().
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim, ), dtype=np.float32)
        if normalize:
            self.normalize = lambda action, max_value : (action + 1)*max_value - max_value
        else:
            self.normalize = lambda action, max_value : action

        # Holds entire system state. Note: for large 3D grids the full
        # (nt, nx, ny, nz, state_dim) history can be memory-heavy; an
        # inheriting environment may override reset() to store a smaller buffer.
        self.U = np.zeros((self.nt, self.nx, self.ny, self.nz, state_dim))
        self.time_index = 0

        # Setup reward function.
        self.reward_class = reward_class

    @abstractmethod
    def step(self, action):
        """
        step

        Implements the environment behavior for a single timestep depending on a given action.

        :param action: The control action to apply to the PDE at the boundary.
        """
        pass

    @abstractmethod
    def reset(self, init_cond, recirculation_func):
        """
        reset

        Resets the environment at the start of each episode according to the parameters given during environment initialization.
        """
        pass
