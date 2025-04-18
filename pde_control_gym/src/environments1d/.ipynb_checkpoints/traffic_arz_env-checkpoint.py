import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
from pde_control_gym.src.environments1d.traffic_arz_utils import Veq, F_r, F_y

class TrafficPDE(PDEEnv1D):
    r""" 
    Traffic ARZ PDE

    This class implements the Traffic ARZ PDE and inhertis from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    :param simulation_type: Sets the type of boundary control. Inputs 1, 2 and 3 represents boundary control at inlet, outlet and both respectively. 
    :param v_steady: Sets the steady state velocity of the environment. 
    :param ro_steady: Sets the steady state density of the environment. 
    :param v_max: Sets the maximum velocity of the environment. 
    :param ro_max: Sets the maximum density of the environment. 
    :param tau: Sets the relaxation time
    """
    def __init__(self, 
                 simulation_type: int = 1, 
                 v_steady: float = 10,
                 ro_steady: float = 0.12,
                 v_max: float = 40,
                 ro_max: float = 0.16,
                 v_desired: float = 10,
                 ro_desired: float = 0.12,
                 tau: float = 60,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.simulation_type = simulation_type
        self.vm = v_max
        self.rm = ro_max
        self.qm = v_max * ro_max/4
        self.tau = tau

        
        if self.simulation_type == 1:
            print('Case 1: Outlet Boundary Control')
        elif self.simulation_type == 2:
            print('Case 2: Inlet Boundary Control')
        elif self.simulation_type == 3:
            print('Case 3: Outlet & Inlet Boundary Control')
        elif self.simulation_type == 4:
            print('Case 4: Stochastic Outlet Boundary Control for Training')
        elif self.simulation_type == 5:
            print('Case 5: Stochastic Inlet Boundary Control for Training')
        elif self.simulation_type == 6:
            print('Case 6: Stochastic Outlet Boundary Control for Validation // Lighter Traffic')
        elif self.simulation_type == 7:
            print('Case 7: Stochastic Inlet Boundary Control for Validation // Denser Traffic')
        else:
            raise ValueError('Invalid simulation type')

        if self.simulation_type == 1 or self.simulation_type == 2 or self.simulation_type == 3:  
            print('Determinstic Env.')
            self.vs = v_steady
            self.vs_desired = v_desired
            self.rs = ro_steady
            self.rs_desired = ro_desired
            self.qs = v_steady * ro_steady
            self.qs_desired = v_desired * ro_desired
            self.ps = self.vm/self.rm * self.qs/self.vs
        
        x = np.arange(0,self.X+self.dx,self.dx)
        self.L = self.X
        self.M = len(x)
        self.qs = self.qs
        self.qs_input = np.linspace(self.qs/2,2*self.qs,40)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])


        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)
        
        self.info = dict()
        self.info['V'] = self.v

	    # Observation space
        self.observation_space = spaces.Box(low=0, high=40, shape=(2 * self.M,), dtype="float64")

        #Action space
        if self.simulation_type == 1 or self.simulation_type == 2 or self.simulation_type == 4 or self.simulation_type == 5 or self.simulation_type == 6 or self.simulation_type == 7:
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(1,))
        elif self.simulation_type == 3:
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(2,))



    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if (self.time_index >= self.T / self.dt):
            self.time_index = 0
            return True
        else:
            return False

    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if all(self.r - self.rs_desired == 0) and all(self.v - self.vs_desired == 0):
            return True
        else:
            return False


    def step(self, action):
        """
        step

        Moves the PDE with control action forward ``control_sample_rate*dt`` steps.

        :param action: The control input to apply to the PDE at the inlet outlet or both.
        :return: A tuple of:
            - observation (np.ndarray): The concatenated state vector containing density (`r`) and velocity (`v`) after taking action.
            - reward (float): The reward computed based on deviation from desired density and velocity after action.
            - done (bool): Whether the simulation should terminate.
            - truncated (bool): Whether the simulation was truncated as desired state has been achieved.
            - info (dict): Additional information about the current state for debugging.
        """
        Nx = self.nx
        dx = self.dx
        dt = self.dt
        self.time_index += dt
        qs_input = action
        
        if self.simulation_type == 1 or self.simulation_type == 2 or self.simulation_type == 4 or self.simulation_type == 5 or self.simulation_type == 6 or self.simulation_type == 7:
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)[0]
        else:
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)
            q_inlet_input = qs_input[0]
            q_outlet_input = qs_input[1]

        #PDE control at inlet
        if self.simulation_type == 1 or self.simulation_type == 4 or self.simulation_type == 6:
			# Fixed inlet boundary input
            self.q_inlet = self.qs

        elif self.simulation_type == 2 or self.simulation_type == 5 or self.simulation_type == 7:
			# Control inlet boundary input (single-input)
            self.q_inlet = qs_input

        elif self.simulation_type == 3:
            # Control inlet boundary input (Multi-input)
            self.q_inlet = q_inlet_input

        # Boundary conditions
        self.r[0] = self.r[1]
        self.y[0] = self.q_inlet - self.r[0] * Veq(self.vm, self.rm, self.r[0])
        self.r[self.M-1] = self.r[self.M-2]

        # PDE control at outlet
        if self.simulation_type == 1 or self.simulation_type == 4 or self.simulation_type == 6:
            # Control outlet boundary input
            self.y[self.M-1] = qs_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.simulation_type == 2  or self.simulation_type == 5 or self.simulation_type == 7:
            # Fixed outlet boundary input
            self.y[self.M-1] = self.qs - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        elif self.simulation_type == 3:
            # Control outlet boundary input (Multi-input)
            self.y[self.M-1] = q_outlet_input - self.r[self.M-1]* Veq(self.vm, self.rm, self.r[self.M-1])
        
        #Finite differencing of PDEs
        for j in range(1,self.M-1) :

            r_pmid = 1/2 * (self.r[j+1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_r(self.vm, self.rm, self.r[j], self.y[j]) )

            y_pmid = 1/2 * (self.y[j+1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j+1], self.y[j+1]) - F_y(self.vm, self.rm, self.r[j], self.y[j])) - 1/4 * dt / self.tau * (self.y[j+1]+self.y[j])

            r_mmid = 1/2 * (self.r[j-1] + self.r[j]) - dt/(2 * dx) * ( F_r(self.vm, self.rm, self.r[j], self.y[j]) - F_r(self.vm, self.rm, self.r[j-1], self.y[j-1]))

            y_mmid = 1/2 * (self.y[j-1] + self.y[j]) - dt/(2 * dx) * ( F_y(self.vm, self.rm, self.r[j], self.y[j]) - F_y(self.vm, self.rm, self.r[j-1], self.y[j-1])) - 1/4 * dt / self.tau * (self.y[j-1]+self.y[j])

            self.r[j] = self.r[j] - dt/dx * (F_r(self.vm, self.rm, r_pmid, y_pmid) - F_r(self.vm, self.rm, r_mmid, y_mmid))
            self.y[j] = self.y[j] - dt/dx * (F_y(self.vm, self.rm, r_pmid, y_pmid) - F_y(self.vm, self.rm, r_mmid, y_mmid)) - 1/2 * dt/self.tau * (y_pmid + y_mmid)

        # Calculate Velocity
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

        # Reward
        # v_desired = self.vs_desired  # vs
        # r_desired = self.rs_desired  # rs
        # reward = -(np.linalg.norm(self.v - v_desired, ord=None) / (v_desired) + np.linalg.norm(self.r - r_desired, ord=None) / (r_desired))

        reward = self.reward_class.reward(self.vs_desired, self.rs_desired, self.v, self.r)
        
        return np.reshape(np.concatenate((self.r, self.v)), -1), reward, self.terminate(), self.truncate(), self.info


    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        Resets the environment to an initial state and returns an initial observation and info.
    
        :param seed: Optional seed for reproducibility.
        :param options: Optional dictionary of initialization options.
        :return: A tuple of (observation, info).
        """

        x = np.arange(0,self.X+self.dx,self.dx)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])


        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + Veq(self.vm, self.rm, self.r)

        obs = np.reshape(np.concatenate((self.r, self.v)), -1)
    
        info = {}  # Optional info dict for debugging/logging
    
        return obs, info



