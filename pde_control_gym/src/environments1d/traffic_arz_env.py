import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D
import random

class TrafficPDE1D(PDEEnv1D):
    r""" 
    Traffic ARZ PDE

    This class implements the Traffic ARZ PDE and inherits from the class :class:`PDEEnv1D`. Thus, for a full list of of arguments, first see the class :class:`PDEEnv1D` in conjunction with the arguments presented here

    :param simulation_type: Defines the type of boundary control. Inputs 'inlet', 'outlet' and 'both' represents boundary control at inlet, outlet and both respectively. 
    :param v_max: Maximum permissible velocity (meters/second) on freeway under simulation 
    :param ro_max: Maximum permissible density (vehicles/meter) on freeway under simulation
    :param v_steady: Desired steady state velocity (meters/second). Ensure that v_steady and ro_steady obey the equilibrium equation v_steady = v_max(1 - ro_steady/v_max)
    :param ro_steady: Desired steady state density (vehicles/meter). Ensure that v_steady and ro_steady obey the equilibrium equation v_steady = v_max(1 - ro_steady/v_max)
    :param tau: Relaxation time (seconds) required by the driver to adjust to the new velocity
    :param limit_pde_state_size: This is a boolean which will terminate the episode early if the observation velocity or density is greater than v_max and ro_max respectively
    :param control_freq: Number of PDE simulation steps performed using same action per environment step() call
    """
    def __init__(self, 
                 simulation_type: str = 'inlet', 
                 v_steady: float = 10,
                 ro_steady: float = 0.12,
                 v_max: float = 40,
                 ro_max: float = 0.16,
                 tau: float = 60,
                 limit_pde_state_size: bool = False,
                 control_freq: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.simulation_type = simulation_type
        self.vm = v_max
        self.rm = ro_max
        self.qm = v_max * ro_max/4
        self.tau = tau
        self.limit_pde_state_size = limit_pde_state_size

        assert(isinstance(control_freq, int) and control_freq >= 1) , f"control_freq must be a positive integer (got {control_freq} of type {type(control_freq).__name__})"
        self.control_freq = control_freq
        
        if self.simulation_type == 'outlet':
            print('Case 1: Outlet Boundary Control')
        elif self.simulation_type == 'inlet':
            print('Case 2: Inlet Boundary Control')
        elif self.simulation_type == 'both':
            print('Case 3: Outlet & Inlet Boundary Control')
        elif self.simulation_type == 'inlet-train':
            print('Case 4: Inlet training')
        elif self.simulation_type == 'outlet-train':
            print('Case 5: Outlet training')
        else:
            raise ValueError('Invalid simulation type')      

        if self.simulation_type == 'inlet' or self.simulation_type == 'outlet' or self.simulation_type == 'both':
            if v_steady != TrafficPDE1D.Veq(v_max, ro_max, ro_steady):
                raise ValueError('The steady state velocity and density do not satisfy the equilibrium condition. Check the values of v_steady and ro_steady and ensure that they obey v_steady = v_max(1 - ro_steady/v_max).')
            self.vs = v_steady
            self.rs = ro_steady
            self.qs = v_steady * ro_steady
            self.ps = self.vm/self.rm * self.qs/self.vs
        else:
            rand_index = random.randint(0, 2)
            rs_values = {0: 0.115, 1: 0.12, 2: 0.125}
            self.rs = rs_values[rand_index]
            self.vs = TrafficPDE1D.Veq(self.vm, self.rm, self.rs)
            self.qs = self.rs * self.vs
            
        print("Steady state density, velocity: ",self.rs, ",", self.vs)
        
        x = np.arange(0,self.X+self.dx,self.dx)
        self.L = self.X
        self.M = len(x)
        self.qs = self.qs
        self.qs_input = np.linspace(self.qs/2, 2*self.qs,40)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])

        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + TrafficPDE1D.Veq(self.vm, self.rm, self.r)
        
        self.info = dict()
        self.info['V'] = self.v

        #Observation space
        if self.simulation_type == 'outlet-train':
            self.observation_space  = spaces.Box(low=-10, high=10, shape=(2 * self.M,), dtype=np.float64)
        else:
            self.observation_space  = spaces.Box(low=0, high=40, shape=(2 * self.M,), dtype=np.float64) 
            
        #Action space
        if self.simulation_type == 'both':
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(2,))
        else:
            self.action_space = spaces.Box(dtype=np.float64, low = self.qs * 0.8, high = 1.2 * self.qs, shape=(1,))   


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
        if (self.limit_pde_state_size and (np.any(self.v > self.vm) or np.any(self.r > self.rm))):
            return True
        elif np.all(self.r - self.rs == 0) and np.all(self.v - self.vs == 0):
            return True
        else:
            return False



    def step(self, action):
        """
        step

        Updates the PDE state based on the action taken and returns the new state, reward, done, truncated and info. The PDE is solved using finite differencing explained in docs and the reward is computed based on the deviation from the desired density and velocity.

        :param action: The control input to apply to the freeway at the inlet, outlet or both.
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
        
        if self.simulation_type == 'both':
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)
            q_inlet_input = qs_input[0]
            q_outlet_input = qs_input[1]

        else:
            qs_input = np.clip(qs_input, a_min=self.action_space.low, a_max=self.action_space.high)[0]

        #PDE control at inlet
        if self.simulation_type == 'outlet' or self.simulation_type == 'outlet-train':
			# Fixed inlet boundary input
            self.q_inlet = self.qs

        elif self.simulation_type == 'inlet':
			# Control inlet boundary 
            self.q_inlet = qs_input

        elif self.simulation_type == 'both':
            # Control inlet boundary 
            self.q_inlet = q_inlet_input
        
        count = 0
        while count < self.control_freq and self.time_index < self.T:
            # Boundary conditions
            self.r[0] = self.r[1]
            self.y[0] = self.q_inlet - self.r[0] * TrafficPDE1D.Veq(self.vm, self.rm, self.r[0])
            self.r[self.M-1] = self.r[self.M-2]

            # PDE control at outlet
            if self.simulation_type == 'outlet' or self.simulation_type == 'outlet-train':
                # Control outlet boundary 
                self.y[self.M-1] = qs_input - self.r[self.M-1]* TrafficPDE1D.Veq(self.vm, self.rm, self.r[self.M-1])
            
            elif self.simulation_type == 'inlet':
                # Fixed outlet boundary 
                self.y[self.M-1] = self.qs - self.r[self.M-1]* TrafficPDE1D.Veq(self.vm, self.rm, self.r[self.M-1])
            
            elif self.simulation_type == 'both':
                # Control outlet boundary 
                self.y[self.M-1] = q_outlet_input - self.r[self.M-1]* TrafficPDE1D.Veq(self.vm, self.rm, self.r[self.M-1])
                
            #Vectorized finite differencing of PDE
            r_jm1 = self.r[0:self.M-2]
            r_j   = self.r[1:self.M-1]
            r_jp1 = self.r[2:self.M]
            
            y_jm1 = self.y[0:self.M-2]
            y_j   = self.y[1:self.M-1]
            y_jp1 = self.y[2:self.M]
            
            # Compute midpoint values
            r_pmid = 0.5 * (r_jp1 + r_j) - (dt / (2 * dx)) * (TrafficPDE1D.F_r(self.vm, self.rm, r_jp1, y_jp1) - TrafficPDE1D.F_r(self.vm, self.rm, r_j, y_j))
            r_mmid = 0.5 * (r_jm1 + r_j) - (dt / (2 * dx)) * (TrafficPDE1D.F_r(self.vm, self.rm, r_j, y_j) - TrafficPDE1D.F_r(self.vm, self.rm, r_jm1, y_jm1))
            
            y_pmid = (
                0.5 * (y_jp1 + y_j)
                - (dt / (2 * dx)) * (TrafficPDE1D.F_y(self.vm, self.rm, r_jp1, y_jp1) - TrafficPDE1D.F_y(self.vm, self.rm, r_j, y_j))
                - 0.25 * dt / self.tau * (y_jp1 + y_j)
            )
            
            y_mmid = (
                0.5 * (y_jm1 + y_j)
                - (dt / (2 * dx)) * (TrafficPDE1D.F_y(self.vm, self.rm, r_j, y_j) - TrafficPDE1D.F_y(self.vm, self.rm, r_jm1, y_jm1))
                - 0.25 * dt / self.tau * (y_jm1 + y_j)
            )
            
            # Update values in the inner domain
            self.r[1:self.M-1] -= (dt / dx) * (TrafficPDE1D.F_r(self.vm, self.rm, r_pmid, y_pmid) - TrafficPDE1D.F_r(self.vm, self.rm, r_mmid, y_mmid))
            self.y[1:self.M-1] -= (
                (dt / dx) * (TrafficPDE1D.F_y(self.vm, self.rm, r_pmid, y_pmid) - TrafficPDE1D.F_y(self.vm, self.rm, r_mmid, y_mmid))
                + 0.5 * dt / self.tau * (y_pmid + y_mmid)
            )

            count += 1

        # Calculate Velocity
        self.v = self.y/(self.r) + TrafficPDE1D.Veq(self.vm, self.rm, self.r)
        reward = self.reward_class.reward(self.vs, self.rs, self.v, self.r)
        
        if self.simulation_type == 'outlet-train':
            return np.reshape(np.concatenate(((self.r-self.rs)/self.rs, (self.v-self.vs)/self.vs)), -1), reward, self.terminate(), self.truncate(), self.info
        else:
            return np.reshape(np.concatenate((self.r, self.v)), -1), reward, (self.terminate() or reward > -0.00023), self.truncate(), self.info

    

    def reset(self, seed: Optional[int]=None, options: Optional[dict]=None):
        """
        Resets the environment to an initial state and returns an initial observation and info.
    
        :param seed: Optional seed for reproducibility.
        :param options: Optional dictionary to define initialization options.
        :return: A tuple of (observation, info).
        """

        x = np.arange(0,self.X+self.dx,self.dx)
        self.r = np.zeros([self.M,1])
        self.y = np.zeros([self.M,1])

        #Stochastic reset of environment during training
        if self.simulation_type == 'outlet-train':
            rand_index = random.randint(0, 2)
            rs_values = {0: 0.115, 1: 0.12, 2: 0.125}
            self.rs = rs_values[rand_index]
            self.vs = TrafficPDE1D.Veq(self.vm, self.rm, self.rs)
            self.qs = self.rs * self.vs
        
        #Initial condition of the PDE
        self.r = self.rs * np.transpose(np.sin(3 * x / self.L * np.pi ) * 0.1 + np.ones([1,self.M]))
        self.y = self.qs * np.ones([self.M,1]) - self.vm * self.r + self.vm / self.rm * (self.r)**(2)
        self.v = self.y/self.r + TrafficPDE1D.Veq(self.vm, self.rm, self.r)

        obs = np.reshape(np.concatenate((self.r, self.v)), -1)
    
        info = {}  # Optional info dict for debugging/logging
    
        return obs, info

    #Helper functions
    @staticmethod
    def Veq(vm, rm, rho):
        return vm * (1 - rho / rm)

    @staticmethod
    def F_r(vm, rm, rho, y):
        return y + rho * TrafficPDE1D.Veq(vm, rm, rho)

    @staticmethod
    def F_y(vm, rm, rho, y):
        return y * (y / rho + TrafficPDE1D.Veq(vm, rm, rho))

