import numpy as np
from scipy.linalg import expm
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
import matplotlib.pyplot as plt


class NeuronPDE1D:
    r""" 
    Neuron Growth Control PDE

    This class implements the Neuron Growth Control PDE.

    :param D: Represents a biological constant 
    :param a: Represents a biological constant 
    :param g: Represents a biological constant 
    :param rsubg: Represents a biological constant 
    :param c_infty: Represents the equilibrium of the tubulin concentration in the cone.
    :param lsubc: Represents the growth ratio, which depends on the cone cross-sectional area A,and a volume of the growth cone Vc.
    :param lsubs: Represents the desired length of the axon in the x-coordinate.
    """


    def __init__(self,  
                 D: float = 10e-12, 
                 a: float = 1e-8,
                 g: float = 5e-7,  
                 rsubg: float = 1.783e-5,
                 c_infty: float = 0.0119,
                 tildeRsubg: float = 0.053,
                 lsubc: float = 4e-6,
                 lsubs: float = 12e-6,
                 lsubzero: float = 1e-6,
                 k1: float = -1e3,
                 k2: float = 1e13,
                 T: float = 180):
    
        # Initializes all physical parameters
        self.lsubzero = lsubzero
        self.lsubs = lsubs
        self.c_infty = 0.0119
        self.a = 1e-8
        self.g = 5e-7
        self.D = 10e-12
        self.gamma = (self.a/self.D)*10
        self.rsubg = rsubg
        self.H = np.zeros([2,1])
        self.H[0,0] = 1
        self.H[1,0] = -1*(((a - g*lsubc)*(c_infty))/D)
        self.atilde1 = ((a - rsubg*c_infty)/lsubc) - g - tildeRsubg
        self.atilde3 = (self.a**2 + self.D*self.g - self.a*self.g*lsubc) / (self.D*2)
        self.Asub1 = np.array([[self.atilde1, 0.0],
                       [rsubg, 0.0]])
        self.beta = self.D/lsubc
        self.B = np.zeros([2,1])
        self.B[0,0] = -1*self.beta
        self.lambda_minus = self.a - np.sqrt(a**2 + 4*self.D*self.g)
        self.lambda_minus = self.lambda_minus/(2*self.D)
        self.lambda_plus = self.a + np.sqrt(a**2 + 4*self.D*self.g)
        self.lambda_plus = self.lambda_plus/(2*self.D)
        self.Kminus = a - 2*g*lsubc
        self.Kminus = self.Kminus/2
        self.Kminus = self.Kminus/(np.sqrt(a**2 + 4*self.D*self.g))
        self.Kminus = 0.5 - self.Kminus
        self.Kplus = a - 2*g*lsubc
        self.Kplus = self.Kplus/2
        self.Kplus = self.Kplus/(np.sqrt(self.a**2 + 4*self.D*self.g))
        self.Kplus = 0.5 + self.Kplus

        # Gain vector and respective gain kernels
        self.K = np.zeros([2,1])
        self.k1 = k1
        self.k2 = k2
        self.K[0,0] = self.k1
        self.K[1,0] = self.k2

        # Initializes necessary simulation values
        self.dx = 0.01     # pick what you need
        self.dxreal = self.dx*self.lsubzero
        self.dt = 5e-6
        self.spatial_to_real_scale = 1e-5
        self.time_index = 0.0
        self.T = T

        # Define N1, used to calculate p, phi, and phi prime

        self.I2 = np.eye(2)
        self.Z2 = np.zeros((2,2))
        self.TR = (1.0/D) * (g*self.I2 + self.Asub1 + (a/D) * (self.B @ self.H.T))   # top-right (2x2)
        self.BR = (1.0/D) * (self.B @ self.H.T + a*self.I2)                  # bottom-right (2x2)
        self.N1 = np.block([[self.Z2, self.TR],
               [self.I2, self.BR]])
        
        # Define phi constants

        self.row_vector_1 = np.zeros((1,2))
        self.row_vector_1 = np.hstack([
            self.H.T,                                  # (1,2)
            self.K.T - ((self.H.T @ self.B @ self.H.T)/D) # (1,2)
            ]) 
        self.identity_vector = np.vstack([self.I2, np.zeros((2,2))])  # (4,2)
        self.identity_vector_flipped = np.vstack([np.zeros((2,2)),self.I2])


        # Define control input constants

        self.coefficient = ((self.H.T @ self.B)/D) + self.gamma


        # Initializes Z
        self.Z = np.zeros([2,1])

        # Intializes spatial grid
        x = np.arange(0,2.01,self.dx)
        self.M = len(x) # Maximum index of the spatial grid

        # Initializes error system
        self.u = np.zeros([self.M,1])

        # Initialize csubeq using spatial grid
        self.csubeq = np.zeros([self.M, 1])
        self.distance_from_tip = (self.spatial_to_real_scale*x) - lsubs
        self.csubeq[:, 0] = c_infty * (
            self.Kplus * np.exp(self.lambda_plus * self.distance_from_tip) +
            self.Kminus * np.exp(self.lambda_minus * self.distance_from_tip)
        )

        #Initial Condition of Z
        self.Z[0,0] = self.c_infty
        self.Z[1,0] = self.lsubzero - self.lsubs


        #Calculate L
        self.lt = self.Z[1,0] + lsubs
        self.W = 1e5  # conversion from physical length -> code length
        self.lt_code = float(self.W * self.lt)            # convert length to code units
        self.L_raw   = int(round(self.lt_code / self.dx))      # map to grid index
        self.L  = max(1, min(self.M - 1, self.L_raw))     # keeps L inside [1, M-1] so that it doesn't index out of bounds


        #Initial Condition of u
        for i in range(0,self.M):
            self.u[i,0] = 2*self.c_infty - self.csubeq[i,0]

        # Initialize and set up phi
        self.phi = np.zeros((self.M, 2))
        for i in range(self.M):
            self.phi[i, :] = (self.row_vector_1 @ expm(-1*(10**(-5))*(x[i])*self.N1) @ self.identity_vector).ravel()
        # Initialize and set up phi prime
        self.phi_prime = np.zeros((self.M, 2))
        for i in range(self.M):
            self.phi_prime[i, :] = (self.row_vector_1 @ expm(-1*(self.spatial_to_real_scale)*(x[i])*self.N1) @ self.identity_vector_flipped).ravel()

        # Initialize and set up p(x)
        self.p = np.zeros((self.M, 2))
        for i in range(self.M):
            self.p[i, :] = (self.phi_prime[i, :]) - (self.gamma * self.phi[i, :])

        # Intializing vectors used to plot length and tubulin concentration graphs
        self.l_graph = np.zeros([3600-1,1]) # to store values of lt
        self.t_graph = np.arange(0,180,0.05) # represents t
        self.cxt_graph = np.zeros([3600-1,self.M]) # to store values of c(x,t)
            


    def step(self):
        """
        step

        Updates the PDE state based on the action taken and returns the new state. The PDE is solved using finite differencing explained in docs.

        :return: A tuple of:
            - observation (np.ndarray): The concatenated state vector containing density (`r`) and velocity (`v`) after taking action.
            - reward (float): The reward computed based on deviation from desired density and velocity after action.
            - done (bool): Whether the simulation should terminate.
            - truncated (bool): Whether the simulation was truncated as desired state has been achieved.
            - info (dict): Additional information about the current state for debugging.
        """

        # Initialize space and time parameters for each step
        dx = self.dx # spatial step size for spatial grid
        dt = self.dt # step size for time
        self.time_index += dt # time update, increases every time that step() runs
        self.lt = self.Z[1,0] + self.lsubs # Computes length of axon at the current time step
        self.dxreal = self.dx*self.lt # physical spatial step size for finite difference scheme calculations

        # Creates an index that is used to store values for length and tubulin concentration
        self.index = self.time_index * 20 
        # Every 10,000 time steps, the values for length and tubulin concentration are stored
        if (self.time_index % 10000 == 0):
            self.l_graph[int(self.index)] = self.lt # stores current value for length of the axon
            self.cxt_graph[int(self.index),:] = (self.u + self.csubeq).ravel() # stores current values for tubulin concentration at time = t
        
        # Storing values for u(x,t), Z(t), and spatial index L before they get overwritten
        self.temp = self.u.copy() # copy of u(x,t)
        self.Z_old = self.Z.copy() # copy of Z(t)
        self.L_old = int(self.L) # copy of L

        # New value for Z[0]
        self.Z[0] = (self.atilde1*self.Z_old[0] 
                       - (self.beta) * ((3*self.u[self.L] - 4*self.u[self.L-1] + self.u[self.L-2])/(2*self.dxreal))
                       ) * dt + self.Z_old[0]
        
        # New value for Z[0]
        self.Z[1] = (self.rsubg*self.Z_old[0]) * dt + self.Z_old[1]
       

        # To compute new spatial index L, given that l(t) changed
        self.lt_new = self.Z[1,0] + self.lsubs # computes new l(t)
        self.lt_code = float(self.W * self.lt_new) # converts length to code units
        self.L_raw   = int(round(self.lt_code / self.dx)) # maps to grid index L
        self.L  = max(1, min(self.M - 1, self.L_raw))     # keep L inside [1, M-1]
        self.L_new = int(self.L) # Stores new L in L_new

        # Printing simulation results every 100,000 time steps
        if self.time_index % 100000 == 0:
            print(f"l(t) is: {self.lt:.3e}, ||u||={np.linalg.norm(self.u):.3e}")
            print(self.time_index)
        
        # Steps to calculate the backstepping control input
        
        # Calculates sum for discretized integral
        self.sum = 0
        for i in range(1,self.L):
            self.sum += self.dxreal*(self.p[i,:] @ self.B)*self.u[i]

        # Builds U(t)
        self.middle_term = (((self.dxreal)*(self.p[0,:] @ self.B)*self.u[0])/2) + self.sum + ((self.dxreal/2)*(self.p[self.L,:] @ self.B)*self.u[self.L]) # trapezoidal sum
        self.middle_term = self.middle_term/self.D
        self.control_input = self.lt * (((self.H.T @ self.B)/self.D + self.gamma) * self.u[0] - self.middle_term + (self.p[self.L,:] @ self.Z))

        # Set U(t) equal to the leftmost boundary
        self.ufic = self.u[1] - self.control_input*self.dxreal*2
        self.u[0] = ((self.D/(self.dxreal**2)) * (self.u[1] - 2*self.u[0] + self.ufic) - self.a/(2*self.dxreal) * (self.u[1] - self.ufic) - self.g * self.u[0]) * dt + self.u[0]


        # New u's for next time step (excluding the boundaries)
        for n in range(1,self.L_new):
            uxx = (self.temp[n+1,0] - 2*self.temp[n,0] + self.temp[n-1,0]) / (self.dxreal**2)
            ux  = ((self.temp[n+1] - self.temp[n-1])) / self.dxreal   # see (2) below
            self.u[n,0] = self.temp[n,0] + self.dt * ( self.D*uxx + (((n-1)/self.lt) * (self.rsubg*self.Z_old[0]) * (self.temp[n+1] - self.temp[n-1])/2)  - self.a*ux/(2) - self.g*self.temp[n,0] )

        # Rightmost boundary condition
        self.u[self.L_new, 0] = float(self.H.T @ self.Z)


        # At the end of the sim, program plots and saves "Axon length vs Time" graph and "Tubulin conentration at a specific length vs Time"
        if self.time_index == 180:
            plt.plot(self.t_graph, self.l_graph)
            plt.title(f"Axon length vs Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Axon length (m)")
            plt.savefig("axon_length.png")

            plt.figure()
            plt.plot(self.t_graph, self.cxt_graph[0,:], label="c(0,t)")
            plt.plot(self.t_graph, self.cxt_graph[60,:], label="c(l_s/2,t)")
            plt.plot(self.t_graph, self.cxt_graph[120,:], label="c(l_s,t)")
            plt.title(f"Tubulin concentration at a specific length vs Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Tubulin concentration 10^-3 mol/m^3")
            plt.savefig("tubulin_concentration.png")


        return self.u
    
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
    