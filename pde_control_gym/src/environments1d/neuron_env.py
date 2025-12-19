import numpy as np
from scipy.linalg import expm
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional
import matplotlib.pyplot as plt
from pde_control_gym.src.environments1d.base_env_1d import PDEEnv1D

class NeuronPDE1D(PDEEnv1D):
    r""" 
    Neuron Growth Control PDE

    This class implements the Neuron Growth Control PDE.

    :param cInfty: Represents the equilibrium of the tubulin concentration in the cone.
    :param LSubS: Represents the desired length of the axon in the x-coordinate.
    :param k1: Represents a gain parameter 
    :param k2: Represents a gain parameter 
    :param LSubZero: Represents the initial length of the axon.
    :param LSubC: Represents the growth ratio.
    :param a: Represents the tubulin velocity constant.
    :param g: Represents the tubulin degradation rate.
    :param D: Represents the tubulin diffusion constant.
    :param TildeRSubG: Represents the reaction rate to create microtubules.
    :param RSubG: Represents a lumped parameter that depends on the density of assembled microtubules and the effective area of created microtubules growth.
    :param limit_pde_state_size: This is a boolean which will terminate the episode early if :math:`\|u(x, t)\|_{L_2} \geq` ``max_state_value``.
    :param max_state_value: Only used when ``limit_pde_state_size`` is ``True``. Then, this sets the value for which the :math:`L_2` norm of the PDE will be compared to at each step asin ``limit_pde_state_size``.
     
    """

    def __init__(self,  
                 cInfty: float = 0.0119,
                 LSubS: float = 12e-6,
                 k1: float = -1e3,
                 k2: float = 1e13,
                 LSubZero: float = 1e-6,
                 LSubC: float = 4e-6,
                 a: float = 1e-8,
                 g: float = 5e-7,
                 D: float = 10e-12,
                 TildeRSubG: float = 0.053,
                 RSubG: float = 1.783e-5,
                 limit_pde_state_size: bool = False,
                 max_state_value: float = 1e10,
                 **kwargs):
        super().__init__(**kwargs)
        self.limit_pde_state_size = limit_pde_state_size
        self.max_state_value = max_state_value

        # Initializes all physical parameters
        
        self.LSubS = LSubS
        self.cInfty = cInfty
        self.k1 = k1
        self.k2 = k2
        self.LSubZero = LSubZero
        self.LSubC = LSubC
        self.a = a
        self.g = g
        self.D = D
        self.TildeRSubG = TildeRSubG
        self.RSubG = RSubG
        self.gamma = (self.a/self.D)*10
        self.H = np.zeros([2,1])
        self.H[0,0] = 1
        self.H[1,0] = -1*(((self.a - self.g*self.LSubC)*(cInfty))/self.D)
        self.aTilde1 = ((self.a - self.RSubG*cInfty)/self.LSubC) - self.g - self.TildeRSubG
        self.aTilde3 = (self.a**2 + self.D*self.g - self.a*self.g*self.LSubC) / (self.D*2)
        self.Asub1 = np.array([[self.aTilde1, 0.0],
                       [self.RSubG, 0.0]])
        self.beta = self.D/self.LSubC
        self.B = np.zeros([2,1])
        self.B[0,0] = -1*self.beta
        self.LambdaMinus = self.a - np.sqrt(self.a**2 + 4*self.D*self.g)
        self.LambdaMinus = self.LambdaMinus/(2*self.D)
        self.LambdaPlus = self.a + np.sqrt(self.a**2 + 4*self.D*self.g)
        self.LambdaPlus = self.LambdaPlus/(2*self.D)
        self.Kminus = self.a - 2*self.g*self.LSubC
        self.Kminus = self.Kminus/2
        self.Kminus = self.Kminus/(np.sqrt(self.a**2 + 4*self.D*self.g))
        self.Kminus = 0.5 - self.Kminus
        self.Kplus = self.a - 2*self.g*self.LSubC
        self.Kplus = self.Kplus/2
        self.Kplus = self.Kplus/(np.sqrt(self.a**2 + 4*self.D*self.g))
        self.Kplus = 0.5 + self.Kplus

        # Gain vector and respective gain kernels
        self.K = np.zeros([2,1])
        self.K[0,0] = self.k1
        self.K[1,0] = self.k2

        # find length of spatial grid
        self.length = self.X
        self.SpatialToRealScale = 1
        while self.length < 1:
            self.length = self.length*10
            self.SpatialToRealScale = self.SpatialToRealScale/10

        # Initializes necessary simulation values
        self.dxreal = self.dx*self.LSubZero 

        # Define N1, used to calculate p, phi, and phi prime

        self.I2 = np.eye(2)
        self.Z2 = np.zeros((2,2))
        self.TR = (1.0/self.D) * (self.g*self.I2 + self.Asub1 + (self.a/self.D) * (self.B @ self.H.T))   # top-right (2x2)
        self.BR = (1.0/self.D) * (self.B @ self.H.T + self.a*self.I2)                  # bottom-right (2x2)
        self.N1 = np.block([[self.Z2, self.TR],
               [self.I2, self.BR]])
        
        # Define phi constants

        self.RowVector1 = np.zeros((1,2))
        self.RowVector1 = np.hstack([
            self.H.T,                                  # (1,2)
            self.K.T - ((self.H.T @ self.B @ self.H.T)/self.D) # (1,2)
            ]) 
        self.IdentityVector = np.vstack([self.I2, np.zeros((2,2))])  # (4,2)
        self.IdentityVectorFlipped = np.vstack([np.zeros((2,2)),self.I2])


        # Define control input constants

        self.coefficient = ((self.H.T @ self.B)/self.D) + self.gamma


        # Initializes Z
        self.Z = np.zeros([2,1])

        # Intializes spatial grid
        x = np.arange(0,self.length + self.dx,self.dx)
        self.M = len(x) # Maximum index of the spatial grid

        # Initializes error system
        self.u = np.zeros([self.M,1])

        # Initialize CSubEq using spatial grid
        self.CSubEq = np.zeros([self.M, 1])
        self.DistanceFromTip = (self.SpatialToRealScale*x) - self.LSubS
        self.CSubEq[:, 0] = self.cInfty * (
            self.Kplus * np.exp(self.LambdaPlus * self.DistanceFromTip) +
            self.Kminus * np.exp(self.LambdaMinus * self.DistanceFromTip)
        )

        #Initial Condition of Z
        self.Z[0,0] = self.cInfty
        self.Z[1,0] = self.LSubZero - self.LSubS


        #Calculate L
        self.L  = NeuronPDE1D.Conversion(self.Z[1,0],self.LSubS,self.SpatialToRealScale,self.dx,self.M)     # keeps L inside [1, M-1] so that it doesn't index out of bounds


        #Initial Condition of u
        for i in range(0,self.M):
            self.u[i,0] = 2*self.cInfty - self.CSubEq[i,0]

        # Initialize and set up phi
        self.phi = np.zeros((self.M, 2))
        for i in range(self.M):
            self.phi[i, :] = (self.RowVector1 @ expm(-1*self.SpatialToRealScale*(x[i])*self.N1) @ self.IdentityVector).ravel()
        # Initialize and set up phi prime
        self.PhiPrime = np.zeros((self.M, 2))
        for i in range(self.M):
            self.PhiPrime[i, :] = (self.RowVector1 @ expm(-1*(self.SpatialToRealScale)*(x[i])*self.N1) @ self.IdentityVectorFlipped).ravel()

        # Initialize and set up p(x)
        self.p = np.zeros((self.M, 2))
        for i in range(self.M):
            self.p[i, :] = (self.PhiPrime[i, :]) - (self.gamma * self.phi[i, :])
            


    def step(self):
        """
        step

        Updates the PDE state based on the action taken and returns the new state. The PDE is solved using finite differencing explained in docs.

        :return:
            - observation: The error vector :math:`u(x,t) = c(x,t) - c_eq(x)` containing density (`r`) and velocity (`v`) after taking action.
        """

        # Initialize space and time parameters for each step
        dx = self.dx # spatial step size for spatial grid
        dt = self.dt # step size for time
        self.time_index += dt # time update, increases every time that step() runs
        self.lt = self.Z[1,0] + self.LSubS # Computes length of axon at the current time step
        self.dxreal = self.dx*self.lt # physical spatial step size for finite difference scheme calculations
        
        
        # Storing values for u(x,t), Z(t), and spatial index L before they get overwritten
        self.temp = self.u.copy() # copy of u(x,t)
        self.ZOld = self.Z.copy() # copy of Z(t)
        self.LOld = int(self.L) # copy of L
        self.LMinusOne = self.L-1
        self.LMinusTwo = self.L-2

        # New value for Z[0]
        self.Z[0] = (self.atilde1*self.ZOld[0] 
                       - (self.beta) * ((3*self.u[self.L] - 4*self.u[self.LMinusOne] + self.u[self.LMinusTwo])/(2*self.dxreal))
                       ) * dt + self.ZOld[0]
        
        # New value for Z[0]
        self.Z[1] = (self.RSubG*self.ZOld[0]) * dt + self.ZOld[1]
       

        # To compute new spatial index L, given that l(t) changed
        self.L = NeuronPDE1D.Conversion(self.Z[1,0],self.LSubS,self.SpatialToRealScale,self.dx,self.M)
        self.LNew = self.L

        
        # Steps to calculate the backstepping control input
        
        # Calculates sum for discretized integral
        self.sum = 0
        for i in range(1,self.L):
            self.sum += self.dxreal*(self.p[i,:] @ self.B)*self.u[i]

        # Builds U(t)
        self.MiddleTerm = ((((self.dxreal)*(self.p[0,:] @ self.B)*self.u[0])/2) + self.sum + ((self.dxreal/2)*(self.p[self.L,:] @ self.B)*self.u[self.L])) / self.D # trapezoidal sum
        self.ControlInput = self.lt * (((self.H.T @ self.B)/self.D + self.gamma) * self.u[0] - self.MiddleTerm + (self.p[self.L,:] @ self.Z))

        # Set U(t) equal to the leftmost boundary
        self.ufic = self.u[1] - self.ControlInput*self.dxreal*2
        self.u[0] = ((self.D/(self.dxreal**2)) * (self.u[1] - 2*self.u[0] + self.ufic) - self.a/(2*self.dxreal) * (self.u[1] - self.ufic) - self.g * self.u[0]) * dt + self.u[0]


        # New u's for next time step (excluding the boundaries)
        for n in range(1,self.LNew):
            uxx = (self.temp[n+1,0] - 2*self.temp[n,0] + self.temp[n-1,0]) / (self.dxreal**2)
            ux  = ((self.temp[n+1] - self.temp[n-1])) / self.dxreal   # see (2) below
            self.u[n,0] = self.temp[n,0] + self.dt * ( self.D*uxx + (((n-1)/self.lt) * (self.RSubG*self.ZOld[0]) * (self.temp[n+1] - self.temp[n-1])/2)  - self.a*ux/(2) - self.g*self.temp[n,0] )

        # Rightmost boundary condition
        self.u[self.LNew, 0] = self.H.T @ self.Z

        terminate = self.terminate()
        truncate = self.truncate()
        return (
            self.reward_class.reward(self.u, self.time_index, terminate, truncate),
            terminate,
            truncate, 
            {},
        )
    
    def terminate(self):
        """
        terminate

        Determines whether the episode should end if the ``T`` timesteps are reached
        """
        if (self.time_index >= self.nt):
            self.time_index = 0
            return True
        else:
            return False
        
    def truncate(self):
        """
        truncate 

        Determines whether to truncate the episode based on the PDE state size and the vairable ``limit_pde_state_size`` given in the PDE environment intialization.
        """
        if (
            self.limit_pde_state_size
            and np.linalg.norm(self.u, 2)  >= self.max_state_value
        ):
            return True
        else:
            return False
        
    #Helper functions
    @staticmethod
    def Conversion(ZSecond,TargetLength,scale,dx,M):
        LengthAtT = ZSecond + TargetLength
        W = float(1/scale)  # conversion from physical length -> code length
        LtCode = float(W * LengthAtT)            # convert length to code units
        LRaw   = int(round(LtCode / dx))      # map to grid index
        return max(1, min(M - 1, LRaw))