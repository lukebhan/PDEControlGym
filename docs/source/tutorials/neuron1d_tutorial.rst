.. _neuron-1d_tutorial:

1D Neuron Growth Control PDE Tutorial
=========================

This tutorial will follow the Jupyter-notebooks found at Github. We will explore the `Neuron Growth Control PDE Environment <https://pdecontrolgym.readthedocs.io/en/latest/environments/neuron-1d.html>`_, 
which consists of 1D PDEs and a system of coupled ODEs with a moving boundary.  The control objective is to reach a target 
tubulin concentration across the neuron, while also reaching a target length in the neuron's axon.



We first start by initializing the Neuron-1d environment. The environment is initialized with the following parameters:

.. code-block:: python

    Parameters = { 
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
                 gamma: float = (a/D)*10
                 H: float = np.zeros([2,1], dtype=float)
                 H[0,0]: float = 1
                 H[1,0]: float = -1*(((a - g*lsubc)*(c_infty))/D)
                atilde1: float = ((a - rsubg*c_infty)/lsubc) - g - tildeRsubg
                atilde3: float = (a**2 + D*g - a*g*lsubc) / (D*2)
                Asub1: np.array([[atilde1, 0.0],
                       [rsubg, 0.0]], dtype=float)
                beta: float = D/lsubc
                B = np.zeros([2,1], dtype=float)
                B[0,0]: float = -1*beta
                lambda_minus: float = (a - np.sqrt(a**2 + 4*D*g))/(2*D)
                lambda_plus: float = a + np.sqrt(a**2 + 4*D*g)/(2*D)
                Kminus: float = 0.5 - ((a - 2*g*lsubc)/2/(np.sqrt(a**2 + 4*D*g)))
                Kplus: float = ((a - 2*g*lsubc)/2/(np.sqrt(a**2 + 4*D*g))) + 0.5
                dt = 5e-6
                dx = 0.01
                T = 180
    }

Detailed explanations about the parameters can be found in the `Neuron Growth Control PDE Environment <https://pdecontrolgym.readthedocs.io/en/latest/environments/neuron-1d.html>`_ documentation.


Let :math:`u(x,t)` and :math:`Z(t)` represent the error of the tubulin concentration of the neuron across its length and an error
system for the axon length and for tubulin concentration in the growth cone of the neuron. The control objective is to bring these error systems
to 0, letting concentrations and length converge to :math:`C_eq(x)` and :math:`l_s`. The PDE backstepping controller is designed to achieve
the control objective by applying a control input at :math:`u(x = 0,t)`

We first define some gain kernel parameters to setup the backstepping controller:

:math:`N1 = \begin{bmatrix} 0 & \frac{1}{D} \left(gI + A_1 + \frac{a}{D}BH^T\right) \\ I & \frac{1}{D} \left(BH^T + aI\right) \end{bmatrix}`

:math:`K = [-1e3\quad 1e13]^T` (Feedback Control Gain Vector)

:math:`\phi(x)^T = [H^T\quad K^T - \frac{1}{D}H^TBH^T]e^{N_1x}\begin{bmatrix} I \\ 0  \end{bmatrix}` (Gain Kernel Function)

:math:`p(x) = \phi'(-x)^T - \gamma\phi(-x)^T`

We then define the backstepping controller by:

:math:`U(t) = \left(\frac{1}{D}H^TB + \gamma\right)u(0,t) - \frac{1}{D}\int_0^{l(t)} p(x) B u(x,t)\,dx + p(l(t))Z(t)`

Lastly, we set :math:`u(x = 0,t) = U(t)`

We implement this in the following way:

.. code-block:: python
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