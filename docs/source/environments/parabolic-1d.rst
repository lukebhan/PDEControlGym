.. _parabolic:

.. automodule:: pde_control_gym.src.environments1d

Reaction-Diffusion 1D PDE
=========================

This documentation is for the 1D Reation-Diffusion PDE Environment defined by the boundary control problem

.. math::
    :nowrap:

	\begin{eqnarray}	
        u_t(x, t) &=& u_{xx}(x, t) + \lambda(x) u(0, t), \quad x \in [0, X], t \in [0, T]\,,\\ 
		u(0, t) &=& 0 \,,
	\end{eqnarray}

where :math:`\lambda(x)` is the nonlinear, spatial-varying recirculation plant coefficient and control can be applied as either second boundary condition following

.. math::
    :nowrap:

    \begin{eqnarray}
        u(X, t) &=& U(t) \qquad \text{Dirchilet Boundary Conditions}\,,\\
        u_x(X, t) &=& U(t) \qquad \text{Neumann Boundary Conditions}\,.
    \end{eqnarray}

This problem is the simplest possible Parabolic PDE benchmark and thus is a good indicator of potentially algorithmic success on more difficult problems. The goal of the problem is to stabilize the system (:math:`\lim_{t \to \infty} \|u(x, t)\| = 0`) where the norm can be varied depending on the problem formulation and continuity of the plant coefficient :math:`\lambda(x)`. A variety of sensing options are supported for the implementation as well if one wants to attempt to perform output feedback stabilization. See the parameter list below.

.. autoclass:: ReactionDiffusionPDE1D
   :members:
   :exclude-members: truncate, terminate

Numerical Implementation
------------------------

We derive the numerical implementation scheme for those looking for inner details of the environment. We use a first-order finite-difference scheme to approximate the PDE leading to super fast implementation speeds - although with sacrifice on spatial and temporal timestep parameterization.

Consider the first-order taylor approximation as

.. math::
    :nowrap:

    \begin{eqnarray}
        u(x, t+1) = u(x, t) + \Delta t u_t(x, t)\,,
    \end{eqnarray}

with finite spatial derviatves approximated by first-order centered diffrences

.. math:: 
    :nowrap:

    \begin{eqnarray} 
        \frac{\partial u}{\partial x} = \frac{u_{j+1}^n - 2u_j^n+u_{j+1}^n}{(\Delta x)^2}\,,
    \end{eqnarray}

where :math:`\Delta t=dt=\text{time step}`, :math:`\Delta x=dx=\text{spatial step}`, :math:`n=0, ..., Nt`, :math:`j=0, ..., Nx`, where :math:`Nt` and :math:`Nx` are the total number of discretized temporal and spatial steps respectively. Then substituting :math:`u_{xx}` and :math:`u_t` into the taylor approximation yields

.. math::
    :nowrap:

    \begin{eqnarray}
        u_j^{n+1} = u_j^n +  \Delta t \left(\frac{u_{j-1}^n - 2 u_j^n + u_{j+1}^n}{(\Delta x)^2} + \lambda_j u_j^n \right) \,.
    \end{eqnarray}

Now, the last thing to consider the is boundary conditions. Begin with the harder Neumann boundary condition. Let :math:`u_\zeta^n|_{\zeta=Nx}` represent the spatial derivative at time :math:`t` of spatial  point :math:`Nx=X` which is given by the user as control input. Then, we have 

.. math::
   :nowrap:
    
    \begin{eqnarray}
        u_\zeta^n|_{\zeta=Nx} = \frac{u_{Nx}^{n} - u_{Nx-1}^n}{\Delta x}\,,
    \end{eqnarray}

which is rearranged for the final boundary point

.. math::
   :nowrap:

    \begin{eqnarray}
        u_{Nx}^n = u_{Nx-1}^n + (\Delta x) u_\zeta^n|_{\zeta=Nx}\,.
    \end{eqnarray}

In the case of Dirchilet boundary conditions, the computation is straightforward as :math:`u_{Nx}^n` is directly set as the given control input. 
