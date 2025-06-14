.. _navierstokes2d:

.. automodule:: pde_control_gym.src.environments2d

Navier-Stokes 2D PDE
====================


This documentation is for the 2D Navier-Stokes PDE Environment defined by the boundary control problem

.. math::

    \begin{eqnarray}
    & \nabla \cdot \vec{u} = 0, \, \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \vec{u}, \,
 x \in [0,1], y \in [0,1], t \in [0, T]   \\
    & \vec{u}(x, 0, t) = U(x,t), \quad \forall x \in [0,1], t \in [0, T] \\
    & \vec{u}(x, 1, t) = \vec{u}(1, y, t) = \vec{u}(0, y, t) = 0, \forall x \in [0,1], y \in [0,1], t \in [0,T]
    \end{eqnarray}

with boundary control input in the top boundary and the velocity at all other boundaries is set to be zero. We incorporate a predictor-corrector scheme for the imcompressive Navier Stokes equations. In the scheme, we denote :math:`\vec{u} = (u(x,y), v(x,y))` is the velocity vector, and :math:`(x,y)` is the spatial position.

.. autoclass:: NavierStokes2D
   :members:
   :exclude-members: truncate, terminate


Numerical Implementation
------------------------

We use the predictor-corrector step to solve the Navier-Stokes 2D problem. The pressure field is solve in an iterative manner. 

First, the predictor step. 

.. math:: 
    :nowrap:

    \begin{eqnarray}
    u^*_{i,j} &= u^n_{i,j} + \Delta t (\nu(\frac{u^n_{i-1,j} - 2 u^n_{i,j} + u^n_{i+1,j}}{(\Delta x)^2} + \frac{u^n_{i,j-1} -2u^n_{i,j}+u^n_{i,j+1}}{(\Delta y)^2}))  \\
    &\quad\quad\quad - \Delta t (u^{n}_{i,j}\frac{u^n_{i+1,j}-u^n_{i-1,j}}{2\Delta x} + v^{n}_{i,j}\frac{u^n_{i,j+1} - u^{n}_{i,j-1}}{2 \Delta y})), \\
    v^*_{i,j} &= v^{n}_{i,j} + \Delta t (\nu(\frac{v^{n}_{i-1,j} - 2 v^{n}_{i,j} + v^{n}_{i+1,j}}{(\Delta x)^2} + \frac{v^{n}_{i,j-1} -2v^{n}_{i,j}+v^{n}_{i,j+1}}{(\Delta y)^2}))  \\
    & \quad\quad\quad - \Delta t (u^{n}_{i,j}\frac{v^{n}_{i+1,j}-v^{n}_{i-1,j}}{2\Delta x} + v^{n}_{i,j}\frac{v^{n}_{i,j+1} - v^{n}_{i,j-1}}{2 \Delta y})
    \end{eqnarray}


Second, solve the pressure for the continuity condition. 

.. math:: 
    :nowrap:

    \begin{eqnarray}
    \nabla^2 p = \frac{\partial^2 p}{\partial x^2} +  \frac{\partial^2 p}{\partial y^2} = -\rho (\frac{\partial^2 u}{\partial x^2} + 2 \frac{\partial u}{\partial x}\frac{\partial v}{\partial y} + \frac{\partial^2 v}{\partial y^2})
    \end{eqnarray}

Third, perform the corrector step.

.. math:: 
    :nowrap:

    \begin{eqnarray}
    &  u^{n+1}_{i,j} = u^*_{i,j} - \Delta t \cdot \frac{1}{\rho} \frac{p^*_{i+1,j}-p^*_{i-1,j}}{2\Delta x} \\
    &  v^{n+1}_{i,j} = v^*_{i,j} - \Delta t \cdot \frac{1}{\rho} \frac{p^*_{i,j+1} - p^*_{i,j-1}}{\Delta y}
    \end{eqnarray}


We apply boundary conditions every time step after the predictor step and corrector step. 