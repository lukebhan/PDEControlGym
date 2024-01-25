.. _preimplementedrewards:

.. automodule:: pde_control_gym.src.rewards

Pre-implemented Rewards
=======================

This gym comes with a series of pre-implemented reward functions that are detailed in the documentation here. See `this page <../utils/customrewards.html>`_ for implementing your own custom rewards either in the gym or using an outside function as in `the tutorial doc <../guide/tutorials.html>`_.

Normalized Reward
-----------------

.. autoclass:: NormReward
  :members:

Tuned 1D Custom Reward
----------------------

This implements the reward as used in the `benchmark paper <https://google.com>`_ which is defined as:

.. math::
   :nowrap:

   \begin{eqnarray}
     Reward(t) = \begin{cases}
        \text{truncate_penalty}*(T-t) & \text{truncate}=\text{True} \\ 
       \text{terminate_reward} - \sum_{i=0}^{nt}|u(-1, i)| / 1000 
        - \|u(x, T)\|_{L_2} & \text{terminate}=\text{True} \\ & \text{and} \\ &  \|u(x, T)\|_{L_2} < 20 \\
        \|u(x, t-dt*100\|_{L_2} - \|u(x, t)\|_{L_2} & \text{Otherwise}
        \end{cases}
   \end{eqnarray}

where :math:`u(x, t)` is the solution vector, T is final simulation time, and :math:`\|u(x, t)\|_{L_2}` represents the :math:`L_2` norm at time :math:`t` over :math:`x`.

.. autoclass:: TunedReward1D
   :members:



NS Reward 
----------------------

This implements the reward to track the reference trajectory as well as minimizing the control action loss which is defined as:

.. math::
   :nowrap:

   \begin{eqnarray}
     Reward(t) = -\frac{1}{2} \|s' - s_{ref}\|^2 - \frac{\gamma}{2} \| a - a_{ref}\|^2
   \end{eqnarray}

where :math:`\gamma` is the coefficient for the control cost.


.. autoclass:: NSReward
   :members:
