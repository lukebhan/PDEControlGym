.. _1D-CustomEnvironments:

.. automodule:: pde_control_gym.src.environments1d.base_env_1d

1D Custom Environments
======================

Any 1D custom environment can easily be added to the PDE ContRoL Gym. It is best to follow the current environments format to ensure project structure errors are avoided. The following steps are required to add a custom 1D environment. 

1. Build you environment implementing all the required functions by inheriting the base class :class:`PDEEnv1D`. All the details about the base class are given below. 

Base 1D Environment Class
-------------------------

.. autoclass:: PDEEnv1D
   :members:
