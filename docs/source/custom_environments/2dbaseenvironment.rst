.. _2D-CustomEnvironments:

.. automodule:: pde_control_gym.src.environments2d.base_env_2d

2D Custom Environments
======================

Any 2D custom environment can easily be added to the PDE ContRoL Gym. It is best to follow the current environments format to ensure project structure errors are avoided. The following steps are required to add a custom 2D environment. 

#. Build you environment (in the folder ``pde_control_gym/src/environments2d/``) implementing all the required functions by inheriting the base class :class:`PDEEnv2D`. All the details about the base class are given below and it is probably helpful to look at the source code for the other 2D environments in ``pde_control_gym/src/environments2d/`` for example templates.

#. Update ``__init__.py`` files in each of the following folders:

   * ``pde_control_gym/src/environments2d/__init__.py``
        Import your class containing the environment and add it to the ``__all__`` array.

   * ``pde_control_gym/src/__init__.py``
        Again import your class from the ``environments2d`` folder and again add it to the ``__all__`` array.

   * ``pde_control_gym/__init__.py`` 
        Register your new environment by giving the environment an ``id`` and set the ``entry_point`` to ``"pde_control_gym.src:{Insert Class Name}"``.

#. Reinstall environment by going to the root directory and running ``pip install -e .``.

#. Use your new environment by importing the gym and making the environment!


   .. code-block:: python

       import pde_control_gym
       env = gym.make("{id_given_in_step_2}", params="{Your env parameters}")

Base 2D Environment Class
-------------------------

.. autoclass:: PDEEnv2D
   :members:
