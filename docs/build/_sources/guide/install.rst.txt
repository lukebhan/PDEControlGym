.. _install:

Installation
============

Prerequisites
-------------
For installation of the prerequisites, we recommend using a virtual environment manager using either pip with `virtualenv <https://virtualenv.pypa.io/en/latest/>`_ or `anaconda <https://www.anaconda.com/>`_.

PDE ContRoL Gym requires a few simple prerequisites. 

**Required**:

* `python <https://www.python.org/>`_ 3.10+
*  `gym <https://github.com/Farama-Foundation/Gymnasium>`_ 0.28.1+
* `numpy <https://numpy.org/>`_ 1.26.2+


**Recommended** \(needed for examples and tutorials\):

* `pytorch <https://pytorch.org/>`_ 1.12.1+
* `stable-baselines3 <https://github.com/DLR-RM/stable-baselines3>`_ 2.2.1+


Installing the PDE ContRoL Gym
------------------------------
Once the prerequisties are installed, one can now install the gym. 

#. Begin by cloning the repository using classic git:
  ``git clone https://github.com/lukebhan/PDEControlGym.git``

#. Once cloned, navigate to the directory:
  ``cd PDEControlGym``

#. Once in the directory, double check that the prerequistes are installed by checking the ``requirements.txt`` file. Finally, install the environment:
  ``pip install -e .``

And Viola! You have just successfully installed the PDE ContRoL Gym. Any issues you may have feel free to open a `github issues <https://github.com/lukebhan/PDEControlGym/issues>`_ with questions and reproducible steps.
