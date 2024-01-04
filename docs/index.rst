.. PDEControlGym documentation master file, created by
   sphinx-quickstart on Sun Dec 24 05:19:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PDEContRoLGym Documentation
=========================================

The PDEContRoLGym is a benchmark containing a series of 1D and 2D problems for PDE control. It is designed for control theory by control theorists with the aim for easy use with Reinforcement Learning algorithms. 

Github Repository: https://github.com/lukebhan/PDEControlGym

Paper: https://arxiv.org/abs/2302.14265

Pre-Trained Models: https://huggingface.co/lukebhan/PDEControlGymModels

Main Features
--------------

- Fully worked examples
- Plug and play with any RL gym. 
- Designed for control theory - not just "PDE solvers"
- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes

.. toctree:: 
  :maxdepth: 2
  :caption: User Guide

  guide/install
  guide/quickstart
  guide/examples

.. toctree::
  :maxdepth: 2
  :caption: Environments

  environments/hyperbolic-1d
  environments/parabolic-1d
  environments/navier-stokes-2d

Contributing
------------
Contributions are warmly welcome including testing, bugs, and features. Please see the `github <https://github.com/lukebhan/PDEControlGym>`_ and the `contribution guidelines <https://github.com/lukebhan/PDEControlGym/blob/main/CONTRIBUTING.md>`_.

Citing
------
To cite this project in publications, please use the following reference:

.. code-block:: bibtex

	@misc{bhan2023neural,
		title={Neural Operators for Bypassing Gain and Control Computations in PDE Backstepping}, 
		author={Luke Bhan and Yuanyuan Shi and Miroslav Krstic},
		year={2023},
		eprint={2302.14265},
		archivePrefix={arXiv},
		primaryClass={eess.SY}
	}

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
