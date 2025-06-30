.. PDEControlGym documentation master file, created by
   sphinx-quickstart on Sun Dec 24 05:19:45 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PDEContRoLGym Documentation
=========================================

The PDEContRoLGym is a benchmark containing a series of 1D and 2D problems for PDE control. It is designed for control theory by control theorists with the aim for easy use with Reinforcement Learning algorithms. 

Github Repository: https://github.com/lukebhan/PDEControlGym

Paper: https://proceedings.mlr.press/v242/bhan24a/bhan24a.pdf

Pre-Trained Models: https://huggingface.co/lukebhan/PDEControlGymModels

We strongly recommend first installing the gym following the instructions in the `documentation here <../guide/install.html>`_. Then, we recommmend exploring the Jupyter-notebooks in the example tutorial found `here <../guide/tutorials.html>`_.

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

.. toctree:: 
  :maxdepth: 2
  :caption: Tutorials

  tutorials/hyperbolic-1d_tutorial
  tutorials/Trafficarz1d_tutorial

.. toctree::
  :maxdepth: 2
  :caption: Environments

  environments/hyperbolic-1d
  environments/parabolic-1d
  environments/navierstokes2d
  environments/Trafficarz1d
  environments/neuron-1d

.. toctree::
   :maxdepth: 2
   :caption: Custom Environments

   custom_environments/1dbaseenvironment
   custom_environments/2dbaseenvironment

.. toctree::
  :maxdepth: 2 
  :caption: Utilities

  utils/preimplementedrewards
  utils/customrewards

Contributing
------------
Contributions are warmly welcome including testing, bugs, and features. Please see the `github <https://github.com/lukebhan/PDEControlGym>`_ and the `contribution guidelines <https://github.com/lukebhan/PDEControlGym/blob/main/CONTRIBUTING.md>`_.

Citing
------
To cite this project in publications, please use the following reference:

.. code-block:: bibtex

  @inproceedings{bhan2024pde,
    title={Pde control gym: A benchmark for data-driven boundary control of partial differential equations},
    author={Bhan, Luke and Bian, Yuexin and Krstic, Miroslav and Shi, Yuanyuan},
    booktitle={6th Annual Learning for Dynamics \& Control Conference},
    pages={1083--1095},
    year={2024},
    organization={PMLR}
  }
