 <a href="#"><img alt="PDE ContRoL Gym" src="PDEGymLogo.png" width="100%"/></a>

<p>
<a href='https://pdecontrolgym.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pdecontrolgym/badge/?version=latest' alt='Documentation Status' />
    </a>
<a href=https://arxiv.org/abs/2302.14265> 
    <img src="https://img.shields.io/badge/arXiv-2302.14265-008000.svg" alt="arXiv Status" />
</a>
</p>

## About this repository

This repository provides implementations of various PDEs including hyperbolic PDEs(Burger's Equation) and parabolic PDEs (Reaction-Diffusion/Heat Equations), and 2-D Navier Stokes PDEs that are all wrapped under a <a href=https://github.com/Farama-Foundation/Gymnasium>gym</a> designed for **control algorithms**.

## Installation
Installing PDE Control Gym is simple, requires few dependencies, and is set to work with most Python Environments. This installation guide assumes you are familiar with either <a href=https://www.anaconda.com/>conda</a> or <a href=https://docs.python.org/3/library/venv.html>virtual env</a> in creating and installing python modules.

- First create a new enviornment for ***Python 3.10+***
- Install the dependencies. PDE ContRoL Gym requires
  - Gym Version 0.28.1 (<a href=https://github.com/Farama-Foundation/Gymnasium>Available here</a>)
  - Numpy Version 1.26.2 (<a href=https://numpy.org/>Available here</a>)
- To run the examples listed below or any reinforcement learning algorithm, the gym is currently setup using <a href=https://stable-baselines3.readthedocs.io/en/master/>Stable-Baselines3</a> Version 2.2.1.
- Once dependencies are installed, clone the repository using
  
  `git clone https://github.com/lukebhan/PDEControlGym.git`
- Then navigate to the working folder
  
  `cd PDEControlGym`
- Install the package using pip in the main folder
  
  `pip install -e .`
- Enjoy your fully functional PDE Control library! For usage, see the examples in either `./PDEControlGym/examples` or in the <a href=https://pdecontrolgym.readthedocs.io/en/latest/>documentation</a> or below in the examples section of this file!

## Documentation
Documentation for all the environments as well as tutorials are available <a href=https://pdecontrolgym.readthedocs.io/en/latest/>here</a>

## Examples
Jupyter notebooks are made in the examples folder (Hyperbolic and Parabolic subfolders) that detail all the examples for both Hyperbolic and Parabolic PDEs. Additionally, full details on numerical implementation and solvers are available in the 
<a href=https://pdecontrolgym.readthedocs.io/en/latest/>documentation</a>. These include tutorials on how to train your own RL controller, implemented backstepping controllers as well as plotting comparisons to analyze all of your work. For managing the training of your RL controller, we recommend you use <a href=https://github.com/tensorflow/tensorboard>tensorboard</a> as outlined in the notebooks. For pretrained models, please see the downloaded files on <a href=https://huggingface.co/lukebhan/PDEControlGymModels>Hugging Face</a>.  

## Contributions
Contributions to the gym are both welcome and encouraged. Please see the <a href=https://github.com/lukebhan/PDEControlGym/blob/main/CONTRIBUTING.md>contribution guidelines</a>. Any questions are welcome to email the author Luke at lbhan@ucsd.edu

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2302.14265,
  doi = {10.48550/ARXIV.2302.14265},
  url = {https://arxiv.org/abs/2302.14265},
  author = {Bhan, Luke and Shi, Yuanyuan and Krstic, Miroslav},
  keywords = {Systems and Control (eess.SY), Optimization and Control (math.OC), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Mathematics, FOS: Mathematics},
  title = {Neural Operators for Bypassing Gain and Control Computations in PDE Backstepping},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
  }
```

## Licensing
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
