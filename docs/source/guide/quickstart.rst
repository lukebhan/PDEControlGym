.. _quickstart:

Quick Start
===============

A quick example to use the gym is as follows. This import the hyperbolic PDE gym, sets some basic parameters
and runs an open-loop controller. Detailed tutorials are avilable for the gym `here <../guide/tutorials.html>`_.

.. code-block:: python

	import gymnasium as gym
	import pdecontrolgym
	import numpy as np


	# NO NOISE
	def noiseFunc(state):
		return state


	# Set initial condition function here
	def getInitialCondition(nx):
		return np.ones(nx) * np.random.uniform(1, 10)


	# Constant beta function
	def getBetaFunction(nx, X):
		return np.ones(nx)


	# Timestep and spatial step for PDE Solver
	T = 5
	dt = 1e-4
	dx = 1e-2
	X = 1

	hyperbolicParameters = {
		"T": T,
		"dt": dt,
		"X": X,
		"dx": dx,
		"sensing_loc": "full",
		"control_type": "Dirchilet",
		"sensing_type": None,
		"sensing_noise_func": lambda state: state,
		"limit_pde_state_size": True,
		"max_state_value": 1e10,
		"max_control_value": 20,
		"reward_norm": 2,
		"reward_horizon": "temporal",
		"reward_average_length": 10,
		"truncate_penalty": -1e3,
		"terminate_reward": 3e2,
		"reset_init_condition_func": getInitialCondition,
		"reset_recirculation_func": getBetaFunction,
		"control_sample_rate": 0.1,
		"normalize": False,
	}

	# Make the hyperbolic PDE gym
	env = gym.make("PDEControlGym-HyperbolicPDE1D", hyperbolicParams=hyperbolicParameters)

	# Run a single environment test case for gamma=7.35
	terminate = False
	truncate = False

	# Reset Environment
	obs, __ = env.reset()
	totalReward = 0

	while not truncate and not terminate:
		# 0 action input
		obs, rewards, terminate, truncate, info = env.step(0)
		totalReward += rewards
	print(totalReward)
