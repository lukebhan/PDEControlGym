import gymnasium as gym
import pde_control_gym
from pde_control_gym.src import TunedReward1D
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# THIS EXAMPLE TRAINS A PPO AGENT FOR THE HYPERBOLIC PDE PROBLEM. 
# The model is saved every 10k timesteps to the directory ./logsPPO/
# The tensorboard results are saved to the directory
# ./tb/

# NO NOISE
def noiseFunc(state):
    return state

# Chebyshev Polynomial Beta Functions
def solveBetaFunction(x, gamma):
    beta = np.zeros(len(x), dtype=np.float32)
    for idx, val in enumerate(x):
        beta[idx] = 5*math.cos(gamma*math.acos(val))
    return beta

# Kernel function solver for backstepping
def solveKernelFunction(theta):
    kappa = np.zeros(len(theta))
    for i in range(0, len(theta)):
        kernelIntegral = 0
        for j in range(0, i):
            kernelIntegral += (kappa[i-j]*theta[j])*dx
        kappa[i] = kernelIntegral  - theta[i]
    return np.flip(kappa)

# Control convolution solver
def solveControl(kernel, u):
    res = 0
    for i in range(len(u)):
        res += kernel[i]*u[i]
    return res*1e-2

# Set initial condition function here
def getInitialCondition(nx):
    return np.ones(nx)*np.random.uniform(1, 10)

# Returns beta functions passed into PDE environment. Currently gamma is always
# set to 7.35, but this can be modified for further problems
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx), 7.35)

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
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize":True, 
        "sensing_loc": "full", 
        "control_type": "Dirchilet", 
        "sensing_type": None,
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "control_sample_rate": 0.1,
}
env = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParameters)

# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path="./logsPPO",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = PPO("MlpPolicy",env, verbose=1, tensorboard_log="./tb/")
# Train for 1 Million timesteps
model.learn(total_timesteps=1e6, callback=checkpoint_callback)
