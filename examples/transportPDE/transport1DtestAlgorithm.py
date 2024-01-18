import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from pde_control_gym.src import TunedReward1D
import pde_control_gym

# THIS EXAMPLE TEST A SERIES OF ALGORITHMS AND CALCULATES THE AVERAGE REWARD OF EACH OVER 1K SAMPLES

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
# set to 7.35, but this can be modified for further problesms
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx), 7.35)

# Timestep and spatial step for PDE Solver
# Run testing cases for 5 seconds
T = 5
dt = 1e-4
dx = 1e-2
X = 1

# Normalize to be set below
hyperbolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize":None, 
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

# Parameter varies. For SAC and PPO it is the model itself
# For backstepping it is the beta function
def runSingleEpisode(model, env, parameter):
    terminate = False
    truncate = False

    # Holds the resulting states
    uStorage = []

    # Reset Environment
    obs,__ = env.reset()
    uStorage.append(obs)

    i = 0
    rew = 0
    while not truncate and not terminate:
        # use backstepping controller
        action = model(obs, parameter)
        obs, rewards, terminate, truncate, info = env.step(action)
        uStorage.append(obs)
        rew += rewards 
    u = np.array(uStorage)
    return rew, u

def bcksController(obs, beta):
    kernel = solveKernelFunction(beta)
    return solveControl(kernel, obs)

def RLController(obs, model):
    action, _state = model.predict(obs)
    return action

# Make the hyperbolic PDE gym
# Backstepping does not normalize the control inputs while RL algoriths do
hyperbolicParametersBackstepping = hyperbolicParameters.copy()
hyperbolicParametersBackstepping["normalize"] = False

hyperbolicParametersRL = hyperbolicParameters.copy()
hyperbolicParametersRL["normalize"] = True

envBcks = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersBackstepping)
envRL = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParametersRL)

# Number of test cases to run
num_instances = 10

# For backstepping controller
spatial = np.linspace(dx, X, int(round(X/dx)))
beta = solveBetaFunction(spatial, 7.35)

# Load RL models. # DUMMY ARGUMENTS NEED TO BE MODIFIED
ppoModelPath = "./logsPPO/rl_model_10000_steps"
sacModelPath = "./logsSAC/rl_model_10000_steps"
ppoModel = PPO.load(ppoModelPath)
sacModel = SAC.load(sacModelPath)

# Run comparisons
# Backstepping
total_bcks_reward = 0
for i in range(num_instances):
    rew, _ = runSingleEpisode(bcksController, envBcks, beta)
    total_bcks_reward += rew
print("Backstepping Reward Average:", total_bcks_reward/num_instances)

# PPO
total_ppo_reward = 0
for i in range(num_instances):
    rew, _ = runSingleEpisode(RLController, envRL, ppoModel)
    total_ppo_reward += rew
print("PPO Reward Average:", total_ppo_reward/num_instances)

# SAC
total_sac_reward = 0
for i in range(num_instances):
    rew, _ = runSingleEpisode(RLController, envRL, sacModel)
    total_sac_reward += rew
print("SAC Reward Average:", total_sac_reward/num_instances)
