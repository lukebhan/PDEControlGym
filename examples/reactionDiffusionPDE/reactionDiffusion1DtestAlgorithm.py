import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import pde_control_gym
from pde_control_gym.src import TunedReward1D

# THIS EXAMPLE TEST A SERIES OF ALGORITHMS AND CALCULATES THE AVERAGE REWARD OF EACH OVER 1K SAMPLES

# NO NOISE
def noiseFunc(state):
    return state

# Chebyshev Polynomial Beta Functions
def solveBetaFunction(x, gamma):
    beta = np.zeros(len(x), dtype=np.float32)
    for idx, val in enumerate(x):
        beta[idx] = 50*math.cos(gamma*math.acos(val))
    return beta

# Kernel function solver for backstepping
def solveKernelFunction(beta):
    k = np.zeros((len(beta), len(beta)))
    # First we calculate a at each timestep
    a = beta

    # FD LOOP
    k[1][1] = -(a[1] + a[0]) * dx / 4
    for i in range(1, len(beta)-1):
        k[i+1][0] = 0
        k[i+1][i+1] = k[i][i]-dx/4.0*(a[i-1] + a[i])
        k[i+1][i] = k[i][i] - dx/2 * a[i]
        for j in range(1, i):
                k[i+1][j] = -k[i-1][j] + k[i][j+1] + k[i][j-1] + a[j]*(dx**2)*(k[i][j+1]+k[i][j-1])/2
    return k

# Control convolution solver
def solveControl(kernel, u):
    return sum(kernel[-1][0:len(u)-1]*u[0:len(u)-1])*dx

# Set initial condition function here
def getInitialCondition(nx):
    return np.ones(nx+1)*np.random.uniform(1, 10)

# Returns beta functions passed into PDE environment. Currently gamma is always
# set to 8, but this can be modified for further problems
def getBetaFunction(nx):
    return solveBetaFunction(np.linspace(0, 1, nx+1), 8)

# Timestep and spatial step for PDE Solver
T = 1
dt = 1e-5
dx = 5e-3
X = 1

# Backstepping does not need to normalize actions to be between -1 and 1, so normalize is set to False. Otherwise,
# parameters are same as RL algorithms
parabolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize": None,
        "sensing_loc": "full", 
        "control_type": "Dirchilet", 
        "sensing_type": None,
        "sensing_noise_func": lambda state: state,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 20,
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "control_sample_rate": 0.001,
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

# Make the parabolic PDE gym
# Backstepping does not normalize the control inputs while RL algoriths do
parabolicParametersBackstepping = parabolicParameters.copy()
parabolicParametersBackstepping["normalize"] = False

parabolicParametersRL = parabolicParameters.copy()
parabolicParametersRL["normalize"] = True

envBcks = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersBackstepping)
envRL = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParametersRL)

# Number of test cases to run
num_instances = 10

# For backstepping controller
spatial = np.linspace(dx, X, int(round(X/dx)))
beta = solveBetaFunction(spatial, 7.35)

# Load RL models. # DUMMY ARGUMENTS NEED TO BE MODIFIED
ppoModelPath = "./logsPPO/rl_model_1000_steps"
sacModelPath = "./logsSAC/rl_model_1000_steps"
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
