import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from pde_control_gym.src import TunedReward1D
import pde_control_gym

# THIS EXAMPLE SOLVES THE HYPERBOLIC PDE PROBLEM USING A BACKSTEPPING CONTROLLER

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
        "normalize":False, 
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

# Make the hyperbolic PDE gym
env = gym.make("PDEControlGym-TransportPDE1D", **hyperbolicParameters)

# Run a single environment test case for gamma=7.35
terminate = False
truncate = False
nt = int(round(X/dx))
x = np.linspace(0, 1, nt)

# Holds the resulting states
uStorage = []

# Reset Environment
obs,__ = env.reset()
uStorage.append(obs)

spatial = np.linspace(dx, X, int(round(X/dx)))
kernel = solveKernelFunction(solveBetaFunction(spatial, 7.35))
i = 0
rew = 0
while not truncate and not terminate:
    # use backstepping controller
    action = solveControl(kernel, obs)
    obs, rewards, terminate, truncate, info = env.step(action)
    uStorage.append(obs)
    rew += rewards 
u = np.array(uStorage)
print("Total Reward", rew)

# Plot the example
res = 1
fig = plt.figure()
spatial = np.linspace(dx, X, int(round(X/dx)))
temporal = np.linspace(0, T, len(uStorage))
u = np.array(uStorage)

subfigs = fig.subfigures(nrows=1, ncols=1, hspace=0)

subfig = subfigs
subfig.subplots_adjust(left=0.07, bottom=0, right=1, top=1.1)
axes = subfig.subplots(nrows=1, ncols=1, subplot_kw={"projection": "3d", "computed_zorder": False})

for axis in [axes.xaxis, axes.yaxis, axes.zaxis]:
    axis._axinfo['axisline']['linewidth'] = 1
    axis._axinfo['axisline']['color'] = "b"
    axis._axinfo['grid']['linewidth'] = 0.2
    axis._axinfo['grid']['linestyle'] = "--"
    axis._axinfo['grid']['color'] = "#d1d1d1"
    axis.set_pane_color((1,1,1))
    
meshx, mesht = np.meshgrid(spatial, temporal)
                     
axes.plot_surface(meshx, mesht, u, edgecolor="black",lw=0.2, rstride=50, cstride=1, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
axes.view_init(10, 15)
axes.set_xlabel("x")
axes.set_ylabel("Time")
axes.set_zlabel(r"$u(x, t)$", rotation=90)
axes.zaxis.set_rotate_label(False)
axes.set_xticks([0, 0.5, 1])
plt.show()
