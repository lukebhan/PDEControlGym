import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from pde_control_gym.src import TunedReward1D
import pde_control_gym

# THIS EXAMPLE SOLVES THE PARABOLIC PDE PROBLEM USING A BACKSTEPPING CONTROLLER

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

parabolicParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": TunedReward1D(int(round(T/dt)), -1e3, 3e2),
        "normalize": False,
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

# Make the hyperbolic PDE gym
env = gym.make("PDEControlGym-ReactionDiffusionPDE1D", **parabolicParameters)

# Run a single environment test case for gamma=8
terminate = False
truncate = False
nt = int(round(X/dx))
x = np.linspace(0, 1, nt+1)

# Holds the resulting states
uStorage = []

# Reset Environment
obs,__ = env.reset()
uStorage.append(obs)

spatial = np.linspace(dx, X, int(round(X/dx))+1)
kernel = solveKernelFunction(solveBetaFunction(spatial, 8))
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
spatial = np.linspace(0, X, int(round(X/dx))+1)
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
                     
axes.plot_surface(meshx, mesht, u, edgecolor="black",lw=0.2, rstride=500, cstride=5, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
axes.view_init(10, 15)
axes.set_xlabel("x")
axes.set_ylabel("Time")
axes.set_zlabel(r"$u(x, t)$", rotation=90)
axes.zaxis.set_rotate_label(False)
axes.set_xticks([0, 0.5, 1])
plt.show()
