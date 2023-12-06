import gymnasium as gym
import pdecontrolgym
import numpy as np
import math
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def noiseFunc(state):
    return state

# Chebyshev Polynomial Beta Functions
def solveBetaFunction(x, gamma):
    beta = np.zeros(len(x), dtype=np.float32)
    for idx, val in enumerate(x):
        beta[idx] = 5*math.cos(gamma*math.acos(val))
    return beta

def solveKernelFunction(theta):
    kappa = np.zeros(len(theta))
    for i in range(0, len(theta)):
        kernelIntegral = 0
        for j in range(0, i):
            kernelIntegral += (kappa[i-j]*theta[j])*dx
        kappa[i] = kernelIntegral  - theta[i]
    return np.flip(kappa)

def solveControl(kernel, u):
    res = 0
    for i in range(len(u)):
        res += kernel[i]*u[i]
    return res*1e-3

def getInitialCondition(nx):
    return np.ones(nx)*5

def getBetaFunction(nx, X, gamma):
    return solveBetaFunction(np.linspace(0, X, nx), gamma)

T = 5
dt = 1e-4
dx = 1e-3
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
        "max_state_value": 100, 
        "max_control_value": 25,
        "reward_norm": 2, 
        "reward_horizon": "t-horizon",
        "reward_average_length": 10,
        "truncate_penalty": -10000, 
        "terminate_reward": 1e2, 
        "reset_init_condition_func": getInitialCondition,
        "reset_recirculation_func": getBetaFunction,
        "reset_recirculation_param": 3,
}

env = gym.make("PDEControlGym-HyperbolicPDE1D", hyperbolicParams=hyperbolicParameters)

terminate = False
truncate = False
nt = int(round(X/dx))
x = np.linspace(0, 1, nt)
uStorage = []

model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=250000)

model.save("firstTest")

del model
model = PPO.load("firstTest")

obs,__ = env.reset()
uStorage.append(obs)

spatial = np.linspace(dx, X, int(round(X/dx)))
kernel = solveKernelFunction(solveBetaFunction(spatial, 3))
i = 0
while not truncate and not terminate:
    # action, _state = model.predict(obs)
    action = solveControl(kernel, obs)
    print(action)
    print(obs)
    i+=1
    if i > 2:
        break
    obs, rewards, terminate, truncate, info = env.step(action)
    uStorage.append(obs)


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
    axis.set_pane_color((0, 0, 0))
    
meshx, mesht = np.meshgrid(spatial, temporal)
                     
axes.plot_surface(meshx, mesht, u, edgecolor="black",lw=0.2, rstride=500, cstride=40, 
                        alpha=1, color="white", shade=False, rasterized=True, antialiased=True)
axes.view_init(10, 15)
axes.set_xlabel("x")
axes.set_ylabel("Time")
axes.set_zlabel(r"$u(x, t)$", rotation=90)
axes.zaxis.set_rotate_label(False)
axes.set_xticks([0, 0.5, 1])
plt.show()
