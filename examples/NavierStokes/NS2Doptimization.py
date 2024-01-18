import gymnasium as gym
import pdecontrolgym
import numpy as np
import math
import matplotlib.pyplot as plt
from pdecontrolgym.envs.NavierStokes2D import Dirchilet, Controllable, Neumann
from pdecontrolgym.envs.util import central_difference_x, central_difference_y, laplace
import time 
from tqdm import tqdm


# THIS EXAMPLE SOLVES THE NavierStokes PROBLEM based on optimization


Nx = 21
control = {'n_action': 2*Nx}
for i, pos in enumerate(['upper', 'lower', 'left', 'right']):
    if i == 0:
        control[pos] = Controllable({'u': np.arange(Nx*2*i, Nx*(2*i+1)), 'v': np.arange(Nx*(2*i+1), Nx*(2*i+2))})
    else:
        control[pos] = Dirchilet(0)

def adapt_action(action):
    a = np.zeros(2*Nx)
    a[:Nx] = action
    return a

def apply_boundary(a1, a2):
    a1[:,[-1, 0]] = 0.
    a1[[-1,0],:] = 0.
    a2[:,[-1, 0]] = 0.
    a2[[-1,0],:] = 0.
    return a1, a2

# Timestep and spatial step for PDE Solver
T = 0.2
dt = 1e-3
dx, dy = 0.05, 0.05
X, Y = 1, 1
u_target = np.load('target.npz')['u']
v_target = np.load('target.npz')['v']
desire_states = np.stack([u_target, v_target], axis=-1) # (NT, Nx, Ny, 2)
NS2DParameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "Y": Y,
        "dy":dy,
        "control": control, 
        "desire_U": desire_states, 
        "adapt": adapt_action,
        "viscosity": 0.1,
        "density": 1.0,
        "pressure_ite": 2000,
        "stable_factor": 0.5,
        "limit_pde_state_size": True,
        "max_state_value": 1e10,
        "max_control_value": 10,
        "reward_norm": 2, 
        "reward_horizon": "temporal",
        "reward_average_length": 10,
        "truncate_penalty": -1e3, 
        "terminate_reward": 3e2, 
        "normalize": False,
}

# Make the hyperbolic PDE gym
env = gym.make("PDEControlGym-NavierStokes2D", NSParams=NS2DParameters)

# Model-Based Optimization to optimize action 
gamma = 1.0
total_reward = 0.
U, V = [], []
T = 199

rewards = []
times = []
for experiment_i in range(1):
    np.random.seed(experiment_i)
    env.reset(seed=400)
    print(env.U[0,0,0])
    s = time.time()
    for t in tqdm(range(T)):
        obs, reward, done, _ , _ = env.step(np.random.uniform(2,4))
        U.append(env.u)
        V.append(env.v)
        total_reward += reward
    print(total_reward)
    u_target = np.load('target.npz')['u']
    v_target = np.load('target.npz')['v']
    u_ref = [2 for _ in range(T)]
    for ite in range(1):
        Lam1, Lam2 = [], []
        Lam1.append(np.zeros_like(U[0]))
        Lam2.append(np.zeros_like(U[0]))
        pressure = np.zeros_like(U[0])
        for t in tqdm(range(T-1)):
            lam1, lam2 = Lam1[-1], Lam2[-1]
            dl1dx, dl1dy = central_difference_x(lam1, dx), central_difference_y(lam1, dy)
            dl2dx, dl2dy = central_difference_x(lam2, dx), central_difference_y(lam2, dy) 
            laplace_l1, laplace_l2 = laplace(lam1, dx, dy), laplace(lam2, dx, dy)
            dlam1dt = - 2 * dl1dx * U[-1-t] - dl1dy * V[-1-t] - dl2dx * V[-1-t] - 0.1 * laplace_l1 + (U[-1-t]-u_target[-1-t])
            dlam2dt = - 2 * dl2dy * V[-1-t] - dl1dy * U[-1-t] - dl2dx * U[-1-t] - 0.1 * laplace_l2 + (V[-1-t]-v_target[-1-t])
            lam1 = lam1 - dt * dlam1dt
            lam2 = lam2 - dt * dlam2dt
            lam1, lam2 = apply_boundary(lam1, lam2)
            pressure = env.solve_pressure(lam1, lam2, pressure)
            lam1 = lam1 - dt * central_difference_x(pressure, dx)
            lam2 = lam2 - dt * central_difference_y(pressure, dy)
            lam1, lam2 = apply_boundary(lam1, lam2)
            Lam1.append(lam1)
            Lam2.append(lam2)
        Lam1 = Lam1[::-1]
        actions = []
        for t in tqdm(range(T)):
            dl1dx2 = central_difference_y(Lam1[t], dy)
            actions.append(u_ref[t] - 0.1/0.1 * sum(dl1dx2[-2, :])*5*dx)
        U, V, desired_U, desired_V = [], [], [], []
        env.reset(seed=400)
        print(env.U[0,0,0])
        total_reward = 0.
        for t in tqdm(range(T)):
            obs, reward, done, _ , _ = env.step(actions[t])
            U.append(env.u)
            V.append(env.v)
            # desired_U.append(env.u_target[int(env.t*1000)-1])
            # desired_V.append(env.v_target[int(env.t*1000)-1])
            total_reward += reward
        print(reward)
        plt.plot(actions)
        plt.show()
        #np.savez('result/NS_optmization.npz', U=np.array(U), V=np.array(V), desired_U=np.array(desired_U), desired_V=np.array(desired_V))