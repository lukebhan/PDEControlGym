import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
from pde_control_gym.src.environments2d.navier_stokes2D import central_difference, laplace
import time 
from tqdm import tqdm
from pde_control_gym.src import NSReward


# THIS EXAMPLE SOLVES THE NavierStokes PROBLEM based on optimization

# Set initial condition function here
def getInitialCondition(X):
    u = np.random.uniform(-5, 5) * np.ones_like(X) 
    v = np.random.uniform(-5, 5) * np.ones_like(X) 
    p = np.random.uniform(-5, 5) * np.ones_like(X) 
    return u, v, p

# Set up boundary conditions here
boundary_condition = {
    "upper": ["Controllable", "Dirchilet"], 
    "lower": ["Dirchilet", "Dirchilet"], 
    "left": ["Dirchilet", "Dirchilet"], 
    "right": ["Dirchilet", "Dirchilet"], 
}

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
        "action_dim": 1, 
        "reward_class": NSReward(0.1),
        "normalize": False, 
        "reset_init_condition_func": getInitialCondition,
        "boundary_condition": boundary_condition,
        "U_ref": desire_states, 
        "action_ref": 2.0 * np.ones(1000), 
}

# Make the NavierStokes PDE gym
env = gym.make("PDEControlGym-NavierStokes2D", **NS2DParameters)

# Model-Based Optimization to optimize action 
def apply_boundary(a1, a2):
    a1[:,[-1, 0]] = 0.
    a1[[-1,0],:] = 0.
    a2[:,[-1, 0]] = 0.
    a2[[-1,0],:] = 0.
    return a1, a2

total_reward = 0.
U, V = [], []
T = 199

rewards = []
times = []
for experiment_i in range(1):
    np.random.seed(experiment_i)
    env.reset(seed=400)
    s = time.time()
    for t in tqdm(range(T)):
        obs, reward, done, _ , _ = env.step(np.random.uniform(2,4)) 
        U.append(env.u)
        V.append(env.v)
        total_reward += reward
    print("Total Reward:", total_reward)
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
            dl1dx, dl1dy = central_difference(lam1,"x",dx), central_difference(lam1, "y", dy)
            dl2dx, dl2dy = central_difference(lam2,"x", dx), central_difference(lam2, "y", dy) 
            laplace_l1, laplace_l2 = laplace(lam1, dx, dy), laplace(lam2, dx, dy)
            dlam1dt = - 2 * dl1dx * U[-1-t] - dl1dy * V[-1-t] - dl2dx * V[-1-t] - 0.1 * laplace_l1 + (U[-1-t]-u_target[-1-t])
            dlam2dt = - 2 * dl2dy * V[-1-t] - dl1dy * U[-1-t] - dl2dx * U[-1-t] - 0.1 * laplace_l2 + (V[-1-t]-v_target[-1-t])
            lam1 = lam1 - dt * dlam1dt
            lam2 = lam2 - dt * dlam2dt
            lam1, lam2 = apply_boundary(lam1, lam2)
            pressure = env.solve_pressure(lam1, lam2, pressure)
            lam1 = lam1 - dt * central_difference(pressure, "x", dx)
            lam2 = lam2 - dt * central_difference(pressure, "y", dy)
            lam1, lam2 = apply_boundary(lam1, lam2)
            Lam1.append(lam1)
            Lam2.append(lam2)
        Lam1 = Lam1[::-1]
        actions = []
        for t in tqdm(range(T)):
            dl1dx2 = central_difference(Lam1[t], "y", dy)
            actions.append(u_ref[t] - 0.1/0.1 * sum(dl1dx2[-2, :])*5*dx)
        U, V = [], []
        env.reset(seed=400)
        total_reward = 0.
        for t in tqdm(range(T)):
            obs, reward, done, _ , _ = env.step(actions[t])
            U.append(env.u)
            V.append(env.v)
            total_reward += reward
        plt.plot(actions)
        plt.show()
        np.savez('result/NS_optmization.npz', U=env.U[:,:,:,0], V=env.U[:,:,:,1], desired_U=np.array(u_target), desired_V=np.array(v_target), actions=actions)
        
