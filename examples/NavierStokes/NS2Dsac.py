import gymnasium as gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time 
from tqdm import tqdm
from pde_control_gym.src import NSReward
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import SAC

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

# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logsSAC",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
# Train for 1 Million timesteps
model.learn(total_timesteps=2e5, callback=checkpoint_callback)
