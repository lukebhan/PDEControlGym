import gymnasium as gym
import pdecontrolgym
import numpy as np
import math
import matplotlib.pyplot as plt
from pdecontrolgym.envs.NavierStokes2D import Dirchilet, Controllable, Neumann
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# THIS EXAMPLE TRAINS A PPO AGENT FOR THE HYPERBOLIC PDE PROBLEM. 
# The model is saved every 10k timesteps to the directory ./logsPPO/
# The tensorboard results are saved to the directory
# ./tb/

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


# Save a checkpoint every 10000 steps
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path="./logsPPO",
  name_prefix="rl_model",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
# Train for 1 Million timesteps
model.learn(total_timesteps=2e5, callback=checkpoint_callback)
