# %load_ext autoreload
# %autoreload 2

import gymnasium as gym
import pde_control_gym
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt

import stable_baselines3
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

T = 240
dt = 0.25
dx = 10
X = 500

from pde_control_gym.src import TrafficARZReward
reward_class =  TrafficARZReward()

Parameters = {
        "T": T, 
        "dt": dt, 
        "X": X,
        "dx": dx, 
        "reward_class": reward_class,
        "simulation_type" : 'outlet-train', 
        "v_steady" : 10,
        "ro_steady" : 0.12,
        "v_max" : 40,
        "ro_max" : 0.16,
        "tau" : 60, 
        "limit_pde_state_size" : True
        # "control_freq" : 6,
        # "reward_buffer_size" : 60
}
def runSingleEpisode(model, env, parameter = None):
    terminate = False
    truncate = False

    # Holds the resulting states 
    uStorage = []

    # Reset Environment
    obs,__ = env.reset()
    uStorage.append(obs)

    ns = 0
    i = 0
    rew = 0
    act_h = []
    rew_h = []
    while not truncate and not terminate:
        # use backstepping controller
        action = model(env,obs,parameter)
        act_h.append(action)
        obs, rewards, terminate, truncate, info = env.step(action)
        uStorage.append(obs)
        rew += rewards
        rew_h.append(rewards)
        ns += 1
        # print(ns)
    u = np.array(uStorage)
    return rew, u, act_h, rew_h


def RLController(env,obs,model):
    # Assume a is the flattened concatenation of self.r and self.v
    half = obs.shape[0] // 2
    
    # Recover self.r and self.v
    r = obs[:half]
    v = obs[half:]
    
    # Compute b
    obs_sc = np.reshape(
        np.concatenate(((r - env.unwrapped.rs) / env.unwrapped.rs, (v - env.unwrapped.vs) / env.unwrapped.vs)),
        -1
    )

    action, _state = model.predict(obs_sc)
    return action

Parameters["simulation_type"] = 'outlet'
PPO_model = PPO.load('./logsPPO/rl_model_500000_steps.zip')
envBcks = gym.make("PDEControlGym-TrafficPDE1D",**Parameters)
rewBcksTen, uBcksTen, act_h, rew_h = runSingleEpisode(RLController, envBcks,PPO_model )
# Scale density
uBcksTen[:, :50] *= 1000

# Common spatial & temporal grids
spatial = np.linspace(0, X, int(round(X/dx)))
temporal = np.linspace(0, T, uBcksTen.shape[0])
meshx, mesht = np.meshgrid(spatial, temporal)

# Create side-by-side 3D plots
fig = plt.figure(figsize=(14, 6))

# Plot density r(x, t)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(meshx, mesht, uBcksTen[:, :50],
                 edgecolor="black", lw=0.2,
                 rstride=5, cstride=1,
                 alpha=1, color="white",
                 shade=False, rasterized=True,
                 antialiased=True)
ax1.set_title("Density r(x, t)")
ax1.set_xlabel("Length x")
ax1.set_ylabel("Time t")
ax1.set_zlabel("Density")

# Plot velocity v(x, t)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(meshx, mesht, uBcksTen[:, 52:],
                 edgecolor="black", lw=0.2,
                 rstride=5, cstride=100,
                 alpha=1, color="white",
                 shade=False, rasterized=True,
                 antialiased=True)
ax2.set_title("Velocity v(x, t)")
ax2.set_xlabel("Length x")
ax2.set_ylabel("Time t")
ax2.set_zlabel("Velocity")

# Set side view for both plots
ax1.view_init(elev=20, azim=10)
ax2.view_init(elev=20, azim=10)


plt.tight_layout()
plt.show()

plt.plot(act_h)
plt.show(  )

plt.plot(rew_h)
plt.show()