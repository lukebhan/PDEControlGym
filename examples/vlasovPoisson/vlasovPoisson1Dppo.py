import numpy as np
import gymnasium as gym

import pde_control_gym  # noqa: F401  (registers the environments)
from pde_control_gym.src import VlasovPoissonReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from utils import twoStreamEquilibrium, multiplicativePerturbation

# THIS EXAMPLE TRAINS A PPO AGENT TO SUPPRESS THE TWO-STREAM INSTABILITY.
#
# The setup deliberately does not hand the agent the full phase space. It observes only
# the charge density perturbation, which is what a real diagnostic measures, and it acts
# through the truncated Fourier basis of arXiv:2509.23063 eq. (3.1) rather than through
# every grid point. Both restrictions are genuine limits on the problem rather than
# conveniences: see the environment documentation for which parts of the state remain
# reachable and observable under them.

X = 10 * np.pi
nx = 100
V = 8.0
nv = 129
T = 30.0
dt = 0.2

fbar_func = twoStreamEquilibrium(vbar=2.4)
fbar = fbar_func(np.linspace(0, X, nx, endpoint=False), np.linspace(-V, V, nv))


def getInitialCondition(x, v):
    """Randomizes the seeded harmonic and its amplitude at every reset."""
    epsilon = np.random.uniform(5e-4, 2e-3)
    mode = np.random.randint(1, 4)
    return multiplicativePerturbation(fbar_func, epsilon=epsilon, mode=mode, X=X)(x, v)


VP1DParameters = {
    "T": T,
    "dt": dt,
    "X": X,
    "dx": X / nx,
    "V": V,
    "dv": 2 * V / (nv - 1),
    "reward_class": VlasovPoissonReward(
        fbar=fbar,
        dx=X / nx,
        dv=2 * V / (nv - 1),
        gamma=1e-3,
        nt=int(round(T / dt)) + 1,
    ),
    "normalize": True,
    "reset_init_condition_func": getInitialCondition,
    "equilibrium_func": fbar_func,
    "sensing_type": "density",
    "action_type": "modal",
    "num_modes": 15,
    # The actuator range has to be matched to the size of the perturbation being fought:
    # the self-consistent field of a 1e-3 perturbation is itself only O(1e-3), so an
    # actuator that can swing by O(1) mostly gives the agent room to make things worse.
    "max_control_value": 0.05,
    "limit_pde_state_size": True,
    "max_state_value": 10.0,
    "control_sample_rate": dt,
    "store_history": False,
}

env = gym.make("PDEControlGym-VlasovPoisson1D", **VP1DParameters)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logsPPO",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb/")
model.learn(total_timesteps=2e5, callback=checkpoint_callback)
model.save("vlasovPoisson1Dppo")
