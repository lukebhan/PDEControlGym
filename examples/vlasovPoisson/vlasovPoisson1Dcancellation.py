import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import pde_control_gym  # noqa: F401  (registers the environments)
from pde_control_gym.src import VlasovPoissonReward
from utils import (
    twoStreamEquilibrium,
    multiplicativePerturbation,
    CancellationController,
)

# THIS EXAMPLE REPRODUCES THE TWO-STREAM EXPERIMENT OF arXiv:2509.23063 USING THE
# TRAINING-FREE CANCELLATION-BASED CONTROLLER OF SECTION 4, AND COMPARES IT AGAINST THE
# UNCONTROLLED SYSTEM.

# Domain and discretization. The paper uses Nx = 100, Nv = 200 on
# [0, 10 pi] x [-8, 8] with dt = 0.2; a coarser velocity grid is used here so the example
# runs in well under a minute.
X = 10 * np.pi
nx = 100
V = 8.0
nv = 129
T = 70.0
dt = 0.2

fbar_func = twoStreamEquilibrium(vbar=2.4)
f0_func = multiplicativePerturbation(fbar_func, epsilon=1e-3, mode=1, X=X)

x_grid = np.linspace(0, X, nx, endpoint=False)
v_grid = np.linspace(-V, V, nv)
fbar = fbar_func(x_grid, v_grid)

VP1DParameters = {
    "T": T,
    "dt": dt,
    "X": X,
    "dx": X / nx,
    "V": V,
    "dv": 2 * V / (nv - 1),
    "reward_class": VlasovPoissonReward(
        fbar=fbar, dx=X / nx, dv=2 * V / (nv - 1), gamma=0.0, nt=int(round(T / dt)) + 1
    ),
    "normalize": False,
    "reset_init_condition_func": f0_func,
    "equilibrium_func": fbar_func,
    "sensing_type": "full",
    "action_type": "field",
    "poisson_solver": "periodic",
    "max_control_value": 1.0,
    "control_sample_rate": dt,
}

env = gym.make("PDEControlGym-VlasovPoisson1D", **VP1DParameters)
vp = env.unwrapped


def rollout(controller):
    """Runs one episode, returning the diagnostics traces and the final distribution."""
    obs, info = env.reset()
    perturbation = [info["l2_perturbation"]]
    electric_energy = [info["electric_energy"]]
    momentum = [info["momentum"]]
    times = [0.0]
    terminate = truncate = False
    while not terminate and not truncate:
        action = (
            np.zeros(vp.action_dim) if controller is None else controller.action(obs)
        )
        obs, reward, terminate, truncate, info = env.step(action)
        perturbation.append(info["l2_perturbation"])
        electric_energy.append(info["electric_energy"])
        momentum.append(info["momentum"])
        times.append(vp.time_index * dt)
    return (
        np.array(times),
        np.array(perturbation),
        np.array(electric_energy),
        np.array(momentum),
        vp.f_current.copy(),
    )


print("Running the uncontrolled two-stream instability...")
t_ol, pert_ol, energy_ol, _, f_ol = rollout(None)

closed = {}
for gamma in [1.0, 4.0, 16.0]:
    print(f"Running the cancellation-based controller with gamma = {gamma:g}...")
    closed[gamma] = rollout(CancellationController(vp, gamma=gamma))

print(f"open loop            ||df(T)||_2 = {pert_ol[-1]:.3e}  (t=0: {pert_ol[0]:.3e})")
for gamma, r in closed.items():
    print(
        f"cancellation gamma={gamma:<5g}||df(T)||_2 = {r[1][-1]:.3e}  (t=0: {r[1][0]:.3e})"
    )

t_cl, pert_cl, energy_cl, _, f_cl = closed[16.0]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

axes[0, 0].semilogy(t_ol, pert_ol, "r", label="no control")
for gamma, r in closed.items():
    axes[0, 0].semilogy(r[0], r[1], label=rf"cancellation, $\gamma={gamma:g}$")
axes[0, 0].set_xlabel("$t$")
axes[0, 0].set_ylabel(r"$\|\delta f(t)\|_{2}$")
axes[0, 0].set_title(r"$L^2$ state perturbation")
axes[0, 0].legend()

axes[0, 1].semilogy(t_ol, energy_ol, "r", label="no control")
for gamma, r in closed.items():
    axes[0, 1].semilogy(r[0], r[2], label=rf"cancellation, $\gamma={gamma:g}$")
axes[0, 1].set_xlabel("$t$")
axes[0, 1].set_ylabel(r"$\frac{1}{2}\int E^2 dx$")
axes[0, 1].set_title("electric energy")
axes[0, 1].legend()

extent = [0, X, -V, V]
axes[1, 0].imshow(
    (f_ol - fbar).T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r"
)
axes[1, 0].set_title(rf"$\delta f(x, v, {T:.0f})$, no control")
axes[1, 0].set_xlabel("$x$")
axes[1, 0].set_ylabel("$v$")

axes[1, 1].imshow(
    (f_cl - fbar).T, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r"
)
axes[1, 1].set_title(rf"$\delta f(x, v, {T:.0f})$, cancellation control $\gamma=16$")
axes[1, 1].set_xlabel("$x$")
axes[1, 1].set_ylabel("$v$")

plt.tight_layout()
plt.savefig("vlasovPoisson1Dcancellation.png", dpi=150)
print("wrote vlasovPoisson1Dcancellation.png")
