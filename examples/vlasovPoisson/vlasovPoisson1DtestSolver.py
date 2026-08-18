import numpy as np
import gymnasium as gym

import pde_control_gym  # noqa: F401  (registers the environments)
from pde_control_gym.src import VlasovPoissonReward
from utils import (
    twoStreamEquilibrium,
    multiplicativePerturbation,
    CancellationController,
)

# THIS SCRIPT VALIDATES THE VLASOV-POISSON SOLVER AND THE CONTROL AUTHORITY OF THE
# ENVIRONMENT. IT CHECKS THE INVARIANTS THAT NO CONTROL CAN MOVE, THE TWO BUDGET
# IDENTITIES THAT THE CONTROL DRIVES DIRECTLY, LINEAR LANDAU DAMPING, AND THE DECAY
# GUARANTEE OF THE CANCELLATION-BASED CONTROLLER OF arXiv:2509.23063 SECTION 4.

# np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in later releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz

X = 10 * np.pi
nx = 64
V = 8.0
nv = 129
dx = X / nx
dv = 2 * V / (nv - 1)


def makeEnv(T, dt, fbar_func, f0_func, domain=X, **overrides):
    fbar = fbar_func(np.linspace(0, domain, nx, endpoint=False), np.linspace(-V, V, nv))
    parameters = {
        "T": T,
        "dt": dt,
        "X": domain,
        "dx": domain / nx,
        "V": V,
        "dv": dv,
        "reward_class": VlasovPoissonReward(
            fbar=fbar, dx=domain / nx, dv=dv, nt=int(round(T / dt)) + 1
        ),
        "normalize": False,
        "reset_init_condition_func": f0_func,
        "equilibrium_func": fbar_func,
        "sensing_type": "full",
        "action_type": "field",
        "control_sample_rate": dt,
        "store_history": False,
    }
    parameters.update(overrides)
    env = gym.make("PDEControlGym-VlasovPoisson1D", **parameters)
    return env, env.unwrapped


def rollout(env, vp, controller=None, fixedH=None):
    """Runs one episode and returns the stacked per-step diagnostics."""
    obs, info = env.reset()
    history = [info]
    terminate = truncate = False
    while not (terminate or truncate):
        if fixedH is not None:
            action = fixedH
        elif controller is not None:
            action = controller.action(obs)
        else:
            action = np.zeros(vp.action_dim)
        obs, reward, terminate, truncate, info = env.step(action)
        history.append(info)
    return {
        k: np.array([d[k] for d in history]) for k in history[0] if k not in ("H", "E")
    }


twoStream = twoStreamEquilibrium(vbar=2.4)
twoStreamInit = multiplicativePerturbation(twoStream, epsilon=1e-3, mode=1, X=X)


print("=" * 72)
print("Casimirs and invariants: nothing here is controllable")
env, vp = makeEnv(20.0, 0.1, twoStream, twoStreamInit)
h = rollout(env, vp)
print(
    f"  relative mass drift over t in [0, 20]         {abs(h['mass'][-1] / h['mass'][0] - 1):.2e}"
)
print(
    f"  absolute momentum drift with H = 0            {abs(h['momentum'][-1] - h['momentum'][0]):.2e}"
)
etot = h["kinetic_energy"] + h["electric_energy"]
print(
    f"  relative total energy drift with H = 0        {abs(etot[-1] / etot[0] - 1):.2e}"
)
print("  (nonzero drift here is discretization error, not physics: the continuous flow")
print("   conserves mass exactly and conserves energy and momentum exactly when H = 0)")


print("=" * 72)
print("Budget identities: the two scalars the control drives directly")
# dP/dt = int H rho dx and d(K + W)/dt = int H j dx, with no contribution from the
# self-consistent field. Both rates are integrated with the trapezoid rule over the step
# endpoints, so the residual below is the time-quadrature error and shrinks with dt.
for dt, order in [(0.1, "linear"), (0.05, "linear"), (0.1, "cubic"), (0.05, "cubic")]:
    env, vp = makeEnv(4.0, dt, twoStream, twoStreamInit, interpolation=order)
    Hnodal = 0.05 * (1.0 + np.sin(2 * np.pi * vp.x / X))
    obs, info = env.reset()
    info = vp.diagnostics(Hnodal)
    P = [info["momentum"]]
    E = [info["kinetic_energy"] + info["electric_energy"]]
    Prate = [info["momentum_rate"]]
    Erate = [info["energy_rate"]]
    terminate = truncate = False
    while not (terminate or truncate):
        obs, reward, terminate, truncate, info = env.step(Hnodal)
        P.append(info["momentum"])
        E.append(info["kinetic_energy"] + info["electric_energy"])
        Prate.append(info["momentum_rate"])
        Erate.append(info["energy_rate"])
    dP = P[-1] - P[0]
    dE = E[-1] - E[0]
    print(
        f"  dt={dt:<6} {order:<6} dP {dP:+.5f} vs {_trapz(Prate, dx=dt):+.5f} (err {abs(dP - _trapz(Prate, dx=dt)) / abs(dP):.1e})"
        f"   dE {dE:+.5f} vs {_trapz(Erate, dx=dt):+.5f} (err {abs(dE - _trapz(Erate, dx=dt)) / abs(dE):.1e})"
    )


print("=" * 72)
print("Linear Landau damping on a stable Maxwellian, no control")
print(
    "  The box length is chosen so that the seeded harmonic lands on a wavenumber with"
)
print("  a tabulated root of the Landau dispersion relation.")


def maxwellian(x, v):
    return np.tile((np.exp(-(v**2) / 2) / np.sqrt(2 * np.pi))[None, :], (len(x), 1))


# (box length, resulting k = 2 pi / L, Landau rate of the least damped root)
for L, reference in [(4 * np.pi, -0.1533), (5 * np.pi, -0.0661)]:
    for order in ["linear", "cubic"]:
        env, vp = makeEnv(
            40.0,
            0.05,
            maxwellian,
            multiplicativePerturbation(maxwellian, 1e-4, 1, L),
            domain=L,
            interpolation=order,
        )
        h = rollout(env, vp)
        t = np.arange(len(h["electric_energy"])) * 0.05
        window = (t > 5) & (t < 30)
        rate = np.polyfit(t[window], 0.5 * np.log(h["electric_energy"][window]), 1)[0]
        print(
            f"  k = {2 * np.pi / L:.1f}, {order:<6}: measured {rate:+.4f}, linear theory {reference:+.4f}"
        )


print("=" * 72)
print("Two-stream instability grows without control")
env, vp = makeEnv(40.0, 0.1, twoStream, twoStreamInit)
h = rollout(env, vp)
t = np.arange(len(h["electric_energy"])) * 0.1
window = (t > 10) & (t < 35)
rate = np.polyfit(t[window], 0.5 * np.log(h["electric_energy"][window]), 1)[0]
print(f"  ||df||_2: {h['l2_perturbation'][0]:.3e} -> {h['l2_perturbation'][-1]:.3e}")
print(f"  fitted growth rate of |E| over t in [10, 35]: {rate:+.4f}")


print("=" * 72)
print("Cancellation control (Section 4): H = -dE + gamma * int df dv_fbar dv")
print("  With gamma = 0 the continuous theory gives d/dt ||df||^2 = 0 exactly, so the")
print("  measured decay at gamma = 0 is purely the numerical dissipation of the")
print("  semi-Lagrangian interpolation. Read the gamma > 0 rows against that baseline.")
for order in ["linear", "cubic"]:
    for gamma in [0.0, 1.0, 4.0]:
        env, vp = makeEnv(40.0, 0.1, twoStream, twoStreamInit, interpolation=order)
        h = rollout(env, vp, CancellationController(vp, gamma=gamma))
        p = h["l2_perturbation"]
        rise = np.max(np.diff(p)) / p[0]
        print(
            f"  {order:<6} gamma={gamma:<4} ||df|| ratio {p[-1] / p[0]:.4f}"
            f"   largest single-step rise {rise:+.2e} (relative)"
        )


print("=" * 72)
print("Modal truncation caps the reachable wavenumbers")
print(
    "  The instability is seeded in harmonic 3 while the actuator is restricted to the"
)
print("  first K harmonics. Below K = 3 the unstable mode is outside the range of the")
print("  input operator and the controller has no authority over it at all.")
seededMode3 = multiplicativePerturbation(twoStream, epsilon=1e-3, mode=3, X=X)
env, vp = makeEnv(40.0, 0.1, twoStream, seededMode3)
openLoop = rollout(env, vp)["l2_perturbation"]
print(f"  no control              : ||df|| ratio {openLoop[-1] / openLoop[0]:.4f}")
for K in [1, 2, 3, 15]:
    env, vp = makeEnv(
        40.0, 0.1, twoStream, seededMode3, action_type="modal", num_modes=K
    )
    h = rollout(env, vp, CancellationController(vp, gamma=4.0))
    p = h["l2_perturbation"]
    print(
        f"  num_modes={K:<3} (action_dim {vp.action_dim:>2}): ||df|| ratio {p[-1] / p[0]:.4f}"
    )


print("=" * 72)
print("Velocity-space decomposition: what the cancellation law can and cannot touch")
print(
    "  The decay estimate of Section 4 only bounds the component of df that is aligned"
)
print(
    "  with dv_fbar in velocity at each x. The orthogonal complement drops out of the"
)
print(
    "  energy estimate entirely, so it is not directly damped; whether it decays anyway"
)
print("  is left open in the paper (Section 5).")
env, vp = makeEnv(40.0, 0.1, twoStream, twoStreamInit)
obs, info = env.reset()
weight = _trapz(vp.dvfbar**2, dx=vp.dv, axis=1)
controller = CancellationController(vp, gamma=4.0)
aligned, orthogonal = [], []


def split(df):
    coeff = _trapz(df * vp.dvfbar, dx=vp.dv, axis=1) / weight
    par = coeff[:, None] * vp.dvfbar
    perp = df - par
    norm = lambda g: np.sqrt(_trapz(_trapz(g**2, dx=vp.dv, axis=1), dx=vp.dx))
    return norm(par), norm(perp)


a, o = split(obs)
aligned.append(a)
orthogonal.append(o)
terminate = truncate = False
while not (terminate or truncate):
    obs, reward, terminate, truncate, info = env.step(controller.action(obs))
    a, o = split(obs)
    aligned.append(a)
    orthogonal.append(o)
total0 = np.hypot(aligned[0], orthogonal[0])
totalT = np.hypot(aligned[-1], orthogonal[-1])
print(
    f"  aligned    ||df_par||  {aligned[0]:.3e} -> {aligned[-1]:.3e}"
    f"   ({aligned[0] / total0:6.2%} -> {aligned[-1] / totalT:6.2%} of the total)"
)
print(
    f"  orthogonal ||df_perp|| {orthogonal[0]:.3e} -> {orthogonal[-1]:.3e}"
    f"   ({orthogonal[0] / total0:6.2%} -> {orthogonal[-1] / totalT:6.2%} of the total)"
)
print(
    "  The multiplicative initial condition f0 = (1 + eps cos kx) fbar of the paper is"
)
print(
    "  exactly orthogonal to dv_fbar at t = 0, because int fbar dv_fbar dv = 0. The whole"
)
print("  perturbation therefore starts in the subspace the decay estimate says nothing")
print("  about, and only leaks into the aligned part through the dynamics.")


print("=" * 72)
print(
    "Actuator rank: at any instant H reaches at most nx of the nx*nv state directions"
)
env, vp = makeEnv(1.0, 0.1, twoStream, twoStreamInit)
env.reset()
# The input operator of the linearization is B: H(x) -> -H(x) dv_fbar(x, v), which is a
# rank-nx map into the nx*nv dimensional phase space, and its range collapses further
# wherever dv_fbar vanishes.
B = -vp.dvfbar
print(f"  phase space dimension nx * nv               {vp.nx * vp.nv}")
print(f"  instantaneous actuator rank (<= nx)         {vp.nx}")
print(f"  actuated fraction of the state              {vp.nx / (vp.nx * vp.nv):.4%}")
weight = np.sqrt(_trapz(B**2, dx=vp.dv, axis=1))
print(
    f"  min / max |dv_fbar|_v over x                {weight.min():.3e} / {weight.max():.3e}"
)
