import numpy as np
import gymnasium as gym

import pde_control_gym  # noqa: F401  (registers the environments)
from pde_control_gym.src import VlasovPoissonReward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from utils import (
    twoStreamEquilibrium,
    CancellationController,
    ScaleFreeVlasov,
    runEpisode,
)

# THIS EXAMPLE TRAINS A PPO AGENT TO SUPPRESS THE TWO-STREAM INSTABILITY, AND SCORES IT
# AGAINST DOING NOTHING AND AGAINST THE TRAINING-FREE FEEDBACK LAW OF arXiv:2509.23063
# SECTION 4. See VlasovPoisson1DExample.ipynb for the same setup worked through with
# plots and an explanation of every choice made here.

X, nx = 10 * np.pi, 100
V, nv = 8.0, 129
T, dt = 30.0, 0.2

# An action of +-1 corresponds to this many multiples of the observed field magnitude.
GAIN = 3.0
N_ENVS = 4
TOTAL_TIMESTEPS = 200_000

fbarFunc = twoStreamEquilibrium(vbar=2.4)
fbar = fbarFunc(np.linspace(0, X, nx, endpoint=False), np.linspace(-V, V, nv))


class InitialConditionSampler:
    r"""
    Randomizes the seeded harmonic, amplitude, and phase at every reset.

    Only harmonics 1 and 2 are seeded, because for this equilibrium those are the only
    ones that grow: harmonic 3 and above are Landau damped and need no control. Carrying
    an explicit Generator rather than the global numpy state keeps evaluation episodes
    reproducible across policies.
    """

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def __call__(self, x, v):
        epsilon = self.rng.uniform(5e-4, 2e-3)
        mode = self.rng.integers(1, 3)
        phase = self.rng.uniform(0, 2 * np.pi)
        modulation = 1 + epsilon * np.cos(2 * np.pi * mode * x / X + phase)
        return modulation[:, None] * fbarFunc(x, v)


def makeEnv(initialConditionFunc, sensing="field"):
    """Builds the environment. normalize=False so that ScaleFreeVlasov owns the scaling."""
    parameters = {
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
            gamma=0.0,
            nt=int(round(T / dt)) + 1,
        ),
        "normalize": False,
        "reset_init_condition_func": initialConditionFunc,
        "equilibrium_func": fbarFunc,
        "sensing_type": sensing,
        "action_type": "modal",
        "num_modes": 2,
        # A uniform H only accelerates the plasma as a whole, so it cannot suppress a
        # spatial instability and is dropped from the action space.
        "include_uniform_mode": False,
        "max_control_value": 1.0,
        "limit_pde_state_size": True,
        "max_state_value": 10.0,
        "control_sample_rate": dt,
        "store_history": False,
    }
    return gym.make("PDEControlGym-VlasovPoisson1D", **parameters)


def evaluate(buildPolicy, sensing="field", wrap=True, n=8, seed=9000):
    """Scores a policy on a fixed seed sequence, so every policy sees the same episodes."""
    papers, ratios = [], []
    for i in range(n):
        env = makeEnv(InitialConditionSampler(seed + i), sensing=sensing)
        unwrapped = env.unwrapped
        if wrap:
            env = ScaleFreeVlasov(env, gain=GAIN)
        paper, trace = runEpisode(env, buildPolicy(unwrapped))
        papers.append(paper)
        ratios.append(trace["l2_perturbation"][-1] / trace["l2_perturbation"][0])
    # Geometric mean, because a growth factor is multiplicative across episodes with a
    # spread of growth rates.
    return np.mean(papers), np.exp(np.mean(np.log(ratios)))


if __name__ == "__main__":
    scores = {
        "no control": evaluate(
            lambda vp: (lambda obs: np.zeros(vp.action_dim)), sensing="full", wrap=False
        ),
        "cancellation gamma=4": evaluate(
            lambda vp: CancellationController(vp, gamma=4.0).action,
            sensing="full",
            wrap=False,
        ),
    }

    vecEnv = DummyVecEnv(
        [
            (
                lambda rank=i: ScaleFreeVlasov(
                    makeEnv(InitialConditionSampler(100 + rank)), gain=GAIN
                )
            )
            for i in range(N_ENVS)
        ]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path="./logsPPO", name_prefix="rl_model"
    )

    model = PPO(
        "MlpPolicy",
        vecEnv,
        verbose=1,
        seed=0,
        n_steps=256,
        batch_size=256,
        learning_rate=1e-3,
        ent_coef=0.0,
        tensorboard_log="./tb/",
        # A linear policy, because Section 2.1 of the paper shows the optimal control for
        # the linearized system is a linear functional of the perturbation. log_std_init
        # keeps the initial exploration below the useful control amplitude, without which
        # exploration destabilizes the plasma faster than the learning signal accumulates.
        policy_kwargs=dict(net_arch=[], log_std_init=-1.5),
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save("vlasovPoisson1Dppo")

    scores["PPO"] = evaluate(
        lambda vp: (lambda obs: model.predict(obs, deterministic=True)[0])
    )

    print(f"\n{'policy':<24} {'paper reward':>14} {'geo-mean growth':>17}")
    for name, (paper, ratio) in scores.items():
        print(f"{name:<24} {paper:>14.4e} {ratio:>17.4g}")
