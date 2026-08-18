import numpy as np
import gymnasium as gym
from gymnasium import spaces

# np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in later releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def twoStreamEquilibrium(vbar=2.4):
    r"""
    The two stream equilibrium of Section 3.2 of arXiv:2509.23063,

    .. math::

        \bar f(v) = \frac{1}{2\sqrt{2\pi}} e^{-(v - \bar v)^2 / 2}
                  + \frac{1}{2\sqrt{2\pi}} e^{-(v + \bar v)^2 / 2}.

    This is a linearly unstable equilibrium: the counter streaming beams pump the
    electrostatic field and the perturbation grows exponentially without control.
    """

    def fbar(x, v):
        profile = 0.5 / np.sqrt(2 * np.pi) * np.exp(
            -((v - vbar) ** 2) / 2
        ) + 0.5 / np.sqrt(2 * np.pi) * np.exp(-((v + vbar) ** 2) / 2)
        return np.tile(profile[None, :], (len(x), 1))

    return fbar


def bumpOnTailEquilibrium(w1=0.9, w2=0.1, v1=-2.0, v2=3.5, vt=0.25):
    r"""
    The bump-on-tail equilibrium of Section 3.3 of arXiv:2509.23063, a cold bulk plus a
    thin fast beam. The positive slope of :math:`\bar f` between the bulk and the beam is
    what drives the instability.
    """

    def fbar(x, v):
        profile = w1 / np.sqrt(2 * np.pi) * np.exp(-((v - v1) ** 2) / 2) + w2 / (
            np.sqrt(2 * np.pi) * vt
        ) * np.exp(-((v - v2) ** 2) / (2 * vt))
        return np.tile(profile[None, :], (len(x), 1))

    return fbar


def multiplicativePerturbation(fbar_func, epsilon=1e-3, mode=1, X=10 * np.pi):
    r"""
    The initial condition :math:`f_0 = (1 + \epsilon \cos(2\pi m x / X)) \bar f(v)` used
    throughout the paper. This seeds a single spatial harmonic of the instability.
    """

    def f0(x, v):
        fbar = fbar_func(x, v)
        return (1 + epsilon * np.cos(2 * np.pi * mode * x / X))[:, None] * fbar

    return f0


def beamPerturbation(
    fbar_func, epsilon=3e-3, w2=0.1, v2=3.5, vt=0.25, mode=1, X=10 * np.pi
):
    r"""
    The bump-on-tail initial condition of Section 3.3, which perturbs only the fast beam,

    .. math::

        f_0 = \bar f(v) + \frac{\epsilon \omega_2}{\sqrt{2\pi} v_t}
              e^{-(v - \bar v_2)^2 / 2 v_t} \sin(2\pi m x / X).
    """

    def f0(x, v):
        beam = (
            epsilon
            * w2
            / (np.sqrt(2 * np.pi) * vt)
            * np.exp(-((v - v2) ** 2) / (2 * vt))
        )
        return (
            fbar_func(x, v) + np.sin(2 * np.pi * mode * x / X)[:, None] * beam[None, :]
        )

    return f0


class CancellationController:
    r"""
    The training-free "quasi-optimal universal" feedback law of Section 4 of
    arXiv:2509.23063,

    .. math::

        H[\delta f](x) = -\delta E[\delta f](x)
                       + \gamma \int \delta f(x, v) \partial_v \bar f(v) dv.

    The first term cancels the destabilizing self-consistent field perturbation, which
    removes the only mechanism by which the linear instability grows. The second term is a
    dissipation term chosen so that

    .. math::

        \frac{1}{2}\frac{d}{dt}\|\delta f\|_2^2
        = -\gamma \int \left( \int \delta f \partial_v \bar f dv \right)^2 dx \le 0,

    i.e. the perturbation norm is nonincreasing for every :math:`\gamma > 0`. Note this is
    only a *monotonicity* guarantee: the part of :math:`\delta f` orthogonal to
    :math:`\partial_v \bar f` in velocity contributes nothing to the right hand side and
    is therefore not directly damped by the control.

    This controller requires the ``"full"`` sensing mode because it needs the velocity
    structure of :math:`\delta f`, not just its density.

    :param env: The :class:`VlasovPoisson1D` environment being controlled.
    :param gamma: The dissipation gain :math:`\gamma > 0`.
    """

    def __init__(self, env, gamma=1.0):
        self.env = env
        self.gamma = gamma
        # Least squares projection of a nodal field onto the environment's action basis.
        # For "field" actions this is the identity; for "modal" actions it truncates the
        # control to the actuated harmonics, which is exactly the authority limit the
        # agent has.
        self.projector = np.linalg.pinv(env.control_basis)
        self.scale = env.max_control_value if env.normalize_actions else 1.0

    def action(self, df):
        r"""
        :param df: The observed perturbation :math:`\delta f` of shape ``(nx, nv)``.
        """
        env = self.env
        dE = env.solve_poisson(env.density(df))
        dissipation = self.gamma * _trapz(df * env.dvfbar, dx=env.dv, axis=1)
        H = -dE + dissipation
        return self.projector @ H / self.scale


class ScaleFreeVlasov(gym.Wrapper):
    r"""
    Conditions the Vlasov-Poisson environment for reinforcement learning.

    The raw problem is badly scaled for a fixed-precision policy network. An unstable
    perturbation grows by two to three orders of magnitude over a single episode and the
    running cost :math:`-\frac{1}{2}\|\delta f\|^2` therefore ranges over five or six, so a
    network that responds usefully at reset saturates once the instability develops, and
    the value function spends its capacity on the tail. This wrapper makes three changes.

    First, the observation is divided by its own largest magnitude and the discarded
    amplitude is appended as :math:`\log_{10}` of the scale, so the policy sees an
    :math:`O(1)` shape plus a slowly varying level.

    Second, the action is interpreted as a multiple of that same scale rather than an
    absolute field, so the policy also only has to produce a shape. This matches the
    structure of the problem rather than papering over it: Section 2.1 of
    arXiv:2509.23063 shows via the Pontryagin maximum principle that the optimal control
    for the linearized system is a *linear* functional of :math:`\delta f`, hence
    positively homogeneous, which is exactly what the rescaling makes representable at
    every amplitude by a single set of weights. It is also why a linear policy
    (``net_arch=[]``) tends to train faster here than an MLP.

    Third, the reward is replaced by the per-step log growth factor
    :math:`-\log(\|\delta f(t)\|_2 / \|\delta f(t - \Delta t)\|_2)`, whose episode sum is
    the negative log of the overall growth. This is bounded, dense, and roughly linear in
    the exponential growth rate, which is the quantity actually being controlled. The
    original quadratic cost is still computed and returned under the ``paper_reward`` key
    of the ``info`` dict, so the objective of the paper remains the reported metric even
    though it is not the training signal.

    :param env: A :class:`VlasovPoisson1D` environment, ideally with
        ``sensing_type="field"`` so that the observation is the very field the control has
        to cancel, and with ``normalize=False`` so that this wrapper owns the action
        scaling outright.
    :param gain: How many multiples of the observed field magnitude an action of
        :math:`\pm 1` corresponds to. Sets the control authority available to the agent.
    """

    def __init__(self, env, gain=3.0):
        super().__init__(env)
        self.gain = gain
        unwrapped = env.unwrapped
        if unwrapped.normalize_actions:
            raise Exception(
                "ScaleFreeVlasov owns the action scaling, so build the environment with normalize=False."
            )
        shape = unwrapped.observation_space.shape
        if len(shape) != 1:
            raise Exception(
                "ScaleFreeVlasov expects a flat observation, so use sensing_type 'density' or 'field'."
            )
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(shape[0] + 1,), dtype=np.float64
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(unwrapped.action_dim,), dtype=np.float32
        )

    def observation(self, raw):
        r"""
        Splits ``raw`` into a unit-magnitude shape and an appended :math:`\log_{10}` level.
        """
        self.scale = max(np.abs(raw).max(), 1e-14)
        return np.concatenate([raw / self.scale, [np.log10(self.scale)]])

    def reset(self, **kwargs):
        raw, info = self.env.reset(**kwargs)
        self.previous = max(info["l2_perturbation"], 1e-30)
        return self.observation(raw), info

    def step(self, action):
        physical = np.asarray(action, dtype=np.float64).ravel() * self.scale * self.gain
        raw, reward, terminate, truncate, info = self.env.step(physical)
        current = max(info["l2_perturbation"], 1e-30)
        shaped = -np.log(current / self.previous)
        self.previous = current
        info["paper_reward"] = reward
        return self.observation(raw), shaped, terminate, truncate, info


def runEpisode(env, policy):
    r"""
    Runs one episode and returns the accumulated reward together with per-step traces.

    :param env: The environment, wrapped or not.
    :param policy: A callable mapping the observation to an action.
    """
    obs, info = env.reset()
    trace = {k: [info[k]] for k in ("l2_perturbation", "electric_energy", "momentum")}
    fields, paper = [], 0.0
    terminate = truncate = False
    while not (terminate or truncate):
        obs, reward, terminate, truncate, info = env.step(policy(obs))
        paper += info.get("paper_reward", reward)
        for k in trace:
            trace[k].append(info[k])
        fields.append(info["H"])
    trace = {k: np.array(v) for k, v in trace.items()}
    trace["H"] = np.array(fields)
    return paper, trace
