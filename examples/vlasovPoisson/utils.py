import numpy as np

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
