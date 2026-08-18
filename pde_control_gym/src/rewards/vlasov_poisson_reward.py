from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional, Union

# np.trapz was renamed to np.trapezoid in numpy 2.0 and removed in later releases.
_trapz = getattr(np, "trapezoid", None) or np.trapz


class VlasovPoissonReward(BaseReward):
    r"""
    VlasovPoissonReward

    This reward implements the running cost of the Vlasov-Poisson control objective,

    .. math::

        r = -\frac{1}{2} \|\delta f(\cdot, \cdot, t)\|_{x,v}^2
            - \frac{\gamma}{2} \|H(\cdot, t)\|_x^2,

    where :math:`\delta f = f - \bar f` is the deviation from the target equilibrium and
    :math:`H` is the applied external field. This is the negated integrand of the cost in
    Lu, Wang and Calder (arXiv:2509.23063, eq. 2.2). Note that the paper reports its best
    numerical results with :math:`\gamma = 0`, which is the default here.

    :param fbar: The target equilibrium :math:`\bar f` on the ``(nx, nv)`` phase space
        grid. No default: an error is thrown if not specified.
    :param dx: The spatial grid spacing, used for the quadrature.
    :param dv: The velocity grid spacing, used for the quadrature.
    :param gamma: The weight on the control effort term. Default is :math:`0`.
    :param truncate_penalty: Penalty applied for each remaining timestep when the episode
        ends early. Default is :math:`-1e2`, which dominates any achievable running cost
        and so makes blow-up strictly worse than surviving.
    :param terminate_reward: Reward for surviving the full episode. Default is :math:`0`,
        matching the paper's objective, which has no terminal bonus. Be careful raising
        this: the running cost here is :math:`O(\|\delta f\|^2)` and is often
        :math:`10^{-6}` or smaller, so a survival bonus of any appreciable size makes the
        return almost independent of how well the instability is actually suppressed.
    :param nt: The number of maximum timesteps, needed to scale ``truncate_penalty``.
    """

    def __init__(
        self,
        fbar: np.ndarray = None,
        dx: float = None,
        dv: float = None,
        gamma: float = 0.0,
        truncate_penalty: float = -1e2,
        terminate_reward: float = 0.0,
        nt: int = None,
    ):
        if fbar is None or dx is None or dv is None:
            raise Exception(
                "The equilibrium fbar and the grid spacings dx and dv must be specified in the VlasovPoissonReward class."
            )
        self.fbar = fbar
        self.dx = dx
        self.dv = dv
        self.gamma = gamma
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward
        self.nt = nt

    def reward(
        self,
        uVec: np.ndarray = None,
        time_index: int = None,
        terminate: Optional[bool] = None,
        truncate: Optional[bool] = None,
        action: Optional[Union[float, np.ndarray]] = None,
    ):
        r"""
        reward

        :param uVec: (required) The distribution history :math:`f`, of shape
            ``(nt, nx, nv)``.
        :param time_index: (required) The time at which to compute the reward, given as an
            index into ``uVec``.
        :param terminate: States whether the episode is the terminal episode.
        :param truncate: States whether the episode is truncated, or ending early.
        :param action: The external field :math:`H` on the spatial grid, as expanded by
            the environment. Only used when ``gamma`` is nonzero.
        """
        if uVec is None:
            raise Exception(
                "Class VlasovPoissonReward attempted to call reward function and recieved a None vector to compute on"
            )
        if time_index is None:
            raise Exception(
                "Class VlasovPoissonReward attempted to call reward function and recieved a None time_index parameter to identify the reward step"
            )

        if truncate:
            if self.nt is None:
                raise Exception(
                    "The number of simulation steps nt must be specified in the VlasovPoissonReward class to penalize truncation."
                )
            return self.truncate_penalty * (self.nt - time_index)

        df = uVec[time_index] - self.fbar
        state_cost = 0.5 * _trapz(
            _trapz(df**2, dx=self.dv, axis=-1), dx=self.dx, axis=-1
        )
        control_cost = 0.0
        if self.gamma != 0.0 and action is not None:
            control_cost = (
                0.5 * self.gamma * _trapz(np.asarray(action) ** 2, dx=self.dx, axis=-1)
            )
        reward = -state_cost - control_cost
        if terminate:
            reward += self.terminate_reward
        return reward
