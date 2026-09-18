from pde_control_gym.src.rewards.base_reward import BaseReward
import numpy as np
from typing import Optional


class DataCenterReward(BaseReward):
    r"""
    DataCenterReward

    Default reward for :class:`DataCenter3D`. Encodes the core data-center
    thermal-management trade-off: **minimize cooling energy subject to
    rack-inlet thermal compliance**. Called by the environment with the
    per-rack inlet temperatures, the current action ``[supply_flow, supply_T]``
    and the current IT load; swap in any :class:`BaseReward` subclass with the
    same call signature to change the objective.

    The energy term is an intentionally simple *proxy*, not a validated plant
    model (documented in the env README):

    - fan power follows the affinity law, :math:`\propto (Q_{sup}/Q_{nom})^3`;
    - chiller work :math:`\propto P_{IT}/\mathrm{COP}(T_{sup})`, with a linear
      COP that improves as the supply (chilled-air) temperature rises.

    :param q_nom_m3s: Nominal supply flow [m^3/s] used to normalize fan power. Required.
    :param t_rec: ASHRAE recommended max rack-inlet temperature [C] (soft limit). Default 27.
    :param t_allow: ASHRAE allowable max rack-inlet temperature [C] (hard limit). Default 35.
    :param p_it_nom_kW: Nominal total IT power [kW] used to normalize chiller work. Required.
    :param w_energy: Weight on the (normalized) energy proxy. Default 1.0.
    :param w_thermal: Weight on the quadratic hot-spot penalty above ``t_rec``. Default 1.0.
    :param w_crit: Flat penalty applied whenever any rack inlet exceeds ``t_allow``. Default 100.
    :param cop0: COP at the reference supply temperature ``t_cop_ref``. Default 4.0.
    :param cop_alpha: Fractional COP gain per degree of supply temperature. Default 0.03.
    :param t_cop_ref: Supply temperature [C] at which COP == ``cop0``. Default 18.
    :param truncate_penalty: Per-remaining-step penalty if the episode is truncated. Default -1e2.
    :param terminate_reward: Bonus for completing the full episode. Default 0.0.
    """

    def __init__(self, q_nom_m3s: float = None, t_rec: float = 27.0, t_allow: float = 35.0,
                 p_it_nom_kW: float = None, w_energy: float = 1.0, w_thermal: float = 1.0,
                 w_crit: float = 100.0, cop0: float = 4.0, cop_alpha: float = 0.03,
                 t_cop_ref: float = 18.0, truncate_penalty: float = -1e2,
                 terminate_reward: float = 0.0, nt: int = None):
        if q_nom_m3s is None or p_it_nom_kW is None:
            raise Exception(
                "DataCenterReward requires q_nom_m3s and p_it_nom_kW to normalize the "
                "energy proxy. DataCenter3D passes these automatically when it builds "
                "its default reward.")
        self.q_nom_m3s = q_nom_m3s
        self.t_rec = t_rec
        self.t_allow = t_allow
        self.p_it_nom_kW = p_it_nom_kW
        self.w_energy = w_energy
        self.w_thermal = w_thermal
        self.w_crit = w_crit
        self.cop0 = cop0
        self.cop_alpha = cop_alpha
        self.t_cop_ref = t_cop_ref
        self.truncate_penalty = truncate_penalty
        self.terminate_reward = terminate_reward
        self.nt = nt

    def energy_proxy(self, Q_sup_m3s, T_sup_C, p_it_kW):
        """Normalized cooling-energy proxy (fan affinity law + COP-scaled chiller work)."""
        e_fan = (Q_sup_m3s / self.q_nom_m3s) ** 3
        cop = max(0.1, self.cop0 * (1.0 + self.cop_alpha * (T_sup_C - self.t_cop_ref)))
        e_chiller = (p_it_kW / self.p_it_nom_kW) / cop
        return e_fan + e_chiller

    def reward(self, rack_inlet_temps: np.ndarray = None, action: np.ndarray = None,
               p_it_kW: float = None, terminate: Optional[bool] = None,
               truncate: Optional[bool] = None, time_index: Optional[int] = None):
        r"""
        :param rack_inlet_temps: (required) per-rack inlet temperatures [C].
        :param action: (required) the applied ``[supply_flow_m3s, supply_T_C]``.
        :param p_it_kW: (required) current total IT power [kW].
        :param terminate: whether this is the terminal step.
        :param truncate: whether the episode is ending early.
        :param time_index: current control-step index (for the truncation penalty).
        """
        if truncate:
            remaining = 0 if (self.nt is None or time_index is None) else max(0, self.nt - time_index)
            return self.truncate_penalty * (remaining if remaining else 1)

        Q_sup, T_sup = float(action[0]), float(action[1])
        T = np.asarray(rack_inlet_temps, dtype=float)

        energy = self.w_energy * self.energy_proxy(Q_sup, T_sup, p_it_kW)
        hot_spot = self.w_thermal * float(np.sum(np.clip(T - self.t_rec, 0.0, None) ** 2))
        critical = self.w_crit if float(T.max()) > self.t_allow else 0.0

        r = -(energy + hot_spot + critical)
        if terminate:
            r += self.terminate_reward
        return r
