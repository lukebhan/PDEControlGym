"""Rack flow/heat model (Han et al. 2021, Eqs. 11-12) and power assignment.

Pure functions. assign_powers accepts any sequence of objects exposing
`.id` and `.u_height` (the future `layout.Rack` dataclass satisfies this, and
so does a plain namedtuple used in tests).
"""

from .units import RHO, CP, RACK_FLOW_M3H_PER_KW, m3h_to_m3s


def rack_flow_m3s(power_kW):
    """Rack air flow [m^3/s] from IT power, Han Eq. 11 (212 m^3/h per kW)."""
    return m3h_to_m3s(power_kW * RACK_FLOW_M3H_PER_KW)


def rack_delta_T(power_W, Q_m3s):
    """Rack exhaust-inlet temperature rise [K], Han Eq. 12: P / (rho cp Q)."""
    return power_W / (RHO * CP * Q_m3s)


def assign_powers(racks, total_kW, mode="u_proportional"):
    """Distribute total_kW of IT power over `racks` -> {rack.id: power_kW}.

    Racks with a truthy `.empty` attribute (Han's G11/G13: physically present,
    unpowered cabinets) are excluded from the distribution and assigned 0 kW;
    they are also excluded from the denominator (U total or count) so the
    remaining racks still absorb all of total_kW.
    """
    powered = [r for r in racks if not getattr(r, "empty", False)]
    empty = [r for r in racks if getattr(r, "empty", False)]
    if mode == "u_proportional":
        total_u = sum(r.u_height for r in powered)
        result = {r.id: total_kW * r.u_height / total_u for r in powered}
    elif mode == "uniform":
        n = len(powered)
        result = {r.id: total_kW / n for r in powered}
    else:
        raise ValueError(f"unknown power assignment mode: {mode!r}")
    result.update({r.id: 0.0 for r in empty})
    return result
