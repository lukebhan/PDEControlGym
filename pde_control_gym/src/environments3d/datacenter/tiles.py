"""Perforated-tile flow model (Han et al. 2021, Eqs. 8-10). Pure functions."""


def loss_coefficient(beta):
    """Tile loss coefficient f(beta), Han Eq. 9. beta = open-area ratio (0,1]."""
    return (1.0 / beta**2) * (
        1.0 + 0.5 * (1.0 - beta) ** 0.75 + 1.414 * (1.0 - beta) ** 0.375
    )


def tile_velocity_from_pressure(p_kin, beta):
    """Approach velocity V [m/s] through a tile given kinematic pressure drop
    p_kin [m^2/s^2] across it, Han Eq. 8 solved for V."""
    f = loss_coefficient(beta)
    return (2.0 * max(p_kin, 0.0) / f) ** 0.5


def pressure_from_tile_velocity(V, beta):
    """Inverse of tile_velocity_from_pressure: kinematic pressure drop for a
    given approach velocity V [m/s], Han Eq. 8."""
    f = loss_coefficient(beta)
    return 0.5 * f * V**2


def body_force(Q_m3s, area_m2, h_m, beta):
    """Kinematic body force [m/s^2] applied in the cell of height h_m directly
    above a tile carrying flow Q_m3s through area area_m2, Han Eq. 10:
    F = Q^2 / (A^2 h) * (1/beta - 1)."""
    return (Q_m3s**2) / (area_m2**2 * h_m) * (1.0 / beta - 1.0)
