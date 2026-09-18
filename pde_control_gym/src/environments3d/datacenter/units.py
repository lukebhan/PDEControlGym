"""Physical constants and unit conversions for the data-center model (pure functions).

Same fluid constants as the room cases (docs/case_parameters.md), plus the raised-floor
tile size and the rack-flow rule of thumb from Han et al. (2021) Eq. 11.
"""

FT = 0.3048  # m per foot
IN = 0.0254  # m per inch
TILE_M = 0.6096  # m, 2 ft raised-floor tile
TILE_AREA_M2 = TILE_M**2

RHO = 1.2  # kg/m^3
CP = 1006.0  # J/(kg K)
NU = 1.5e-5  # m^2/s
ALPHA = 2.1e-5  # m^2/s
BETA_THERMAL = 1.0 / 295.15  # 1/K, Boussinesq thermal expansion

RACK_FLOW_M3H_PER_KW = 212.0  # Han Eq. 11


def ft_to_m(x_ft):
    return x_ft * FT


def m3h_to_m3s(x_m3h):
    return x_m3h / 3600.0


def cfm_to_m3s(x_cfm):
    return x_cfm * (FT**3) / 60.0
