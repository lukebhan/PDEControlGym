"""Configuration dataclasses and boundary specifications for the FFD solver.

These describe *what* to simulate (domain boundaries, internal solids/racks and
the numerical / physical options); the numerics that consume them live in
`solver.py`. See `Solver` for how each field is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Domain-face names, ordered (low, high) per axis. Used as dict keys throughout.
FACES = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")


# ---------------------------------------------------------------------------
# Boundary specification
# ---------------------------------------------------------------------------
@dataclass
class Boundary:
    """One domain face. kind in {'wall', 'inlet', 'outlet'}.

    For 'wall', `vel` is the (vx, vy, vz) wall velocity (nonzero for a moving
    lid); the normal component is forced to zero, tangential components are the
    no-slip target. For 'inlet', `vel` is the prescribed velocity vector, and
    `mask_fn(a, b) -> bool array` optionally restricts the inlet to part of the
    face (a, b are the two in-plane cell-center coordinates); default: whole face.
    For 'outlet', velocity is extrapolated (zero-gradient) as a predictor; under
    the default Config.outlet_mode='pressure' the opening is a Dirichlet p=0
    boundary and the projection then solves for the actual outflow.

    `mask`, `vel_map`, `temp_map` are per-cell alternatives to the scalar
    `vel`/`temp`, for data-center BCs that vary across a face (a perforated-
    tile floor, ceiling return tiles): in-plane arrays (shape matching the two
    non-normal axes, meshgrid 'ij' order) giving, respectively, the opening
    mask (alternative to `mask_fn`), the prescribed *normal* velocity per
    opening cell (overrides `vel[normal]`), and the prescribed temperature per
    opening cell (overrides `temp`).
    """

    kind: str
    vel: tuple = (0.0, 0.0, 0.0)
    mask_fn: object = None
    temp: float = None  # Dirichlet T at the opening / wall face
    #                           (None -> adiabatic, zero-gradient in energy)
    wall_temp: float = None  # Dirichlet T on the *solid remainder* of an
    #                           inlet/outlet face (the wall around the slot)
    mask: np.ndarray = None
    vel_map: np.ndarray = None
    temp_map: np.ndarray = None


@dataclass
class RackSpec:
    """A flow-through rack: draws `Q_m3s` in the front face and exhausts it,
    warmed by `power_W`, out the rear face (Han Eqs. 11-12).

    `axis` is the rack's depth axis ('x', 'y' or 'z'); `front` is the
    cell-index side of the block ('lo' or 'hi') the intake sits on -- the
    opposite side is the exhaust. Both faces carry the same signed normal
    velocity (flow direction = +axis if front='lo', -axis if front='hi').
    """

    axis: str
    front: str
    Q_m3s: float
    power_W: float


@dataclass
class Solid:
    """An axis-aligned internal solid block (e.g. the heated box, Case 2).

    `bounds` = (x0, x1, y0, y1, z0, z1) in metres. A cell whose center lies
    inside the block is blocked: its velocity faces are frozen at zero (no
    penetration + no-slip) and it is excluded from the pressure system. If
    `temp` is given, the block's surface is a Dirichlet-T boundary for the
    energy equation; if None, the surface is adiabatic. For the discretization
    to place the surface exactly on cell faces, build the grid with faces on
    the block bounds (as the Case 2 driver does).

    If `rack` is given, the block is a flow-through rack (`RackSpec`): its
    front/rear faces are prescribed velocities (not frozen at zero) and its
    rear-face surface temperature is recomputed every step from the front-
    face inlet temperature (see `Solver._update_rack_exhaust`), overriding
    `temp`.
    """

    bounds: tuple
    temp: float = None
    rack: RackSpec = None


@dataclass
class Config:
    nu: float = 1.5e-5  # kinematic (molecular) viscosity [m^2/s]
    dt: float = 0.05  # time step [s]
    rho: float = 1.2  # density [kg/m^3] (only for reporting)
    n_mom_sweeps: int = 4  # Jacobi sweeps per momentum solve per step
    bcs: dict = field(default_factory=dict)  # face name -> Boundary
    # Turbulence: Chen & Xu (1998) zero-equation, nu_t = C * |V| * l with l the
    # distance to the nearest wall. `turb_model=None` -> laminar (nu_t = 0).
    # Outlet treatment. 'pressure' (default): the exhaust opening is a Dirichlet
    # p=0 boundary and the projection solves for the outflow, so global mass
    # balance emerges rather than being imposed (and no arbitrary pin cell is
    # needed).  'velocity': the Zuo & Chen / Han FFD form -- the outflow profile
    # is extrapolated (zero-gradient) and corrected to match the inflow, with a
    # Neumann pressure BC everywhere.  Both give the same profiles to within
    # 0.04 NRMSD points; 'pressure' is numerically cleaner (max|div| ~25x lower
    # on Case 2) and is what the drivers use.
    outlet_mode: str = "pressure"  # 'pressure' or 'velocity'
    # Pressure Poisson solver. The operator is built once and reused every step,
    # so this only changes *how* the same linear system is solved, never the
    # system itself; all direct options agree to ~1e-14 relative residual.
    #   'cholesky'  cholespy Cholesky -- exploits the operator's symmetry
    #               (fastest measured: 4.1x the default LU back-solve).
    #               Falls back to 'lu_mmd' if cholespy is not installed.
    #   'lu_mmd'    SuperLU with MMD_AT_PLUS_A ordering (2.1x, no extra deps).
    #   'lu'        SuperLU with scipy's default COLAMD ordering (the original).
    #   'amg'       pyamg Ruge-Stuben + CG, warm-started. Iterative, so it is
    #               the only option whose accuracy depends on `pressure_tol`.
    #               Measured ~10x SLOWER than 'cholesky' at 40^3 -- kept because
    #               it is the option that scales best to much larger grids.
    # 'cholesky' and 'amg' require a symmetric operator; the operator is always
    # symmetric here, whether it has a pressure Dirichlet (outlet_mode=
    # 'pressure') or a pure-Neumann pin (outlet_mode='velocity', or a domain
    # with no pressure outlet at all, e.g. a data-center case) -- the pin zeros
    # both its row and column (see pressure.build_pressure_operator).
    pressure_solver: str = "cholesky"
    pressure_tol: float = 1e-10  # relative residual; 'amg' only
    turb_model: str = None  # None or 'chen'
    turb_C: float = 0.03874  # Chen zero-equation coefficient
    # Dhoot et al. approximate wall function: cells adjacent to a domain
    # boundary (wall or inlet/outlet opening) or an internal solid use a
    # reduced coefficient instead of turb_C. Values (chen_a, jim_a) and the
    # adjacency rule are taken verbatim from Han's own reference solver,
    # doetools/isat_ffd, src/ffd_isat/Kernels_3D.cl::nu_t_chen_zero_equ (the
    # code behind Han et al.'s Table 2 numbers) -- Han's paper cites the wall
    # function [42] but never states its formula, so this is sourced from the
    # implementation, not tuned to our own NRMSD.
    turb_C_wall: float = 0.0185  # isat_ffd's `jim_a`
    # Energy equation + Boussinesq buoyancy (all ignored unless solve_energy).
    solve_energy: bool = False
    alpha: float = 2.1e-5  # molecular thermal diffusivity [m^2/s] (Pr~0.71)
    # No turbulent-Prandtl division: isat_ffd's diff_T kernel (Kernels_3D.cl,
    # same source as turb_C_wall above) reuses the momentum eddy viscosity
    # nu_t UNDIVIDED for the temperature equation's diffusion coefficient
    # (effectively Pr_t=1). We keep the physically-correct molecular alpha
    # (the nu-vs-alpha choice is negligible either way -- nu_t/nu ~200-300x
    # in this flow) but match their turbulent term exactly, since that is the
    # numerically significant, code-sourced part.
    beta: float = 1.0 / 295.15  # thermal expansion coefficient [1/K]
    g: float = 9.81  # gravitational acceleration [m/s^2]
    T_ref: float = 22.2  # Boussinesq reference temperature (T units)
    T_init: float = None  # initial uniform T (default: T_ref)
    n_energy_sweeps: int = 4  # Jacobi sweeps per energy solve per step
    cp: float = 1006.0  # specific heat [J/kg K] (rack exhaust carry-through)
    solids: list = field(default_factory=list)  # internal Solid blocks
