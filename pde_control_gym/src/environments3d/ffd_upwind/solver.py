"""FFD-Upwind solver on a staggered (MAC) grid.

Implements the proposed model of Han et al. (2021):

  1. Solve the combined advection-diffusion momentum equation (their Eq. 7)
     IMPLICITLY with a first-order-upwind finite-volume scheme (point-Jacobi),
     giving an intermediate velocity u*.  Pressure is NOT in this step.
  2. Project u* onto a divergence-free field (their Eqs. 5-6) by solving a
     pressure Poisson equation and correcting the face velocities.

Everything is written in kinematic form (momentum divided by density, so the
pressure here is p/rho). Convective face flux F = u_face * area is a volume flow
rate; diffusion coefficient D = nu * area / distance.

Grid / field layout (see grid.py):
    p, T, nu_t : cell centers            (nx,   ny,   nz)
    u          : x-faces                 (nx+1, ny,   nz)
    v          : y-faces                 (nx,   ny+1, nz)
    w          : z-faces                 (nx,   ny,   nz+1)

This module supports: no-slip walls (with optional tangential wall velocity,
e.g. a moving lid), prescribed inlets, zero-gradient outlets, the Chen & Xu
zero-equation turbulence model (M2), internal solid blocks with Dirichlet-
temperature surfaces (the heated box, M3), a cell-centered energy transport
equation, and Boussinesq buoyancy in the z-momentum (M3).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .grid import Grid
from .kernels import jacobi, linf_residual


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
    temp: float = None        # Dirichlet T at the opening / wall face
    #                           (None -> adiabatic, zero-gradient in energy)
    wall_temp: float = None   # Dirichlet T on the *solid remainder* of an
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


FACES = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")


@dataclass
class Config:
    nu: float = 1.5e-5          # kinematic (molecular) viscosity [m^2/s]
    dt: float = 0.05            # time step [s]
    rho: float = 1.2            # density [kg/m^3] (only for reporting)
    n_mom_sweeps: int = 4       # Jacobi sweeps per momentum solve per step
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
    outlet_mode: str = "pressure"   # 'pressure' or 'velocity'
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
    # both its row and column (see _build_pressure_operator).
    pressure_solver: str = "cholesky"
    pressure_tol: float = 1e-10     # relative residual; 'amg' only
    turb_model: str = None      # None or 'chen'
    turb_C: float = 0.03874     # Chen zero-equation coefficient
    # Dhoot et al. approximate wall function: cells adjacent to a domain
    # boundary (wall or inlet/outlet opening) or an internal solid use a
    # reduced coefficient instead of turb_C. Values (chen_a, jim_a) and the
    # adjacency rule are taken verbatim from Han's own reference solver,
    # doetools/isat_ffd, src/ffd_isat/Kernels_3D.cl::nu_t_chen_zero_equ (the
    # code behind Han et al.'s Table 2 numbers) -- Han's paper cites the wall
    # function [42] but never states its formula, so this is sourced from the
    # implementation, not tuned to our own NRMSD.
    turb_C_wall: float = 0.0185     # isat_ffd's `jim_a`
    # Energy equation + Boussinesq buoyancy (all ignored unless solve_energy).
    solve_energy: bool = False
    alpha: float = 2.1e-5       # molecular thermal diffusivity [m^2/s] (Pr~0.71)
    # No turbulent-Prandtl division: isat_ffd's diff_T kernel (Kernels_3D.cl,
    # same source as turb_C_wall above) reuses the momentum eddy viscosity
    # nu_t UNDIVIDED for the temperature equation's diffusion coefficient
    # (effectively Pr_t=1). We keep the physically-correct molecular alpha
    # (the nu-vs-alpha choice is negligible either way -- nu_t/nu ~200-300x
    # in this flow) but match their turbulent term exactly, since that is the
    # numerically significant, code-sourced part.
    beta: float = 1.0 / 295.15  # thermal expansion coefficient [1/K]
    g: float = 9.81             # gravitational acceleration [m/s^2]
    T_ref: float = 22.2         # Boussinesq reference temperature (T units)
    T_init: float = None        # initial uniform T (default: T_ref)
    n_energy_sweeps: int = 4    # Jacobi sweeps per energy solve per step
    cp: float = 1006.0          # specific heat [J/kg K] (rack exhaust carry-through)
    solids: list = field(default_factory=list)   # internal Solid blocks


class Solver:
    def __init__(self, grid: Grid, cfg: Config):
        self.g = grid
        self.cfg = cfg
        nx, ny, nz = grid.nx, grid.ny, grid.nz

        # --- fields --------------------------------------------------------
        self.u = np.zeros((nx + 1, ny, nz))
        self.v = np.zeros((nx, ny + 1, nz))
        self.w = np.zeros((nx, ny, nz + 1))
        self.p = np.zeros((nx, ny, nz))
        self.nu_t = np.zeros((nx, ny, nz))          # eddy viscosity (0 => laminar)
        self._nue = np.full((nx, ny, nz), cfg.nu)   # effective viscosity nu + nu_t
        self._alpha_eff = np.full((nx, ny, nz), cfg.alpha)  # alpha + nu_t
        # Temperature field (cell centers). Solid cells are pinned to their
        # surface temperature; fluid starts uniform at T_init (default T_ref).
        T0 = cfg.T_ref if cfg.T_init is None else cfg.T_init
        self.T = np.full((nx, ny, nz), float(T0))
        # Extra kinematic w-momentum source (m/s^2), e.g. a tile body force
        # (Eq. 10); added to buoyancy (if any) in step(). None -> no-op.
        self.w_source_extra: np.ndarray = None

        self._precompute_metrics()
        self._setup_solids()
        self._precompute_wall_distance()
        self._setup_face_bcs()
        self._setup_energy_bcs()
        self._build_pressure_operator()
        self.apply_velocity_bcs()

    # ------------------------------------------------------------------
    # Metrics: 1-D geometry arrays broadcast in the assembly routines
    # ------------------------------------------------------------------
    def _precompute_metrics(self):
        g = self.g
        self.dx, self.dy, self.dz = g.x.d, g.y.d, g.z.d
        self.dxc, self.dyc, self.dzc = g.x.dc, g.y.dc, g.z.dc
        # CV widths for the staggered momentum control volumes.
        self.wxu = g.x.dc            # u-CV width in x, length nx+1
        self.wyv = g.y.dc            # v-CV width in y, length ny+1
        self.wzw = g.z.dc            # w-CV width in z, length nz+1

    # ------------------------------------------------------------------
    # Internal solids (heated box, M3)
    # ------------------------------------------------------------------
    def _setup_solids(self):
        """Build the cell-centered solid mask and its velocity-face masks.

        A cell is solid if its center lies inside any configured Solid block.
        A velocity face is a solid face if *either* adjacent cell is solid;
        those faces carry a frozen zero velocity (no penetration + no-slip),
        EXCEPT a rack's front/rear faces, which carry its prescribed flow
        (see `_setup_rack`) instead of zero -- still frozen (fixed), just not
        at zero. `solid_temp` holds each solid cell's surface temperature (NaN
        if the block is adiabatic), read by neighbouring fluid cells in the
        energy assembly; a rack's rear-face temperature is overwritten every
        step by `_update_rack_exhaust`.
        """
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        self.solid = np.zeros((nx, ny, nz), dtype=np.bool_)
        self.solid_temp = np.full((nx, ny, nz), np.nan)
        Xc, Yc, Zc = g.centers()
        rack_masks = []
        for sol in self.cfg.solids:
            x0, x1, y0, y1, z0, z1 = sol.bounds
            m = ((Xc > x0) & (Xc < x1) & (Yc > y0) & (Yc < y1)
                 & (Zc > z0) & (Zc < z1))
            self.solid |= m
            if sol.temp is not None:
                self.solid_temp[m] = sol.temp
            if sol.rack is not None:
                rack_masks.append((sol.rack, m))

        S = self.solid
        self._u_solid = np.zeros_like(self.u, dtype=np.bool_)
        self._v_solid = np.zeros_like(self.v, dtype=np.bool_)
        self._w_solid = np.zeros_like(self.w, dtype=np.bool_)
        self._u_solid[1:nx, :, :] = S[:-1, :, :] | S[1:, :, :]
        self._v_solid[:, 1:ny, :] = S[:, :-1, :] | S[:, 1:, :]
        self._w_solid[:, :, 1:nz] = S[:, :, :-1] | S[:, :, 1:]

        self._racks = [self._setup_rack(spec, m) for spec, m in rack_masks]

    _RACK_AXES = {"x": 0, "y": 1, "z": 2}

    def _rack_velocity_array(self, axis):
        return {"x": self.u, "y": self.v, "z": self.w}[axis]

    def _rack_transverse_area(self, axis):
        if axis == "x":
            return np.multiply.outer(self.dy, self.dz)
        if axis == "y":
            return np.multiply.outer(self.dx, self.dz)
        return np.multiply.outer(self.dx, self.dy)

    def _setup_rack(self, spec, m):
        """Prescribe a rack's front/rear face velocities; return its exhaust
        bookkeeping tuple `(spec, ax, front_idx, rear_idx, footprint, V)` used
        by `_update_rack_exhaust` (front_idx/rear_idx are cell indices along
        `ax`: the fluid inlet layer and the rack's rear-most solid layer).
        """
        if spec.front not in ("lo", "hi"):
            raise ValueError(f"rack {spec}: front must be 'lo' or 'hi'")
        ax = self._RACK_AXES.get(spec.axis)
        if ax is None:
            raise ValueError(f"rack {spec}: axis must be 'x', 'y' or 'z'")
        n_axis = (self.g.nx, self.g.ny, self.g.nz)[ax]
        other = tuple(a for a in range(3) if a != ax)
        along = np.where(m.any(axis=other))[0]
        if along.size == 0:
            raise ValueError(f"rack {spec}: solid block has no cells")
        i0, i1 = int(along.min()), int(along.max())
        footprint = np.take(m, i0, axis=ax)

        outside_lo, outside_hi = i0 - 1, i1 + 1
        if spec.front == "lo":
            front_face, rear_face = i0, i1 + 1
            front_idx, rear_outside_idx = outside_lo, outside_hi
            rear_idx = i1
            sign = 1.0
        else:
            front_face, rear_face = i1 + 1, i0
            front_idx, rear_outside_idx = outside_hi, outside_lo
            rear_idx = i0
            sign = -1.0

        def axis_slice(idx):
            sl = [slice(None)] * 3
            sl[ax] = idx
            return tuple(sl)

        for label, idx in (("front", front_idx), ("rear", rear_outside_idx)):
            if not (0 <= idx < n_axis):
                raise ValueError(f"rack {spec}: {label} face is at the domain "
                                  f"boundary")
            if self.solid[axis_slice(idx)][footprint].any():
                raise ValueError(f"rack {spec}: {label} cell is not fluid")

        area2d = self._rack_transverse_area(spec.axis)
        A_front = float(area2d[footprint].sum())
        V = spec.Q_m3s / A_front

        vel = self._rack_velocity_array(spec.axis)
        vel[axis_slice(front_face)][footprint] = sign * V
        vel[axis_slice(rear_face)][footprint] = sign * V

        return (spec, ax, front_idx, rear_idx, footprint, V,
                front_face, rear_face, sign, A_front)

    def _axis_slice(self, ax, idx):
        sl = [slice(None)] * 3
        sl[ax] = idx
        return tuple(sl)

    def _update_rack_exhaust(self):
        """Carry the front-face inlet temperature through to the rear-face
        surface temperature, raised by the rack's power (Eq. 12): each rack's
        rear-most solid layer gets `solid_temp = T[front fluid layer] +
        power_W / (rho * cp * Q_m3s)`.
        """
        rho, cp = self.cfg.rho, self.cfg.cp
        for spec, ax, front_idx, rear_idx, footprint, *_ in self._racks:
            sl = [slice(None)] * 3
            sl[ax] = front_idx
            T_front = self.T[tuple(sl)]
            sl[ax] = rear_idx
            dT = spec.power_W / (rho * cp * spec.Q_m3s)
            self.solid_temp[tuple(sl)][footprint] = T_front[footprint] + dT

    def set_rack_powers(self, powers_W, flows_m3s):
        """Update every flow-through rack's IT power and intake flow in place,
        re-prescribing its front/rear face velocities, so a warm-started field
        can be driven with a time-varying load without rebuilding the solver.

        `powers_W[k]`/`flows_m3s[k]` are the new power [W] and airflow [m^3/s]
        of the k-th rack, indexed in `self._racks` order (the order racks appear
        in `cfg.solids`). The new power feeds the Eq. 12 exhaust rise on the next
        `_update_rack_exhaust`; the new flow re-freezes the front/rear face
        velocities here (the momentum solve holds those solid faces fixed, so
        rewriting the array is enough)."""
        if len(powers_W) != len(self._racks) or len(flows_m3s) != len(self._racks):
            raise ValueError(
                f"set_rack_powers: expected {len(self._racks)} racks, got "
                f"{len(powers_W)} powers / {len(flows_m3s)} flows")
        for (spec, ax, front_idx, rear_idx, footprint, _V,
             front_face, rear_face, sign, A_front), P, Q in zip(
                self._racks, powers_W, flows_m3s):
            spec.power_W = float(P)
            spec.Q_m3s = float(Q)
            V = float(Q) / A_front
            vel = self._rack_velocity_array(spec.axis)
            vel[self._axis_slice(ax, front_face)][footprint] = sign * V
            vel[self._axis_slice(ax, rear_face)][footprint] = sign * V

    # ------------------------------------------------------------------
    # Turbulence: Chen & Xu (1998) zero-equation eddy viscosity
    # ------------------------------------------------------------------
    def _precompute_wall_distance(self):
        """Cell-centered distance to the nearest *wall*, l(x,y,z) -- the Chen
        zero-equation length scale.

        Matches Han's reference solver isat_ffd (utility.c::min_distance):
        the length scale is the distance to the nearest no-slip wall, and
        inlet/outlet openings are NOT walls (isat_ffd skips INLET/OUTLET cells
        when building the min-distance field).  A cell just under a perforated
        tile, or beside an AHU inlet, takes l from the nearest real wall, not
        from the opening it faces.  (The earlier version treated every bounding
        face as a wall; that badly under-set l near the plenum's full-short-wall
        inlets and perforated top -- where the flow is fastest -- flattening
        nu_t there and under-predicting tile-flow non-uniformity in shallow
        plenums; see the D8 Fig-12 work.)

        For a fluid cell away from any opening the nearest wall point sits
        directly perpendicular at the same in-plane position, so l equals the
        old perpendicular-to-face distance there; the two differ only near an
        opening (where the perpendicular foot would land in the opening and the
        nearest wall is offset).  Enumerating the wall boundary cells and taking
        the per-cell nearest (via a KDTree) is isat_ffd's own approach.  Each
        internal solid adds its axis-aligned exterior distance analytically;
        solid cells get l = 0.
        """
        from scipy.spatial import cKDTree
        g = self.g
        xc, yc, zc = g.x.c, g.y.c, g.z.c
        # per face: (in-plane axis-a coords, axis-b coords, normal coord, order)
        face_geom = {
            "xlo": (yc, zc, g.x.f[0], "yz"), "xhi": (yc, zc, g.x.f[-1], "yz"),
            "ylo": (xc, zc, g.y.f[0], "xz"), "yhi": (xc, zc, g.y.f[-1], "xz"),
            "zlo": (xc, yc, g.z.f[0], "xy"), "zhi": (xc, yc, g.z.f[-1], "xy"),
        }
        pts = []
        for face, (a, b, n_coord, order) in face_geom.items():
            wall = ~self._opening_face_mask(face, a, b)   # True where face is wall
            if not wall.any():
                continue
            A, B = np.meshgrid(a, b, indexing="ij")
            Aw, Bw, Nw = A[wall], B[wall], np.full(int(wall.sum()), n_coord)
            if order == "yz":
                pts.append(np.column_stack([Nw, Aw, Bw]))
            elif order == "xz":
                pts.append(np.column_stack([Aw, Nw, Bw]))
            else:  # xy
                pts.append(np.column_stack([Aw, Bw, Nw]))

        Xc, Yc, Zc = g.centers()
        query = np.column_stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])
        if pts:
            dist = cKDTree(np.vstack(pts)).query(query)[0].reshape(Xc.shape)
        else:
            dist = np.full(Xc.shape, np.inf)

        zero = np.zeros_like(Xc)
        for sol in self.cfg.solids:
            x0, x1, y0, y1, z0, z1 = sol.bounds
            ddx = np.maximum.reduce([x0 - Xc, Xc - x1, zero])
            ddy = np.maximum.reduce([y0 - Yc, Yc - y1, zero])
            ddz = np.maximum.reduce([z0 - Zc, Zc - z1, zero])
            dist = np.minimum(dist, np.sqrt(ddx**2 + ddy**2 + ddz**2))
        dist[self.solid] = 0.0
        self.wall_dist = dist
        self._precompute_near_boundary()

    def _opening_face_mask(self, face, a, b):
        """bool[(len(a), len(b))]: True where `face` carries an inlet/outlet
        opening.  Reads cfg.bcs directly, since this runs before
        _setup_face_bcs builds self.bc.  A face with no BC, or a wall/slip BC,
        has no openings; a mask/mask_fn restricts the opening to part of the
        face; otherwise the whole face is the opening.  In-plane axis order
        matches the solver's own `opening_mask` (meshgrid 'ij')."""
        bc = self.cfg.bcs.get(face)
        if bc is None or bc.kind not in ("inlet", "outlet"):
            return np.zeros((len(a), len(b)), dtype=bool)
        if bc.mask is not None:
            return np.asarray(bc.mask, dtype=bool)
        if bc.mask_fn is not None:
            A, B = np.meshgrid(a, b, indexing="ij")
            return np.asarray(bc.mask_fn(A, B), dtype=bool)
        return np.ones((len(a), len(b)), dtype=bool)

    def _precompute_near_boundary(self):
        """Cell-centered mask for the Dhoot wall-function coefficient switch.

        True for any cell adjacent (6-neighbor) to a domain boundary (the
        outermost layer of cells, which sit against the wall/inlet/outlet
        ghost layer) or to an internal solid. Matches isat_ffd's condition
        `flagp[neighbor] >= 0` for a non-fluid neighbor (SOLID, INLET,
        OUTLET, RACK_INLET/OUTLET all satisfy that; only FLUID is < 0).
        """
        nx, ny, nz = self.g.nx, self.g.ny, self.g.nz
        m = np.zeros((nx, ny, nz), dtype=np.bool_)
        m[0, :, :] = m[nx - 1, :, :] = True
        m[:, 0, :] = m[:, ny - 1, :] = True
        m[:, :, 0] = m[:, :, nz - 1] = True
        S = self.solid
        m[:-1, :, :] |= S[1:, :, :]; m[1:, :, :] |= S[:-1, :, :]
        m[:, :-1, :] |= S[:, 1:, :]; m[:, 1:, :] |= S[:, :-1, :]
        m[:, :, :-1] |= S[:, :, 1:]; m[:, :, 1:] |= S[:, :, :-1]
        self._near_bnd = m

    def update_turbulence(self):
        """Recompute nu_t and the effective viscosity nu_eff = nu + nu_t.

        No-op (nu_eff stays at the molecular nu) unless turb_model == 'chen'.
        """
        if self.cfg.turb_model is None:
            return
        Uc, Vc, Wc = self.velocity_at_centers()
        Vmag = np.sqrt(Uc * Uc + Vc * Vc + Wc * Wc)
        coeff = np.where(self._near_bnd, self.cfg.turb_C_wall, self.cfg.turb_C)
        self.nu_t = coeff * Vmag * self.wall_dist
        self._nue = self.cfg.nu + self.nu_t

    # ------------------------------------------------------------------
    # Boundary bookkeeping
    # ------------------------------------------------------------------
    def _setup_face_bcs(self):
        """Precompute inlet masks and fixed-node markers for each velocity."""
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        self.bc = {name: self.cfg.bcs.get(name, Boundary("wall")) for name in FACES}

        # Fixed (Dirichlet) markers for the normal-velocity boundary nodes.
        self.u_fixed = np.zeros_like(self.u, dtype=np.bool_)
        self.v_fixed = np.zeros_like(self.v, dtype=np.bool_)
        self.w_fixed = np.zeros_like(self.w, dtype=np.bool_)
        # Boundary u-nodes are i=0 and i=nx; v-nodes j=0,ny; w-nodes k=0,nz.
        self.u_fixed[0, :, :] = True
        self.u_fixed[nx, :, :] = True
        self.v_fixed[:, 0, :] = True
        self.v_fixed[:, ny, :] = True
        self.w_fixed[:, :, 0] = True
        self.w_fixed[:, :, nz] = True
        # Internal solid faces are frozen at zero as well (no penetration +
        # no-slip); they start at 0 and are never touched by the BC/projection.
        self.u_fixed |= self._u_solid
        self.v_fixed |= self._v_solid
        self.w_fixed |= self._w_solid
        # Pin solid cells to their surface temperature (used for buoyancy /
        # visualization; the energy solve holds them fixed).
        m = self.solid & ~np.isnan(self.solid_temp)
        self.T[m] = self.solid_temp[m]

        # Opening mask per face: the in-plane cells where an inlet/outlet is
        # active (mask_fn selects a sub-region of the face; default = whole
        # face).  A face can therefore be part opening, part solid wall, which
        # is how the room's slot diffuser (inlet) and exhaust (outlet) sit on
        # otherwise-solid end walls.  In-plane coords are the two axes NOT
        # normal to the face.
        yc, zc, xc = g.y.c, g.z.c, g.x.c

        def opening_mask(face, a, b):
            bc = self.bc[face]
            if bc.mask is not None:
                return np.asarray(bc.mask, dtype=np.bool_)
            A, B = np.meshgrid(a, b, indexing="ij")
            if bc.mask_fn is None:
                return np.ones_like(A, dtype=np.bool_)
            return np.asarray(bc.mask_fn(A, B), dtype=np.bool_)

        inplane = {"xlo": (yc, zc), "xhi": (yc, zc),
                   "ylo": (xc, zc), "yhi": (xc, zc),
                   "zlo": (xc, yc), "zhi": (xc, yc)}
        self._facemask = {f: opening_mask(f, *inplane[f]) for f in FACES}

        # Pressure BC per boundary face.  Walls and inlets are always Neumann:
        # an inlet is velocity-specified, so also fixing its pressure would
        # over-determine it.  Under cfg.outlet_mode == 'pressure' the outlet
        # OPENING is Dirichlet p=0, which lets the projection solve for the
        # outflow (and makes the system nonsingular without a pin cell); under
        # 'velocity' every face is Neumann and _enforce_global_mass imposes the
        # balance by hand instead.
        self._pdir = {f: np.zeros_like(self._facemask[f], dtype=np.bool_)
                      for f in FACES}
        if self.cfg.outlet_mode == "pressure":
            for f in FACES:
                if self.bc[f].kind == "outlet":
                    self._pdir[f] = self._facemask[f].copy()
        elif self.cfg.outlet_mode != "velocity":
            raise ValueError(f"outlet_mode must be 'pressure' or 'velocity', "
                             f"got {self.cfg.outlet_mode!r}")

    def _setup_energy_bcs(self):
        """Per-domain-face temperature BC: which face cells are Dirichlet and
        at what value.  Everything else (outlet opening, adiabatic/slip face)
        is a zero-gradient Neumann face, which contributes no energy term.

        The convective inflow at an inlet is handled in the assembly directly
        from the face velocity, so only the Dirichlet value is stored here.
        Layout per face: (is_dir, T_value) as in-plane arrays.
        """
        self._Tbc = {}
        if not self.cfg.solve_energy:
            return
        for face in FACES:
            bc = self.bc[face]
            m = self._facemask[face]                     # opening (slot) mask
            is_dir = np.zeros_like(m, dtype=np.bool_)
            Tval = np.zeros_like(m, dtype=np.float64)
            if bc.kind in ("wall",):
                if bc.temp is not None:
                    is_dir[:] = True
                    Tval[:] = bc.temp
            elif bc.kind == "inlet":
                if bc.temp_map is not None:               # per-cell supply temp
                    is_dir[m] = True
                    Tval[m] = bc.temp_map[m]
                elif bc.temp is not None:                 # slot: supply air temp
                    is_dir[m] = True
                    Tval[m] = bc.temp
                if bc.wall_temp is not None:             # wall around the slot
                    is_dir[~m] = True
                    Tval[~m] = bc.wall_temp
            elif bc.kind == "outlet":
                if bc.wall_temp is not None:             # wall around the exhaust
                    is_dir[~m] = True
                    Tval[~m] = bc.wall_temp
                # opening cells stay Neumann (zero-gradient outflow)
            # 'slip' -> adiabatic symmetry (Neumann), nothing to store.
            self._Tbc[face] = (is_dir, Tval)

    def apply_velocity_bcs(self):
        """Set boundary normal-velocity nodes (inlet/wall/outlet) on u, v, w."""
        nx, ny, nz = self.g.nx, self.g.ny, self.g.nz

        # x-normal faces act on u
        for face, iface in (("xlo", 0), ("xhi", nx)):
            bc = self.bc[face]
            m = self._facemask[face]
            if bc.kind == "inlet":
                self.u[iface, :, :] = 0.0            # solid part of the face
                if bc.vel_map is not None:
                    self.u[iface][m] = bc.vel_map[m]  # per-cell prescribed vel
                else:
                    self.u[iface][m] = bc.vel[0]     # opening: prescribed normal vel
            elif bc.kind in ("wall", "slip"):
                self.u[iface, :, :] = 0.0            # no through-flow
            elif bc.kind == "outlet":
                src = self.u[1, :, :] if iface == 0 else self.u[nx - 1, :, :]
                self.u[iface, :, :] = 0.0            # solid part of the face
                self.u[iface][m] = src[m]            # opening: zero-gradient

        for face, jface in (("ylo", 0), ("yhi", ny)):
            bc = self.bc[face]
            m = self._facemask[face]
            if bc.kind == "inlet":
                self.v[:, jface, :] = 0.0
                if bc.vel_map is not None:
                    self.v[:, jface, :][m] = bc.vel_map[m]
                else:
                    self.v[:, jface, :][m] = bc.vel[1]
            elif bc.kind in ("wall", "slip"):
                self.v[:, jface, :] = 0.0
            elif bc.kind == "outlet":
                src = self.v[:, 1, :] if jface == 0 else self.v[:, ny - 1, :]
                self.v[:, jface, :] = 0.0
                self.v[:, jface, :][m] = src[m]

        for face, kface in (("zlo", 0), ("zhi", nz)):
            bc = self.bc[face]
            m = self._facemask[face]
            if bc.kind == "inlet":
                self.w[:, :, kface] = 0.0
                if bc.vel_map is not None:
                    self.w[:, :, kface][m] = bc.vel_map[m]
                else:
                    self.w[:, :, kface][m] = bc.vel[2]
            elif bc.kind in ("wall", "slip"):
                self.w[:, :, kface] = 0.0
            elif bc.kind == "outlet":
                src = self.w[:, :, 1] if kface == 0 else self.w[:, :, nz - 1]
                self.w[:, :, kface] = 0.0
                self.w[:, :, kface][m] = src[m]

        self._enforce_global_mass()

    def _enforce_global_mass(self):
        """Correct the outlet normal velocity so total outflow == total inflow.

        Only used when cfg.outlet_mode == 'velocity'; under the default
        'pressure' outlet the projection solves for the outflow itself and this
        is a no-op (imposing the balance here as well would fight the pressure
        correction).

        The correction is ADDITIVE -- a uniform velocity offset spread over the
        outlet opening -- not a multiplicative rescale.  A multiplicative fix
        (u_out *= inflow/cur) has a pole wherever the extrapolated outflow `cur`
        passes through zero, and in Case 2 it sits right on it: the raw
        zero-gradient profile at the floor-level exhaust is net *backflow*
        (cur ~ -0.06 m^3/s against a +0.10 target), so the rescale sign-flips and
        stretches the profile by ~-1.7 every step.  Small dt orbits that pole in
        a limit cycle; at Han's dt = 0.05 s the state crosses cur = 0, the scale
        factor spikes (~+17), and the run diverges within ~90 steps.  The
        additive form has no pole, is stable to at least dt = 0.1 s, and leaves a
        physical ~0.5 m/s exhaust profile instead of a +/-5 m/s artifact.
        """
        if self.cfg.outlet_mode == "pressure":
            return
        nx, ny, nz = self.g.nx, self.g.ny, self.g.nz
        Ax = self.dy[:, None] * self.dz[None, :]     # (ny,nz)
        Ay = self.dx[:, None] * self.dz[None, :]     # (nx,nz)
        Az = self.dx[:, None] * self.dy[None, :]     # (nx,ny)

        inflow = 0.0
        outflow_faces = []  # (array_slice_setter, current_flux, area)

        # Gather inflow (into domain) and outlet faces.
        def face_flux(vel_plane, area, sign):
            return sign * np.sum(vel_plane * area)

        # x faces: outward normal is -x at xlo, +x at xhi
        specs = [
            ("xlo", self.u[0, :, :], Ax, -1.0, ("u", 0)),
            ("xhi", self.u[nx, :, :], Ax, +1.0, ("u", nx)),
            ("ylo", self.v[:, 0, :], Ay, -1.0, ("v", 0)),
            ("yhi", self.v[:, ny, :], Ay, +1.0, ("v", ny)),
            ("zlo", self.w[:, :, 0], Az, -1.0, ("w", 0)),
            ("zhi", self.w[:, :, nz], Az, +1.0, ("w", nz)),
        ]
        outlets = []
        for face, plane, area, sign, setter in specs:
            m = self._facemask[face]
            if self.bc[face].kind == "inlet":
                # inflow is negative outward flux, over the opening only
                inflow += -sign * np.sum(plane[m] * area[m])
            elif self.bc[face].kind == "outlet":
                outlets.append((face, setter, area, sign))

        if not outlets:
            return
        # Current total outflow through the outlet openings.
        cur = 0.0
        a_open = 0.0
        for face, (comp, idx), area, sign in outlets:
            m = self._facemask[face]
            arr = getattr(self, comp)
            plane = arr[idx, :, :] if comp == "u" else (
                arr[:, idx, :] if comp == "v" else arr[:, :, idx])
            cur += sign * np.sum(plane[m] * area[m])
            a_open += np.sum(area[m])
        if a_open <= 0.0:
            return
        # Uniform offset that closes the balance: sign * (inflow - cur) / A_open.
        # `sign` maps the outward-normal deficit back onto the stored component.
        for face, (comp, idx), area, sign in outlets:
            m = self._facemask[face]
            arr = getattr(self, comp)
            plane = arr[idx, :, :] if comp == "u" else (
                arr[:, idx, :] if comp == "v" else arr[:, :, idx])
            plane[m] += sign * (inflow - cur) / a_open

    # ------------------------------------------------------------------
    # Pressure Poisson operator (assembled once, factorized once)
    # ------------------------------------------------------------------
    def _build_pressure_operator(self):
        """Discrete Laplacian L (SPD) for the pressure, with Neumann at walls/
        inlets and Dirichlet p=0 at outlets. RHS is filled each step from the
        divergence of u*.  L p = -(1/dt) div(u*) * Vol.
        """
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        N = nx * ny * nz

        def idx(i, j, k):
            return (i * ny + j) * nz + k

        Ax = np.multiply.outer(self.dy, self.dz)   # (ny,nz)
        Ay = np.multiply.outer(self.dx, self.dz)   # (nx,nz)
        Az = np.multiply.outer(self.dx, self.dy)   # (nx,ny)

        # A face contributes a coefficient A/dist unless it's a domain-boundary
        # face with a Neumann pressure BC (wall/inlet) -> no term. Outlet faces
        # get a Dirichlet ghost at p=0 with distance = half cell.
        rows, cols, vals = [], [], []
        diag = np.zeros(N)

        # A boundary face gets a Dirichlet (p=0) ghost only where an outlet
        # opening sits (self._pdir); every other boundary face is Neumann
        # (wall/inlet/slip solid), which contributes no pressure term.
        pdir = self._pdir
        S = self.solid

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    P = idx(i, j, k)
                    # Solid cells are decoupled from the pressure system:
                    # identity row p=0 (RHS forced to 0 in project()).
                    if S[i, j, k]:
                        diag[P] = 1.0
                        continue
                    # A face couples two cells only if the neighbour is fluid;
                    # a solid neighbour is a no-flux (Neumann) interface, just
                    # like a wall, so it contributes no pressure term.
                    # East (+x)
                    if i + 1 < nx:
                        if not S[i + 1, j, k]:
                            c = Ax[j, k] / self.dxc[i + 1]
                            rows.append(P); cols.append(idx(i + 1, j, k)); vals.append(-c)
                            diag[P] += c
                    elif pdir["xhi"][j, k]:
                        diag[P] += Ax[j, k] / (self.dx[i] * 0.5)
                    # West (-x)
                    if i - 1 >= 0:
                        if not S[i - 1, j, k]:
                            c = Ax[j, k] / self.dxc[i]
                            rows.append(P); cols.append(idx(i - 1, j, k)); vals.append(-c)
                            diag[P] += c
                    elif pdir["xlo"][j, k]:
                        diag[P] += Ax[j, k] / (self.dx[i] * 0.5)
                    # North (+y)
                    if j + 1 < ny:
                        if not S[i, j + 1, k]:
                            c = Ay[i, k] / self.dyc[j + 1]
                            rows.append(P); cols.append(idx(i, j + 1, k)); vals.append(-c)
                            diag[P] += c
                    elif pdir["yhi"][i, k]:
                        diag[P] += Ay[i, k] / (self.dy[j] * 0.5)
                    # South (-y)
                    if j - 1 >= 0:
                        if not S[i, j - 1, k]:
                            c = Ay[i, k] / self.dyc[j]
                            rows.append(P); cols.append(idx(i, j - 1, k)); vals.append(-c)
                            diag[P] += c
                    elif pdir["ylo"][i, k]:
                        diag[P] += Ay[i, k] / (self.dy[j] * 0.5)
                    # Front (+z)
                    if k + 1 < nz:
                        if not S[i, j, k + 1]:
                            c = Az[i, j] / self.dzc[k + 1]
                            rows.append(P); cols.append(idx(i, j, k + 1)); vals.append(-c)
                            diag[P] += c
                    elif pdir["zhi"][i, j]:
                        diag[P] += Az[i, j] / (self.dz[k] * 0.5)
                    # Back (-z)
                    if k - 1 >= 0:
                        if not S[i, j, k - 1]:
                            c = Az[i, j] / self.dzc[k]
                            rows.append(P); cols.append(idx(i, j, k - 1)); vals.append(-c)
                            diag[P] += c
                    elif pdir["zlo"][i, j]:
                        diag[P] += Az[i, j] / (self.dz[k] * 0.5)

        # Pure-Neumann system (no Dirichlet cell anywhere) is singular: pin the
        # first fluid cell to p=0. Zeroing both the row AND the column (not
        # just the row) keeps the operator symmetric -- valid because p_pin is
        # truly 0, so dropping the -c*p_pin term other rows carried for it
        # changes nothing numerically, and it lets 'cholesky'/'amg' handle a
        # pure-Neumann domain (a data-center case with no pressure outlet)
        # without falling back to an unsymmetric solver.
        any_dirichlet = any(pdir[f].any() for f in FACES)
        rows.extend(range(N)); cols.extend(range(N)); vals.extend(diag)
        L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
        if not any_dirichlet:
            pin = int(np.argmin(S.reshape(-1)))   # first fluid cell
            L = L.tolil()
            L[pin, :] = 0.0
            L[:, pin] = 0.0
            L[pin, pin] = 1.0
            L = L.tocsc()
            self._pin_cell = pin
        else:
            self._pin_cell = None
        self._N = N
        self._idx_shape = (nx, ny, nz)
        self.Lsolve = self._make_pressure_solver(L)

    def _make_pressure_solver(self, L):
        """Build the reusable pressure solve for cfg.pressure_solver.

        The operator is constant, so every option factors/sets up once here and
        the per-step cost is only the solve.  See Config.pressure_solver. The
        pin (if any) is applied symmetrically (see _build_pressure_operator),
        so every pressure_solver option is available regardless of whether the
        domain has a pressure outlet.
        """
        want = self.cfg.pressure_solver
        N = L.shape[0]
        if want == "cholesky":
            try:
                from cholespy import CholeskySolverD, MatrixType
            except ImportError:
                want = "lu_mmd"
            else:
                coo = L.tocoo()
                chol = CholeskySolverD(N, coo.row.astype(np.int32),
                                       coo.col.astype(np.int32),
                                       coo.data.astype(np.float64), MatrixType.COO)
                buf = np.empty(N)

                def solve(rhs, _c=chol, _b=buf):
                    _c.solve(np.ascontiguousarray(rhs, dtype=np.float64), _b)
                    return _b.copy()        # caller keeps the result as self.p
                self.pressure_solver_used = "cholesky"
                return solve
        if want == "amg":
            try:
                import pyamg
            except ImportError:
                # pyamg is the default backend but an optional dependency; when
                # it is not installed, fall back to the scipy-only sparse LU
                # (same graceful-degradation pattern as 'cholesky' -> 'lu_mmd').
                want = "lu_mmd"
            else:
                ml = pyamg.ruge_stuben_solver(L.tocsr())
                tol = self.cfg.pressure_tol
                last = {"x": np.zeros(N)}

                def solve(rhs, _ml=ml, _t=tol, _s=last):
                    x = _ml.solve(rhs, x0=_s["x"], tol=_t, accel="cg", maxiter=200)
                    _s["x"] = x
                    return x
                self.pressure_solver_used = "amg"
                return solve
        if want == "lu_mmd":
            self.pressure_solver_used = "lu_mmd"
            return spla.splu(L.tocsc(), permc_spec="MMD_AT_PLUS_A").solve
        self.pressure_solver_used = "lu"
        return spla.factorized(L)

    def divergence(self, u=None, v=None, w=None):
        """Cell-centered divergence of the (staggered) velocity field."""
        u = self.u if u is None else u
        v = self.v if v is None else v
        w = self.w if w is None else w
        du = (u[1:, :, :] - u[:-1, :, :]) / self.dx[:, None, None]
        dv = (v[:, 1:, :] - v[:, :-1, :]) / self.dy[None, :, None]
        dw = (w[:, :, 1:] - w[:, :, :-1]) / self.dz[None, None, :]
        return du + dv + dw

    def project(self):
        """Solve the pressure Poisson equation and correct velocities."""
        dt = self.cfg.dt
        Vol = self.g.cell_volumes()
        div = self.divergence()
        rhs = -(1.0 / dt) * div * Vol
        # Solid cells are identity rows (p=0); force their RHS to 0.
        rhs[self.solid] = 0.0
        rhs = rhs.reshape(-1)
        if self._pin_cell is not None:
            rhs[self._pin_cell] = 0.0
        p = self.Lsolve(rhs).reshape(self._idx_shape)
        self.p = p
        # Correct interior face velocities: u -= dt * dp/dx.  Faces on a solid
        # interface stay frozen at zero (no correction), so the pressure never
        # drives flow into the box.
        cu = dt * (p[1:, :, :] - p[:-1, :, :]) / self.dxc[1:-1, None, None]
        cv = dt * (p[:, 1:, :] - p[:, :-1, :]) / self.dyc[None, 1:-1, None]
        cw = dt * (p[:, :, 1:] - p[:, :, :-1]) / self.dzc[None, None, 1:-1]
        cu[self._u_solid[1:-1, :, :]] = 0.0
        cv[self._v_solid[:, 1:-1, :]] = 0.0
        cw[self._w_solid[:, :, 1:-1]] = 0.0
        self.u[1:-1, :, :] -= cu
        self.v[:, 1:-1, :] -= cv
        self.w[:, :, 1:-1] -= cw
        # Inlet and wall normal faces were set in momentum_predict and are held
        # fixed; we do NOT re-apply the BCs here (re-extrapolating would clobber
        # the correction just made).
        #
        # Under a pressure outlet the outlet BOUNDARY face must be corrected too.
        # The operator gave those cells a Dirichlet ghost at p = 0 a half-cell
        # away, so the flux it assumes is corrected there is dt * (0 - p_P) /
        # (d/2).  Skipping it would leave the cell a residual divergence of
        # -(dt/Vol) * (A / (d/2)) * p_P -- small but growing linearly with dt.
        # This is also what closes the global mass balance: the outflow is
        # solved for here, not imposed by _enforce_global_mass.
        if self.cfg.outlet_mode == "pressure":
            self._correct_pressure_outlets(dt, p)

    def _correct_pressure_outlets(self, dt, p):
        """Apply the Dirichlet-ghost (p=0) velocity correction on outlet faces."""
        nx, ny, nz = self.g.nx, self.g.ny, self.g.nz
        for f in FACES:
            if self.bc[f].kind != "outlet":
                continue
            m = self._pdir[f]
            if not m.any():
                continue
            # Outward-normal faces gain +dt*p_P/(d/2); inward-normal (lo) faces
            # take the opposite sign, so both push flow out of the domain.
            if f == "xhi":
                self.u[nx][m] += dt * p[nx - 1, :, :][m] / (self.dx[nx - 1] * 0.5)
            elif f == "xlo":
                self.u[0][m] -= dt * p[0, :, :][m] / (self.dx[0] * 0.5)
            elif f == "yhi":
                self.v[:, ny, :][m] += dt * p[:, ny - 1, :][m] / (self.dy[ny - 1] * 0.5)
            elif f == "ylo":
                self.v[:, 0, :][m] -= dt * p[:, 0, :][m] / (self.dy[0] * 0.5)
            elif f == "zhi":
                self.w[:, :, nz][m] += dt * p[:, :, nz - 1][m] / (self.dz[nz - 1] * 0.5)
            elif f == "zlo":
                self.w[:, :, 0][m] -= dt * p[:, :, 0][m] / (self.dz[0] * 0.5)

    # ------------------------------------------------------------------
    # Momentum: implicit first-order-upwind advection-diffusion (Eq. 7)
    # ------------------------------------------------------------------
    def _face_mode(self, face, comp):
        """Classify a domain face for the *tangential* component `comp`.

        Returns (mode, value):
          'wall' -> no-slip Dirichlet target `value` (0 or moving-wall speed)
          'slip' -> free-slip / symmetry: zero tangential stress
          'none' -> outlet: zero-gradient, no wall pull
        Inlets impose only the normal component, so tangentially they read as a
        zero-velocity wall."""
        bc = self.bc[face]
        ci = {"u": 0, "v": 1, "w": 2}[comp]
        if bc.kind == "wall":
            return "wall", bc.vel[ci]
        if bc.kind == "slip":
            return "slip", 0.0
        if bc.kind == "inlet":
            return "wall", 0.0
        # outlet: a full-face opening extrapolates (zero tangential pull); a
        # masked slot sits on a mostly-solid wall, so treat it as no-slip wall.
        if bc.mask_fn is not None or bc.mask is not None:
            return "wall", 0.0
        return "none", 0.0

    def _apply_tangential_bcs(self, comp, AP, B, entries):
        """entries: list of (face, coeff_array, index_slice). Modifies AP, B."""
        for face, coeff, sl in entries:
            mode, val = self._face_mode(face, comp)
            if mode == "wall":
                if val != 0.0:
                    B[sl] += coeff[sl] * val
            elif mode == "slip":
                AP[sl] -= coeff[sl]   # remove the wall diffusion (zero stress)

    def _assemble_u(self):
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        dt = self.cfg.dt
        u, v, w = self.u, self.v, self.w
        dx, dy, dz = self.dx, self.dy, self.dz
        dxc, dyc, dzc = self.dxc, self.dyc, self.dzc
        wxu = self.wxu

        # Interior u-nodes: i = 1 .. nx-1
        Ax = np.multiply.outer(dy, dz)                         # (ny,nz) x-face area
        wx = wxu[1:nx]                                          # (nx-1,) CV x-width
        An = wx[:, None, None] * dz[None, None, :]              # (nx-1,1?,nz) y-face area
        An = wx[:, None, None] * np.ones(ny)[None, :, None] * dz[None, None, :]
        Az = wx[:, None, None] * dy[None, :, None] * np.ones(nz)[None, None, :]

        # Convecting fluxes (volumetric)
        ue = 0.5 * (u[1:nx, :, :] + u[2:nx + 1, :, :])
        uw = 0.5 * (u[0:nx - 1, :, :] + u[1:nx, :, :])
        Fe = ue * Ax[None, :, :]
        Fw = uw * Ax[None, :, :]
        v_at = 0.5 * (v[0:nx - 1, :, :] + v[1:nx, :, :])        # (nx-1,ny+1,nz)
        Fn = v_at[:, 1:ny + 1, :] * An
        Fs = v_at[:, 0:ny, :] * An
        w_at = 0.5 * (w[0:nx - 1, :, :] + w[1:nx, :, :])        # (nx-1,ny,nz+1)
        Ff = w_at[:, :, 1:nz + 1] * Az
        Fb = w_at[:, :, 0:nz] * Az

        # Effective viscosity (nu + nu_t). The x-faces of the u control volume
        # sit at cell centers i-1 and i, so use those directly; the transverse
        # (y-, z-) faces sit at the u-node, so use nu interpolated there.
        ne = self._nue
        ne_e = ne[1:nx, :, :]
        ne_w = ne[0:nx - 1, :, :]
        ne_u = 0.5 * (ne_e + ne_w)

        # Diffusion
        De = ne_e * Ax[None, :, :] / dx[1:nx][:, None, None]
        Dw = ne_w * Ax[None, :, :] / dx[0:nx - 1][:, None, None]
        Dn = ne_u * An / dyc[1:ny + 1][None, :, None]
        Ds = ne_u * An / dyc[0:ny][None, :, None]
        Df = ne_u * Az / dzc[1:nz + 1][None, None, :]
        Db = ne_u * Az / dzc[0:nz][None, None, :]

        AE = De + np.maximum(-Fe, 0.0)
        AW = Dw + np.maximum(Fw, 0.0)
        AN = Dn + np.maximum(-Fn, 0.0)
        AS = Ds + np.maximum(Fs, 0.0)
        AF = Df + np.maximum(-Ff, 0.0)
        AB = Db + np.maximum(Fb, 0.0)
        Vol = wx[:, None, None] * dy[None, :, None] * dz[None, None, :]
        AP0 = Vol / dt
        conv = Fe - Fw + Fn - Fs + Ff - Fb
        AP = AE + AW + AN + AS + AF + AB + AP0 + conv
        B = AP0 * u[1:nx, :, :]

        # Tangential BCs on the y- and z-boundaries (u is tangential there)
        self._apply_tangential_bcs("u", AP, B, (
            ("ylo", AS, (slice(None), 0, slice(None))),
            ("yhi", AN, (slice(None), ny - 1, slice(None))),
            ("zlo", AB, (slice(None), slice(None), 0)),
            ("zhi", AF, (slice(None), slice(None), nz - 1))))

        # Scatter interior into full-shape coefficient arrays
        return self._pack(u.shape, AP, AE, AW, AN, AS, AF, AB, B,
                          islice=slice(1, nx))

    def _assemble_v(self):
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        dt = self.cfg.dt
        u, v, w = self.u, self.v, self.w
        dx, dy, dz = self.dx, self.dy, self.dz
        dxc, dyc, dzc = self.dxc, self.dyc, self.dzc
        wyv = self.wyv

        wy = wyv[1:ny]                                          # (ny-1,) CV y-width
        Ay = np.multiply.outer(dx, dz)                         # (nx,nz) y-face area
        Ax = wy[None, :, None] * dz[None, None, :] * np.ones(nx)[:, None, None]
        Az = dx[:, None, None] * wy[None, :, None] * np.ones(nz)[None, None, :]

        u_at = 0.5 * (u[:, 0:ny - 1, :] + u[:, 1:ny, :])        # (nx+1,ny-1,nz)
        Fe = u_at[1:nx + 1, :, :] * Ax
        Fw = u_at[0:nx, :, :] * Ax
        vn = 0.5 * (v[:, 1:ny, :] + v[:, 2:ny + 1, :])
        vs = 0.5 * (v[:, 0:ny - 1, :] + v[:, 1:ny, :])
        Fn = vn * Ay[:, None, :]
        Fs = vs * Ay[:, None, :]
        w_at = 0.5 * (w[:, 0:ny - 1, :] + w[:, 1:ny, :])        # (nx,ny-1,nz+1)
        Ff = w_at[:, :, 1:nz + 1] * Az
        Fb = w_at[:, :, 0:nz] * Az

        # y-faces of the v control volume sit at cell centers j-1 and j; the
        # transverse (x-, z-) faces sit at the v-node.
        ne = self._nue
        ne_n = ne[:, 1:ny, :]
        ne_s = ne[:, 0:ny - 1, :]
        ne_v = 0.5 * (ne_n + ne_s)

        De = ne_v * Ax / dxc[1:nx + 1][:, None, None]
        Dw = ne_v * Ax / dxc[0:nx][:, None, None]
        Dn = ne_n * Ay[:, None, :] / dy[1:ny][None, :, None]
        Ds = ne_s * Ay[:, None, :] / dy[0:ny - 1][None, :, None]
        Df = ne_v * Az / dzc[1:nz + 1][None, None, :]
        Db = ne_v * Az / dzc[0:nz][None, None, :]

        AE = De + np.maximum(-Fe, 0.0)
        AW = Dw + np.maximum(Fw, 0.0)
        AN = Dn + np.maximum(-Fn, 0.0)
        AS = Ds + np.maximum(Fs, 0.0)
        AF = Df + np.maximum(-Ff, 0.0)
        AB = Db + np.maximum(Fb, 0.0)
        Vol = dx[:, None, None] * wy[None, :, None] * dz[None, None, :]
        AP0 = Vol / dt
        conv = Fe - Fw + Fn - Fs + Ff - Fb
        AP = AE + AW + AN + AS + AF + AB + AP0 + conv
        B = AP0 * v[:, 1:ny, :]

        self._apply_tangential_bcs("v", AP, B, (
            ("xlo", AW, (0, slice(None), slice(None))),
            ("xhi", AE, (nx - 1, slice(None), slice(None))),
            ("zlo", AB, (slice(None), slice(None), 0)),
            ("zhi", AF, (slice(None), slice(None), nz - 1))))

        return self._pack(v.shape, AP, AE, AW, AN, AS, AF, AB, B,
                          jslice=slice(1, ny))

    def _assemble_w(self, source=None):
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        dt = self.cfg.dt
        u, v, w = self.u, self.v, self.w
        dx, dy, dz = self.dx, self.dy, self.dz
        dxc, dyc, dzc = self.dxc, self.dyc, self.dzc
        wzw = self.wzw

        wz = wzw[1:nz]                                          # (nz-1,)
        Az = np.multiply.outer(dx, dy)                         # (nx,ny) z-face area
        Ax = wz[None, None, :] * dy[None, :, None] * np.ones(nx)[:, None, None]
        Ay = dx[:, None, None] * wz[None, None, :] * np.ones(ny)[None, :, None]

        u_at = 0.5 * (u[:, :, 0:nz - 1] + u[:, :, 1:nz])        # (nx+1,ny,nz-1)
        Fe = u_at[1:nx + 1, :, :] * Ax
        Fw = u_at[0:nx, :, :] * Ax
        v_at = 0.5 * (v[:, :, 0:nz - 1] + v[:, :, 1:nz])        # (nx,ny+1,nz-1)
        Fn = v_at[:, 1:ny + 1, :] * Ay
        Fs = v_at[:, 0:ny, :] * Ay
        wf = 0.5 * (w[:, :, 1:nz] + w[:, :, 2:nz + 1])
        wb = 0.5 * (w[:, :, 0:nz - 1] + w[:, :, 1:nz])
        Ff = wf * Az[:, :, None]
        Fb = wb * Az[:, :, None]

        # z-faces of the w control volume sit at cell centers k-1 and k; the
        # transverse (x-, y-) faces sit at the w-node.
        ne = self._nue
        ne_f = ne[:, :, 1:nz]
        ne_b = ne[:, :, 0:nz - 1]
        ne_w2 = 0.5 * (ne_f + ne_b)

        De = ne_w2 * Ax / dxc[1:nx + 1][:, None, None]
        Dw = ne_w2 * Ax / dxc[0:nx][:, None, None]
        Dn = ne_w2 * Ay / dyc[1:ny + 1][None, :, None]
        Ds = ne_w2 * Ay / dyc[0:ny][None, :, None]
        Df = ne_f * Az[:, :, None] / dz[1:nz][None, None, :]
        Db = ne_b * Az[:, :, None] / dz[0:nz - 1][None, None, :]

        AE = De + np.maximum(-Fe, 0.0)
        AW = Dw + np.maximum(Fw, 0.0)
        AN = Dn + np.maximum(-Fn, 0.0)
        AS = Ds + np.maximum(Fs, 0.0)
        AF = Df + np.maximum(-Ff, 0.0)
        AB = Db + np.maximum(Fb, 0.0)
        Vol = dx[:, None, None] * dy[None, :, None] * wz[None, None, :]
        AP0 = Vol / dt
        conv = Fe - Fw + Fn - Fs + Ff - Fb
        AP = AE + AW + AN + AS + AF + AB + AP0 + conv
        B = AP0 * w[:, :, 1:nz]
        if source is not None:                                  # buoyancy S*Vol
            B = B + source[:, :, 1:nz] * Vol

        self._apply_tangential_bcs("w", AP, B, (
            ("xlo", AW, (0, slice(None), slice(None))),
            ("xhi", AE, (nx - 1, slice(None), slice(None))),
            ("ylo", AS, (slice(None), 0, slice(None))),
            ("yhi", AN, (slice(None), ny - 1, slice(None)))))

        return self._pack(w.shape, AP, AE, AW, AN, AS, AF, AB, B,
                          kslice=slice(1, nz))

    @staticmethod
    def _pack(shape, AP, AE, AW, AN, AS, AF, AB, B,
              islice=None, jslice=None, kslice=None):
        """Scatter interior coefficient blocks into full-shape arrays; boundary
        (fixed) nodes get aP=1 so the Jacobi divide is safe."""
        sl = [slice(None), slice(None), slice(None)]
        if islice is not None:
            sl[0] = islice
        if jslice is not None:
            sl[1] = jslice
        if kslice is not None:
            sl[2] = kslice
        sl = tuple(sl)
        out = {n: np.zeros(shape) for n in ("aE", "aW", "aN", "aS", "aF", "aB", "b")}
        aP = np.ones(shape)
        aP[sl] = AP
        out["aE"][sl] = AE; out["aW"][sl] = AW
        out["aN"][sl] = AN; out["aS"][sl] = AS
        out["aF"][sl] = AF; out["aB"][sl] = AB
        out["b"][sl] = B
        return aP, out

    # ------------------------------------------------------------------
    # Energy: cell-centered advection-diffusion of temperature (M3)
    # ------------------------------------------------------------------
    def update_energy_diffusivity(self):
        """Effective thermal diffusivity alpha_eff = alpha + nu_t (isat_ffd's diff_T)."""
        self._alpha_eff = self.cfg.alpha + self.nu_t

    def _assemble_T(self):
        """Implicit first-order-upwind FV for the energy equation.

        T lives at cell centers, so the convecting face velocities are the MAC
        face velocities directly (no interpolation).  Domain walls, the inlet
        slot and box surfaces enter as half-cell Dirichlet faces (added to the
        diagonal and RHS); outlets and adiabatic faces are zero-gradient
        (Neumann) and contribute nothing.  Coefficients follow the same
        Patankar form as the momentum assembly.
        """
        g = self.g
        nx, ny, nz = g.nx, g.ny, g.nz
        dt = self.cfg.dt
        u, v, w = self.u, self.v, self.w
        dx, dy, dz = self.dx, self.dy, self.dz
        dxc, dyc, dzc = self.dxc, self.dyc, self.dzc
        ae = self._alpha_eff
        S = self.solid

        Ax = np.multiply.outer(dy, dz)           # (ny,nz) x-face area
        Ay = np.multiply.outer(dx, dz)           # (nx,nz) y-face area
        Az = np.multiply.outer(dx, dy)           # (nx,ny) z-face area
        Vol = g.cell_volumes()

        # Convective volume fluxes on all six faces of every cell.
        Fe = u[1:, :, :] * Ax[None, :, :]
        Fw = u[:-1, :, :] * Ax[None, :, :]
        Fn = v[:, 1:, :] * Ay[:, None, :]
        Fs = v[:, :-1, :] * Ay[:, None, :]
        Ff = w[:, :, 1:] * Az[:, :, None]
        Fb = w[:, :, :-1] * Az[:, :, None]

        # Interior (fluid-fluid) neighbour coefficients: diffusion + upwind.
        aE = np.zeros((nx, ny, nz)); aW = np.zeros_like(aE)
        aN = np.zeros_like(aE); aS = np.zeros_like(aE)
        aF = np.zeros_like(aE); aB = np.zeros_like(aE)

        aexf = 0.5 * (ae[:-1, :, :] + ae[1:, :, :])
        Dx = aexf * Ax[None, :, :] / dxc[1:nx][:, None, None]
        aE[:-1, :, :] = Dx + np.maximum(-Fe[:-1, :, :], 0.0)
        aW[1:, :, :] = Dx + np.maximum(Fw[1:, :, :], 0.0)

        aeyf = 0.5 * (ae[:, :-1, :] + ae[:, 1:, :])
        Dy = aeyf * Ay[:, None, :] / dyc[1:ny][None, :, None]
        aN[:, :-1, :] = Dy + np.maximum(-Fn[:, :-1, :], 0.0)
        aS[:, 1:, :] = Dy + np.maximum(Fs[:, 1:, :], 0.0)

        aezf = 0.5 * (ae[:, :, :-1] + ae[:, :, 1:])
        Dz = aezf * Az[:, :, None] / dzc[1:nz][None, None, :]
        aF[:, :, :-1] = Dz + np.maximum(-Ff[:, :, :-1], 0.0)
        aB[:, :, 1:] = Dz + np.maximum(Fb[:, :, 1:], 0.0)

        # Dirichlet contributions accumulate into the diagonal (wall_aP) and the
        # RHS (wall_b): coefficient = half-cell diffusion + convective inflow.
        wall_aP = np.zeros((nx, ny, nz))
        wall_b = np.zeros((nx, ny, nz))

        def add_domain(face, sl, area, half, flux_in):
            is_dir, Tval = self._Tbc[face]
            D = ae[sl] * area / half
            a_bc = np.where(is_dir, D + np.maximum(flux_in, 0.0), 0.0)
            wall_aP[sl] += a_bc
            wall_b[sl] += a_bc * Tval

        # Domain boundary layers (half distances are exactly dxc[0], dxc[nx], ...).
        add_domain("xlo", (0, slice(None), slice(None)), Ax, dxc[0], Fw[0, :, :])
        add_domain("xhi", (nx - 1, slice(None), slice(None)), Ax, dxc[nx], -Fe[nx - 1, :, :])
        add_domain("ylo", (slice(None), 0, slice(None)), Ay, dyc[0], Fs[:, 0, :])
        add_domain("yhi", (slice(None), ny - 1, slice(None)), Ay, dyc[ny], -Fn[:, ny - 1, :])
        add_domain("zlo", (slice(None), slice(None), 0), Az, dzc[0], Fb[:, :, 0])
        add_domain("zhi", (slice(None), slice(None), nz - 1), Az, dzc[nz], -Ff[:, :, nz - 1])

        # Internal solid interfaces: a fluid cell facing a solid neighbour gets
        # a half-cell Dirichlet at the solid's surface temperature. `flux_in`
        # adds the convective-inflow term (identical to add_domain's) for a
        # rack's exhaust face, where the face velocity is nonzero; for every
        # other (non-rack) solid it is 0, so this is unchanged from before.
        def add_solid(nbr_solid, Ts_nbr, coeff, area, half, flux_in):
            fluid_face = nbr_solid & ~S
            coeff[fluid_face] = 0.0                       # drop fluid-fluid guess
            dir_face = fluid_face & ~np.isnan(Ts_nbr)
            D = ae * area / half
            a_bc = D + np.maximum(flux_in, 0.0)
            wall_aP[dir_face] += a_bc[dir_face]
            wall_b[dir_face] += a_bc[dir_face] * Ts_nbr[dir_face]

        ST = self.solid_temp
        half_x = 0.5 * dx[:, None, None] * np.ones((nx, ny, nz))
        half_y = 0.5 * dy[None, :, None] * np.ones((nx, ny, nz))
        half_z = 0.5 * dz[None, None, :] * np.ones((nx, ny, nz))
        Axb = np.broadcast_to(Ax[None, :, :], (nx, ny, nz))
        Ayb = np.broadcast_to(Ay[:, None, :], (nx, ny, nz))
        Azb = np.broadcast_to(Az[:, :, None], (nx, ny, nz))

        def shift_solid(axis, up):
            """Solid-neighbour mask and its surface temp, viewed from each cell."""
            nbr = np.zeros((nx, ny, nz), dtype=np.bool_)
            Ts = np.full((nx, ny, nz), np.nan)
            src = slice(1, None) if up else slice(None, -1)
            dst = slice(None, -1) if up else slice(1, None)
            sl_dst = [slice(None)] * 3; sl_dst[axis] = dst
            sl_src = [slice(None)] * 3; sl_src[axis] = src
            nbr[tuple(sl_dst)] = S[tuple(sl_src)]
            Ts[tuple(sl_dst)] = ST[tuple(sl_src)]
            return nbr, Ts

        nbrE, TsE = shift_solid(0, up=True);  add_solid(nbrE, TsE, aE, Axb, half_x, -Fe)
        nbrW, TsW = shift_solid(0, up=False); add_solid(nbrW, TsW, aW, Axb, half_x, Fw)
        nbrN, TsN = shift_solid(1, up=True);  add_solid(nbrN, TsN, aN, Ayb, half_y, -Fn)
        nbrS, TsS = shift_solid(1, up=False); add_solid(nbrS, TsS, aS, Ayb, half_y, Fs)
        nbrF, TsF = shift_solid(2, up=True);  add_solid(nbrF, TsF, aF, Azb, half_z, -Ff)
        nbrB, TsB = shift_solid(2, up=False); add_solid(nbrB, TsB, aB, Azb, half_z, Fb)

        AP0 = Vol / dt
        conv = Fe - Fw + Fn - Fs + Ff - Fb
        aP = aE + aW + aN + aS + aF + aB + AP0 + conv + wall_aP
        b = AP0 * self.T + wall_b
        return aP, {"aE": aE, "aW": aW, "aN": aN, "aS": aS,
                    "aF": aF, "aB": aB, "b": b}

    def solve_energy_step(self):
        c = self._assemble_T()
        aP, co = c
        jacobi(self.T, aP, co["aE"], co["aW"], co["aN"], co["aS"], co["aF"],
               co["aB"], co["b"], self.solid, self.cfg.n_energy_sweeps)

    def buoyancy_source(self):
        """Boussinesq body force (kinematic) on the w-momentum control volumes:
        F_z = g * beta * (T - T_ref), interpolated to the interior z-faces."""
        nx, ny, nz = self.g.nx, self.g.ny, self.g.nz
        src = np.zeros((nx, ny, nz + 1))
        Tf = 0.5 * (self.T[:, :, :-1] + self.T[:, :, 1:])
        src[:, :, 1:nz] = self.cfg.g * self.cfg.beta * (Tf - self.cfg.T_ref)
        src[self._w_solid] = 0.0
        return src

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------
    def momentum_predict(self, source=None):
        ns = self.cfg.n_mom_sweeps
        aP, c = self._assemble_u()
        jacobi(self.u, aP, c["aE"], c["aW"], c["aN"], c["aS"], c["aF"], c["aB"],
               c["b"], self.u_fixed, ns)
        aP, c = self._assemble_v()
        jacobi(self.v, aP, c["aE"], c["aW"], c["aN"], c["aS"], c["aF"], c["aB"],
               c["b"], self.v_fixed, ns)
        aP, c = self._assemble_w(source=source)
        jacobi(self.w, aP, c["aE"], c["aW"], c["aN"], c["aS"], c["aF"], c["aB"],
               c["b"], self.w_fixed, ns)
        self.apply_velocity_bcs()

    def residuals(self):
        """Max-norm of the discrete linear-system residual ||b - A*psi||_inf
        for each implicitly solved field, assembled from the CURRENT fields and
        normalized by max|aP*psi| over the solved cells (dimensionless).

        This is the distance from the true discrete solution, and is distinct
        from the per-step field change reported by `run()`: `d_u/step` says the
        march has stopped moving, this says whether the equations are actually
        satisfied. At a genuine steady fixed point every value -> 0
        *independent of dt*, because the AP0 = Vol/dt inertia term (folded into
        both aP and b via b = AP0*psi_old) cancels exactly when psi == psi_old.
        A residual that grows with dt, or that only a large sweep count drives
        down, is direct evidence the reported steady state is the fixed point
        of the under-converged per-step operator rather than of the discrete
        equations. Uses the frozen turbulence/diffusivity state from the last
        step (does not mutate anything)."""
        def _one(field, fixed, assembled):
            aP, c = assembled
            r = linf_residual(field, aP, c["aE"], c["aW"], c["aN"], c["aS"],
                              c["aF"], c["aB"], c["b"], fixed)
            scale = np.abs(aP * field)[~fixed].max()
            return float(r / scale) if scale > 0 else float(r)

        out = {"u": _one(self.u, self.u_fixed, self._assemble_u()),
               "v": _one(self.v, self.v_fixed, self._assemble_v())}
        src = None
        if self.cfg.solve_energy:
            out["T"] = _one(self.T, self.solid, self._assemble_T())
            src = self.buoyancy_source()
        if self.w_source_extra is not None:
            src = self.w_source_extra if src is None else src + self.w_source_extra
        out["w"] = _one(self.w, self.w_fixed, self._assemble_w(source=src))
        return out

    def step(self):
        self.update_turbulence()
        src = None
        if self.cfg.solve_energy:
            self.update_energy_diffusivity()
            if self._racks:
                self._update_rack_exhaust()
            self.solve_energy_step()
            src = self.buoyancy_source()
        if self.w_source_extra is not None:
            src = self.w_source_extra if src is None else src + self.w_source_extra
        self.momentum_predict(source=src)
        self.project()

    def run(self, max_steps=20000, tol=1e-4, check_every=50, verbose=True):
        """March to steady state. Convergence = max |field change|/ref per step."""
        ref = max(1e-9, self._velocity_scale())
        u_prev = self.u.copy()
        history = []
        for n in range(1, max_steps + 1):
            self.step()
            if n % check_every == 0:
                d = np.abs(self.u - u_prev).max() / ref / check_every
                divmax = np.abs(self.divergence()).max()
                history.append((n, d, divmax))
                if verbose:
                    print(f"  step {n:6d}  d_u/step={d:.2e}  max|div|={divmax:.2e}")
                if d < tol:
                    if verbose:
                        print(f"  converged at step {n}")
                    break
                u_prev = self.u.copy()
        return history

    def _velocity_scale(self):
        return max(np.abs(self.u).max(), np.abs(self.v).max(),
                   np.abs(self.w).max())

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def velocity_at_centers(self):
        """Interpolate face velocities to cell centers -> (Uc, Vc, Wc)."""
        Uc = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        Vc = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        Wc = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return Uc, Vc, Wc
