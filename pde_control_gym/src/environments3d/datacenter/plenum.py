"""Underfloor plenum model + VanGilder pressure-shift tile-flow solve
(Han et al. 2021 Sec. 2.2.2.3).

The plenum is its own one-way-coupled domain (see the plan's design-question
answer): AHU supply enters through the short-side walls (`layout.plenum_inlets`)
and the only outflow is the perforated floor tiles, modelled here as a
Dirichlet *velocity* opening on `zhi` (kind='inlet' is reused purely for its
per-cell `vel_map` mechanism -- physically this is the plenum's outflow, with
w > 0 meaning upward/out). Each outer iteration prescribes a per-tile approach
velocity V_i, advances the flow one time step, reads back the tile-top
pressure, and shifts that pressure by a constant `c` (found with `brentq`) so
that the implied tile flows (Han Eq. 8) sum to the supply flow; `c` absorbs
the arbitrary pressure-pin constant, so where the pin cell sits is irrelevant.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..ffd_upwind import Boundary, Config, Solver

from .mesh import plenum_grid, tile_cell_index
from .metrics import prd_from_mean
from .tiles import loss_coefficient

_INLET_AXIS = {
    "xlo": (0, +1.0),
    "xhi": (0, -1.0),
    "ylo": (1, +1.0),
    "yhi": (1, -1.0),
}


@dataclass
class PlenumResult:
    layout_name: str
    tile_ix: np.ndarray  # (n_tiles,)
    tile_iy: np.ndarray  # (n_tiles,)
    Q_m3s: np.ndarray  # (n_tiles,) per-tile flow
    V_ms: np.ndarray  # (n_tiles,) per-tile approach velocity
    p_kin: np.ndarray  # (n_tiles,) per-tile shifted kinematic pressure
    prd_m_pct: np.ndarray  # (n_tiles,) Eq. 16 percent deviation from mean flow
    prd_max: float
    prd_min: float
    prd_std: float
    steps: int
    converged: bool
    wall_time_s: float


def build_plenum(layout, cells_per_tile, depth_m, beta, Q_sup_m3s):
    """Grid + Solver for the plenum, plus the tile-index map used by run_plenum.

    Returns (grid, solver, tile_index). `solver.bc['zhi'].vel_map` is the
    per-cell tile-outflow array `run_plenum` mutates every outer iteration.
    """
    grid = plenum_grid(layout, cells_per_tile, depth_m)
    tile_index = tile_cell_index(grid, layout)

    bcs = {}
    for pi in layout.plenum_inlets:
        axis, sign = _INLET_AXIS[pi.face]
        if axis == 0:
            area = grid.y.length * grid.z.length
        else:
            area = grid.x.length * grid.z.length
        U = sign * pi.fraction * Q_sup_m3s / area
        vel = (U, 0.0, 0.0) if axis == 0 else (0.0, U, 0.0)
        bcs[pi.face] = Boundary("inlet", vel=vel)

    W_tiles = np.zeros((grid.nx, grid.ny))
    bcs["zhi"] = Boundary("inlet", mask=(tile_index >= 0), vel_map=W_tiles)

    cfg = Config(
        dt=1.0,
        bcs=bcs,
        turb_model="chen",
        solve_energy=False,
        outlet_mode="pressure",
        pressure_solver="amg",
    )
    solver = Solver(grid, cfg)
    return grid, solver, tile_index


def _solve_pressure_shift(p_i, A_tile, f_loss, Q_sup_m3s):
    """brentq root of sum_i A_tile*sqrt(2*max(p_i+c,0)/f_loss) - Q_sup == 0."""

    def g(c):
        V = np.sqrt(2.0 * np.clip(p_i + c, 0.0, None) / f_loss)
        return A_tile * V.sum() - Q_sup_m3s

    c_lo, c_hi = -1.0, 1.0
    while g(c_lo) > 0.0:
        c_lo *= 2.0
    while g(c_hi) < 0.0:
        c_hi *= 2.0
    return brentq(g, c_lo, c_hi, xtol=1e-12)


def run_plenum(
    layout,
    cells_per_tile,
    depth_m,
    beta,
    Q_sup_m3s,
    omega=0.1,
    tol=1e-4,
    max_steps=2000,
    min_steps=20,
):
    """VanGilder pressure-shift iteration to steady, mass-balanced tile flows.

    `omega=0.5` (this function's value through D4/D7) is unstable at the
    shallow-plenum/high-open-area corner of the D8 sweep (e.g. 305 mm/56 %):
    tracing `V`/`rel` step by step there shows an exact period-2 limit cycle,
    not slow convergence -- about 3 in 5 tiles pinned at V=0 every other step
    (PRD swinging to +-100%), forever. A per-step oscillation detector on
    `rel` doesn't see it (`rel`'s own magnitude is nearly constant across the
    cycle even though `V` itself alternates between two states), so the fix
    is a smaller fixed relaxation factor rather than an adaptive one.
    omega=0.15 looked like it fixed that corner (a plausible-looking
    max=29.5% at 1500 steps, close to Han's ~28%) but that was a moving
    snapshot, not a fixed point -- it still hadn't met the 3-consecutive-tol
    criterion. omega=0.1 (`max_steps` raised to 2000 to give it room) reaches
    a genuine fixed point there (max=18.28%, min=-18.64%, std=8.89%, no
    tiles pinned at zero) in ~110-230 steps depending on depth/beta, and
    costs only about 2-3x the steps of the old omega=0.5 on the well-behaved
    corners (e.g. 914 mm/25 %, Han's baseline point). See D8 notes for the
    full step-by-step trace.
    """
    grid, solver, tile_index = build_plenum(
        layout, cells_per_tile, depth_m, beta, Q_sup_m3s
    )

    n_tiles = len(layout.tiles)
    A_tile = layout.tile_m**2
    f_loss = loss_coefficient(beta)
    tile_mask = tile_index >= 0
    flat_index = tile_index[tile_mask]

    V = np.full(n_tiles, Q_sup_m3s / (n_tiles * A_tile))
    vel_map = solver.bc["zhi"].vel_map

    t0 = time.perf_counter()
    n_ok = 0
    converged = False
    step = 0
    for step in range(1, max_steps + 1):
        vel_map[tile_mask] = V[flat_index]
        solver.step()

        p_top = solver.p[:, :, -1]
        sums = np.bincount(flat_index, weights=p_top[tile_mask], minlength=n_tiles)
        counts = np.bincount(flat_index, minlength=n_tiles)
        p_i = sums / counts

        c = _solve_pressure_shift(p_i, A_tile, f_loss, Q_sup_m3s)
        V_new = np.sqrt(2.0 * np.clip(p_i + c, 0.0, None) / f_loss)

        rel = np.abs(V_new - V).max() / max(1e-12, V.mean())
        V = (1.0 - omega) * V + omega * V_new

        n_ok = n_ok + 1 if rel < tol else 0
        if n_ok >= 3 and step >= min_steps:
            converged = True
            break
    wall_time_s = time.perf_counter() - t0

    p_final = p_i + c
    Q_i = V * A_tile
    prd_m, prd_max, prd_min, prd_std = prd_from_mean(Q_i)

    tile_ix = np.array([t.tile_ix for t in layout.tiles])
    tile_iy = np.array([t.tile_iy for t in layout.tiles])

    result = PlenumResult(
        layout_name=layout.name,
        tile_ix=tile_ix,
        tile_iy=tile_iy,
        Q_m3s=Q_i,
        V_ms=V,
        p_kin=p_final,
        prd_m_pct=prd_m,
        prd_max=prd_max,
        prd_min=prd_min,
        prd_std=prd_std,
        steps=step,
        converged=converged,
        wall_time_s=wall_time_s,
    )
    _save_csv(result, depth_m, beta, Q_sup_m3s)

    status = "CONVERGED" if converged else "NOT CONVERGED"
    print(
        f"plenum {layout.name} depth={depth_m*1000:.0f}mm beta={beta:.2f} "
        f"Q_sup={Q_sup_m3s*3600:.0f} m3/h: {status} in {step} steps "
        f"({wall_time_s:.1f}s)  PRD_m max={result.prd_max:+.2f}% "
        f"min={result.prd_min:+.2f}% std={result.prd_std:.2f}%"
    )
    return result


def _save_csv(result, depth_m, beta, Q_sup_m3s):
    out_dir = os.path.join("results", "dc")
    os.makedirs(out_dir, exist_ok=True)
    fname = (
        f"plenum_{result.layout_name}_d{round(depth_m*1000):d}"
        f"_b{round(beta*100):d}_q{round(Q_sup_m3s*3600):d}.csv"
    )
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tile_ix", "tile_iy", "Q_m3s", "V_ms", "p_kin", "prd_m_pct"])
        for i in range(len(result.tile_ix)):
            w.writerow(
                [
                    result.tile_ix[i],
                    result.tile_iy[i],
                    f"{result.Q_m3s[i]:.6f}",
                    f"{result.V_ms[i]:.6f}",
                    f"{result.p_kin[i]:.6f}",
                    f"{result.prd_m_pct[i]:.4f}",
                ]
            )
    return path
