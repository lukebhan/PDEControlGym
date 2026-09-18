"""Rack-inlet sampling, PRD_*, RCI_*, T_max_in, energy closure (Han et al. 2021
Eqs. 14-16, 18-22).

Pure post-processing: every function here takes a solved `Solver`/`grid`/
`layout` (or, for the RCI/PRD/count functions, plain arrays) and returns
numbers -- no solving happens in this module.
"""
from __future__ import annotations

import csv

import numpy as np

from .mesh import _cells_per_tile, _tile_offset_cells, _RACK_AXIS_FRONT, tile_cell_mask
from .units import CP, RHO, m3h_to_m3s

RCI_HEIGHTS_M = (0.53, 0.91, 1.30, 1.68)  # Han Eq. 18-22: 4 sampling heights


def _rack_front_cell_ij(grid, layout, rack):
    """(ix, iy) of the fluid cell adjacent to `rack`'s front (inlet) face, at
    the rack's spanwise-centre cell. Racks are always 1x1 tile (`mesh.
    solids_from_layout`'s only supported footprint)."""
    cpt = _cells_per_tile(grid, layout)
    off_x, off_y = _tile_offset_cells(grid, layout, cpt)
    axis, front = _RACK_AXIS_FRONT[rack.facing]
    x0 = off_x + rack.tile_ix * cpt
    y0 = off_y + rack.tile_iy * cpt
    center = cpt // 2
    if axis == "x":
        iy = y0 + center
        ix = x0 + cpt if front == "hi" else x0 - 1
    else:
        ix = x0 + center
        iy = y0 + cpt if front == "hi" else y0 - 1
    return ix, iy


def rack_inlet_profile(solver, grid, layout, rack, heights=RCI_HEIGHTS_M):
    """T [C] at each of `heights` [m] in the fluid column directly in front of
    `rack`'s front face, linearly interpolated in z between cell centres."""
    ix, iy = _rack_front_cell_ij(grid, layout, rack)
    T_col = solver.T[ix, iy, :]
    zc = grid.z.c
    return tuple(float(np.interp(h, zc, T_col)) for h in heights)


def rack_inlet_temperature(solver, grid, layout, rack, heights=RCI_HEIGHTS_M):
    """Mean of the 4-height inlet profile -- Han's per-rack "rack inlet temperature"."""
    return float(np.mean(rack_inlet_profile(solver, grid, layout, rack, heights)))


def rack_inlet_csv(solver, grid, layout, out_path, heights=RCI_HEIGHTS_M):
    """Write one row per rack (`id,row,number,empty,T_<h>...,T_inlet_mean`) --
    the "rack-inlet CSV" `whitespace.run_whitespace` saves alongside its npz."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "row", "number", "empty"]
                   + [f"T_{h:.2f}" for h in heights] + ["T_inlet_mean"])
        for r in layout.racks:
            profile = rack_inlet_profile(solver, grid, layout, r, heights)
            w.writerow([r.id, r.row, r.number, int(r.empty)]
                       + [f"{t:.4f}" for t in profile]
                       + [f"{np.mean(profile):.4f}"])
    return out_path


def rci_hi(T_racks, T_max_rec=27.0, T_max_all=35.0):
    """Eq. 18/19: fraction of the (T_max_rec, T_max_all] headroom not used up
    by racks running above the recommended max; 1.0 = every rack <= T_max_rec."""
    T = np.asarray(T_racks, dtype=float)
    excess = np.clip(T - T_max_rec, 0.0, None)
    return 1.0 - excess.sum() / (T.size * (T_max_all - T_max_rec))


def rci_lo(T_racks, T_min_rec=18.0, T_min_all=15.0):
    """Eq. 20/21: same idea on the cold side; 1.0 = every rack >= T_min_rec."""
    T = np.asarray(T_racks, dtype=float)
    shortfall = np.clip(T_min_rec - T, 0.0, None)
    return 1.0 - shortfall.sum() / (T.size * (T_min_rec - T_min_all))


def t_max_in(T_racks):
    """Eq. 22: the hottest rack-inlet temperature."""
    return float(np.max(T_racks))


def prd_from_mean(Q_tiles):
    """Eq. 16: per-tile percent deviation from the mean flow, plus max/min/std."""
    Q = np.asarray(Q_tiles, dtype=float)
    Q_mean = Q.mean()
    prd_pct = 100.0 * (Q - Q_mean) / Q_mean
    return prd_pct, float(prd_pct.max()), float(prd_pct.min()), float(prd_pct.std())


def prd_Q(sim, exp):
    """Eq. 14: percent deviation of a simulated flow from a reference (e.g.
    measured or digitized) flow, normalized by the reference value itself."""
    sim, exp = np.asarray(sim, dtype=float), np.asarray(exp, dtype=float)
    return 100.0 * (sim - exp) / exp


def prd_T(sim, exp, dT=14.0):
    """Eq. 15: percent deviation of a simulated temperature from a reference,
    normalized by the characteristic rack temperature rise `dT` (Han: 14 C) --
    not by the reference value itself, since T is measured on an offset (C)
    scale where that would blow up or be ill-defined near 0."""
    sim, exp = np.asarray(sim, dtype=float), np.asarray(exp, dtype=float)
    return 100.0 * (sim - exp) / dT


#: Fig 15 left panel (hot side, overheating risk): 1 C bins from Han's
#: T_max_rec (27 C) to the Class-A1 T_max_in allowable limit (32 C -- distinct
#: from rci_hi's own T_max_all=35 C default, which is the general ASHRAE
#: allowable used *inside* the RCI_HI formula, Eqs. 18-19; the paper uses the
#: narrower, class-specific 32 C only for this histogram). Paper plots this
#: against AR in {0.5, 0.75, 1.0}, at T_sup=16 C: <27 catch-all, 27-28, 28-29,
#: 29-30, 30-31, 31-32, >32 catch-all -- 7 bins, columns n_hot_0..n_hot_6.
HOT_BIN_EDGES = (-np.inf, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, np.inf)


def cold_bin_edges(t_sup_C):
    """Fig 15 right panel (cold side, overcooling risk): 0.5 C bins from
    `t_sup_C` to `t_sup_C + 2` (Han's recommended min is 18 C; at T_sup=16 C
    this gives 16-16.5, 16.5-17, 17-17.5, 17.5-18, >18 -- 5 bins, columns
    n_cold_0..n_cold_4). No bin below `t_sup_C`: a rack inlet can never be
    colder than the supply air (`whitespace.py`'s own invariant), so Han's
    chart has no such category either -- unlike the hot side there is no
    catch-all bin at the cold end."""
    return (t_sup_C, t_sup_C + 0.5, t_sup_C + 1.0, t_sup_C + 1.5, t_sup_C + 2.0, np.inf)


def rack_count_by_range(T_racks, edges):
    """Count of racks whose inlet temperature falls in each bin of `edges`
    (Fig 15): half-open `[edges[i], edges[i+1])`, except the last bin, which is
    closed on both ends. Pass `-np.inf`/`np.inf` as the outer edges for
    open-ended bins (e.g. "> 32 C"). See `HOT_BIN_EDGES`/`cold_bin_edges` for
    the two bin schemes Han's own Fig 15 actually uses -- they are different
    from each other, not one shared scheme."""
    T = np.asarray(T_racks, dtype=float)
    edges = np.asarray(edges, dtype=float)
    n_bins = edges.size - 1
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            counts[i] = int(np.sum((T >= lo) & (T <= hi)))
        else:
            counts[i] = int(np.sum((T >= lo) & (T < hi)))
    return counts


def energy_closure(solver, grid, layout, powers_kW):
    """Steady-state energy balance: ceiling-outflow enthalpy above the supply
    baseline vs. total rack IT power. Returns `(closure_W, total_P_W, ratio)`;
    `ratio` == 1.0 is exact closure (same definition `test_dc_whitespace.py`
    checks inline)."""
    A_cell = grid.x.d[0] * grid.y.d[0]
    ceiling_mask = tile_cell_mask(grid, layout, layout.ceiling_tiles)
    Q_sup = m3h_to_m3s(layout.supply_flow_m3h)

    T_ceil = solver.T[:, :, -1]
    W_ceil = solver.w[:, :, -1]
    closure_W = RHO * CP * float((W_ceil[ceiling_mask] * A_cell * T_ceil[ceiling_mask]).sum())
    closure_W -= RHO * CP * Q_sup * layout.supply_T_C

    total_P_W = sum(powers_kW.values()) * 1000.0
    return closure_W, total_P_W, closure_W / total_P_W
