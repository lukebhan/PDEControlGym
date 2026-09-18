"""White-space (above-floor room) model: tiles as prescribed-velocity inlets,
ceiling as a fixed-flow outlet, racks as flow-through solids with a Han
Eq. 12 exhaust, and the Eq. 10 tile body force.

Unlike the plenum, the white space does not iterate its own boundary
condition: `tile_flows` (Han's coupling variable, normally the converged
`PlenumResult.Q_m3s` from `plenum.run_plenum`) is a fixed input, so this
module only has to build the case and march it to steady state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from ..ffd_upwind import Boundary, Config, Solver

from .mesh import solids_from_layout, tile_cell_index, tile_cell_mask, whitespace_grid
from .metrics import rack_inlet_csv
from .tiles import body_force


@dataclass
class WhitespaceResult:
    layout_name: str
    profile: str
    steps: int
    converged: bool
    du_per_step: float
    dT_per_step: float
    wall_time_s: float
    npz_path: str
    rack_csv_path: str


def build_whitespace(layout, cells_per_tile, tile_flows, T_sup_C, Q_sup_m3s, powers_kW):
    """Grid + Solver for the white space above a layout with given tile flows.

    `tile_flows` and `powers_kW` are dicts keyed the same way `plenum.py` and
    `mesh.solids_from_layout` key their inputs: `tile_flows[i]` is the flow
    [m^3/s] through `layout.tiles[i]` (i.e. indexed like `tile_cell_index`'s
    output, since `Tile` carries no separate id), `powers_kW[rack.id]` is a
    rack's IT power [kW] (see `racks.assign_powers`).
    """
    grid = whitespace_grid(layout, cells_per_tile)
    tile_index = tile_cell_index(grid, layout)
    tile_mask = tile_index >= 0
    ceiling_mask = tile_cell_mask(grid, layout, layout.ceiling_tiles)

    A_tile = layout.tile_m**2
    n_ceil = len(layout.ceiling_tiles)
    W_out = Q_sup_m3s / (n_ceil * A_tile)

    W_in = np.zeros((grid.nx, grid.ny))
    T_in = np.full((grid.nx, grid.ny), T_sup_C)
    for i, tile in enumerate(layout.tiles):
        W_in[tile_index == i] = tile_flows[i] / A_tile

    bcs = {
        "zlo": Boundary("inlet", mask=tile_mask, vel_map=W_in, temp_map=T_in),
        "zhi": Boundary(
            "inlet", mask=ceiling_mask, vel_map=np.full((grid.nx, grid.ny), W_out)
        ),
    }

    solids = solids_from_layout(layout, grid, powers_kW=powers_kW)

    cfg = Config(
        dt=0.2,
        bcs=bcs,
        turb_model="chen",
        solve_energy=True,
        T_ref=T_sup_C,
        T_init=T_sup_C,
        beta=1.0 / 295.15,
        outlet_mode="pressure",
        pressure_solver="amg",
        solids=solids,
    )
    solver = Solver(grid, cfg)

    h = grid.z.dc[1]
    src = np.zeros((grid.nx, grid.ny, grid.nz + 1))
    for i, tile in enumerate(layout.tiles):
        F = body_force(tile_flows[i], A_tile, h, tile.open_area)
        src[:, :, 1][tile_index == i] = F
    solver.w_source_extra = src

    return grid, solver


def run_whitespace(
    layout,
    cells_per_tile,
    tile_flows,
    T_sup_C,
    Q_sup_m3s,
    powers_kW,
    profile=None,
    tol=5e-5,
    check_every=100,
    max_steps=2000,
):
    """March the white space to steady state; report CONVERGED/NOT CONVERGED."""
    if profile is None:
        profile = f"cpt{cells_per_tile}"
    grid, solver = build_whitespace(
        layout, cells_per_tile, tile_flows, T_sup_C, Q_sup_m3s, powers_kW
    )

    A_tile = layout.tile_m**2
    U_ref = max(1e-9, max(tile_flows[i] for i in range(len(layout.tiles))) / A_tile)
    dT_ref = 14.0  # K, Han's typical rack rise -- fixed reference, not derived per-run

    u_prev = solver.u.copy()
    T_prev = solver.T.copy()
    du = dT = float("nan")
    converged = False
    step = 0
    t0 = time.perf_counter()
    for step in range(1, max_steps + 1):
        solver.step()
        if step % check_every == 0:
            du = np.abs(solver.u - u_prev).max() / U_ref / check_every
            dT = np.abs(solver.T - T_prev).max() / dT_ref / check_every
            if du < tol and dT < tol:
                converged = True
                break
            u_prev = solver.u.copy()
            T_prev = solver.T.copy()
    wall_time_s = time.perf_counter() - t0

    status = "CONVERGED" if converged else "NOT CONVERGED"
    print(
        f"whitespace {layout.name}: {status} at step {step} "
        f"(t={step * solver.cfg.dt:.1f}s), du/step={du:.2e}, dT/step={dT:.2e} "
        f"({wall_time_s:.1f}s)"
    )

    npz_path = _save_npz(layout, profile, Q_sup_m3s, T_sup_C, grid, solver)
    rack_csv_path = rack_inlet_csv(
        solver, grid, layout, _rack_csv_path(layout, profile, Q_sup_m3s, T_sup_C)
    )

    return WhitespaceResult(
        layout_name=layout.name,
        profile=profile,
        steps=step,
        converged=converged,
        du_per_step=float(du),
        dT_per_step=float(dT),
        wall_time_s=wall_time_s,
        npz_path=npz_path,
        rack_csv_path=rack_csv_path,
    )


def _result_stem(layout, profile, Q_sup_m3s, T_sup_C):
    return f"ws_{layout.name}_{profile}_q{round(Q_sup_m3s*3600):d}_t{round(T_sup_C):d}"


def _save_npz(layout, profile, Q_sup_m3s, T_sup_C, grid, solver):
    out_dir = os.path.join("results", "dc")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, _result_stem(layout, profile, Q_sup_m3s, T_sup_C) + ".npz"
    )
    Uc, Vc, Wc = solver.velocity_at_centers()
    np.savez(
        path,
        xc=grid.x.c,
        yc=grid.y.c,
        zc=grid.z.c,
        Uc=Uc,
        Vc=Vc,
        Wc=Wc,
        T=solver.T,
        solid=solver.solid,
    )
    return path


def _rack_csv_path(layout, profile, Q_sup_m3s, T_sup_C):
    out_dir = os.path.join("results", "dc")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(
        out_dir, _result_stem(layout, profile, Q_sup_m3s, T_sup_C) + "_racks.csv"
    )
