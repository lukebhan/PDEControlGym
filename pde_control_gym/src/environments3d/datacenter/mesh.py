"""Tile-aligned structured grid builder for data-center cases (plan milestone D3).

White space and plenum share the same in-plane (x, y) scheme: uniform cells,
`cells_per_tile` per tile edge, so every tile boundary, rack footprint and
opening (perforated/ceiling tile) lands exactly on a cell face -- no `mask_fn`
geometry search is needed, just integer tile-index arithmetic (`_cells_per_tile`
/ `_tile_offset_cells`).

The white-space z-axis is uniform at `dz = tile_m / cells_per_tile` from 0 to
`height_m`, with extra faces forced in at every distinct solid top height
(both rack heights in `layout.rack_height_m`, and the PDU height) so those
tops always land on a face too, regardless of whether a solid of that height
is actually present in a given layout (mirrors Case 2's box-faces-always-cut
convention in `ffd_upwind/room_grid.py`). A generated face closer than
`0.25 * dz` to an inserted one is dropped, per the plan.

The plenum z-axis is always `PLENUM_NZ` (6) uniform layers, independent of
depth and of the white-space `cells_per_tile` profile: Han meshed the plenum
with a fixed layer count, not a fixed cell height (the "914 mm plenum is 6
cells" fact is 914/6 = 152 mm = 6 in, equally consistent with 6 layers, and
the D8 Fig-12 sweep only reproduces Han's depth trend under fixed-count --
fixed 6-in cells leaves the shallow 305 mm plenum at 2 layers, badly
under-resolving the wall-jet dynamic pressure and under-predicting tile-flow
non-uniformity by ~35 %, while over-predicting the deep plenums).
"""
from __future__ import annotations

import numpy as np

from ..ffd_upwind.grid import Grid, uniform_faces
from ..ffd_upwind.solver import RackSpec, Solid

from .racks import rack_flow_m3s

PLENUM_NZ = 6  # Han's fixed plenum layer count (dz = depth / PLENUM_NZ)

_RACK_AXIS_FRONT = {
    "+x": ("x", "hi"), "-x": ("x", "lo"),
    "+y": ("y", "hi"), "-y": ("y", "lo"),
}


def _insert_faces(faces, values, min_dx):
    """Drop generated faces within `0.25*min_dx` of any `values`, then add them."""
    tol = 0.25 * min_dx
    values = sorted(set(values))
    kept = [f for f in faces if all(abs(f - v) >= tol for v in values)]
    return np.array(sorted(set(kept) | set(values)), dtype=np.float64)


def whitespace_grid(layout, cells_per_tile) -> Grid:
    """Tile-aligned white-space grid; z-faces forced at every solid top height."""
    nx_tiles, ny_tiles = layout.room_tiles
    xf = uniform_faces(nx_tiles * layout.tile_m, nx_tiles * cells_per_tile)
    yf = uniform_faces(ny_tiles * layout.tile_m, ny_tiles * cells_per_tile)

    dz = layout.tile_m / cells_per_tile
    nz = max(1, round(layout.height_m / dz))
    zf_base = uniform_faces(layout.height_m, nz)

    solid_heights = set(layout.rack_height_m.values()) | {layout.pdu_height_m}
    solid_heights = {h for h in solid_heights if 0.0 < h < layout.height_m}
    zf = _insert_faces(zf_base, solid_heights, dz)

    return Grid(xf, yf, zf)


def plenum_grid(layout, cells_per_tile, depth_m) -> Grid:
    """Tile-aligned plenum grid (with `plenum_margin_tiles` extra border), fixed `PLENUM_NZ` z-layers."""
    nx_tiles, ny_tiles = layout.room_tiles
    margin = layout.plenum_margin_tiles
    tnx, tny = nx_tiles + 2 * margin, ny_tiles + 2 * margin
    origin_x = -margin * layout.tile_m
    origin_y = -margin * layout.tile_m
    xf = uniform_faces(tnx * layout.tile_m, tnx * cells_per_tile, origin=origin_x)
    yf = uniform_faces(tny * layout.tile_m, tny * cells_per_tile, origin=origin_y)

    zf = uniform_faces(depth_m, PLENUM_NZ)

    return Grid(xf, yf, zf)


def _cells_per_tile(grid: Grid, layout) -> int:
    dx = grid.x.length / grid.nx
    return round(layout.tile_m / dx)


def _tile_offset_cells(grid: Grid, layout, cpt: int):
    """Cell-index of tile (0, 0)'s low corner, allowing for a plenum margin."""
    dx = layout.tile_m / cpt
    off_x = round(-grid.x.f[0] / dx)
    off_y = round(-grid.y.f[0] / dx)
    return off_x, off_y


def tile_cell_mask(grid: Grid, layout, tiles) -> np.ndarray:
    """bool[(nx,ny)], True on floor cells belonging to any of `tiles`."""
    cpt = _cells_per_tile(grid, layout)
    off_x, off_y = _tile_offset_cells(grid, layout, cpt)
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    for t in tiles:
        x0 = off_x + t.tile_ix * cpt
        y0 = off_y + t.tile_iy * cpt
        mask[x0:x0 + cpt, y0:y0 + cpt] = True
    return mask


def tile_cell_index(grid: Grid, layout) -> np.ndarray:
    """int[(nx,ny)]: index into `layout.tiles` per floor cell, -1 elsewhere."""
    cpt = _cells_per_tile(grid, layout)
    off_x, off_y = _tile_offset_cells(grid, layout, cpt)
    idx = -np.ones((grid.nx, grid.ny), dtype=int)
    for i, t in enumerate(layout.tiles):
        x0 = off_x + t.tile_ix * cpt
        y0 = off_y + t.tile_iy * cpt
        idx[x0:x0 + cpt, y0:y0 + cpt] = i
    return idx


def solids_from_layout(layout, grid: Grid, powers_kW: dict | None = None):
    """Solid blocks (racks + PDUs + stairs) for the white-space model.

    Racks get a `RackSpec` (flow `Q_m3s` from `racks.rack_flow_m3s(powers_kW
    [rack.id])`, `power_W = 1000*powers_kW[rack.id]`) whenever `powers_kW` is
    given and the rack is not `empty`; otherwise the rack is a plain adiabatic
    solid (`rack=None`) -- e.g. for D3's geometry-only checks, or Han's
    empty G11/G13. Only 1x1-tile rack footprints are supported (matches every
    digitized layout so far); PDUs/stairs are always plain adiabatic solids.
    """
    if layout.rack_width_tiles != 1 or layout.rack_depth_tiles != 1:
        raise NotImplementedError(
            "solids_from_layout only supports 1x1-tile rack footprints "
            f"(layout '{layout.name}' has {layout.rack_width_tiles}x"
            f"{layout.rack_depth_tiles})")

    solids = []
    for r in layout.racks:
        x0, y0 = r.tile_ix * layout.tile_m, r.tile_iy * layout.tile_m
        z1 = layout.rack_height_m[str(r.u_height)]
        bounds = (x0, x0 + layout.tile_m, y0, y0 + layout.tile_m, 0.0, z1)

        rack_spec = None
        if powers_kW is not None and not r.empty:
            axis, front = _RACK_AXIS_FRONT[r.facing]
            power_kW = powers_kW[r.id]
            rack_spec = RackSpec(axis=axis, front=front,
                                  Q_m3s=rack_flow_m3s(power_kW),
                                  power_W=power_kW * 1000.0)
        solids.append(Solid(bounds=bounds, temp=None, rack=rack_spec))

    for b in layout.blocks:
        x0, y0 = b.tile_ix * layout.tile_m, b.tile_iy * layout.tile_m
        bounds = (x0, x0 + layout.tile_m, y0, y0 + layout.tile_m, 0.0, b.height_m)
        solids.append(Solid(bounds=bounds, temp=None))

    return solids
