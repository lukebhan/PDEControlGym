#!/usr/bin/env python3
"""Generate ceiling_map.txt: N ceiling return tiles evenly spaced above the
hot aisles of a layout.

Han gives fixed-flow ceiling outlets but not their positions (Section
2.2.2.1 / Fig 9 has no ceiling-tile drawing), so this placement is our own
choice, not a digitization. A "hot aisle" band is the line directly behind a
rack's rear (exhaust) face -- opposite its `facing` direction: a vertical
band (fixed x) for x-facing racks, a horizontal band (fixed y) for y-facing
racks. n_tiles are distributed across bands proportional to each band's
occupied length (largest-remainder rounding), then spaced as evenly as
possible along each band.

Run:  ../../.venv/bin/python3 make_ceiling_map.py <layout_dir> <n_tiles>
"""

import csv
import json
import os
import sys

_FACING_DELTA = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


def _load_racks_and_size(layout_dir):
    with open(os.path.join(layout_dir, "site.json")) as f:
        site = json.load(f)
    nx, ny = site["room_tiles"]
    racks = []
    with open(os.path.join(layout_dir, "racks.csv"), newline="") as f:
        for row in csv.DictReader(f):
            racks.append((int(row["tile_ix"]), int(row["tile_iy"]), row["facing"]))
    return racks, nx, ny


def hot_aisle_bands(racks):
    """{('x', rear_x) or ('y', rear_y): sorted [pos, ...]} behind each rack's rear face.

    x-facing racks (front points along x) form a vertical band at a fixed
    rear_x, with racks at varying iy; y-facing racks form a horizontal band
    at a fixed rear_y, with racks at varying ix. `pos` is the coordinate
    along the band (iy for an 'x' band, ix for a 'y' band).
    """
    bands = {}
    for ix, iy, facing in racks:
        dx, dy = _FACING_DELTA[facing]
        rear_x, rear_y = ix - dx, iy - dy
        if dx != 0:
            key, pos = ("x", rear_x), iy
        else:
            key, pos = ("y", rear_y), ix
        bands.setdefault(key, set()).add(pos)
    return {k: sorted(v) for k, v in sorted(bands.items())}


def _largest_remainder(counts, total):
    """counts: {key: weight}. Return {key: int} summing to total, proportional to weight."""
    weight_sum = sum(counts.values())
    raw = {k: total * w / weight_sum for k, w in counts.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(floors.values())
    order = sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def _evenly_spaced_indices(n_available, n_pick):
    if n_pick <= 0:
        return []
    if n_pick >= n_available:
        return list(range(n_available))
    return sorted(
        {round(i * (n_available - 1) / (n_pick - 1)) for i in range(n_pick)}
        if n_pick > 1
        else {n_available // 2}
    )


def make_ceiling_grid(layout_dir, n_tiles):
    racks, nx, ny = _load_racks_and_size(layout_dir)
    bands = hot_aisle_bands(racks)
    counts = {key: len(positions) for key, positions in bands.items()}
    alloc = _largest_remainder(counts, n_tiles)

    grid = [["."] * nx for _ in range(ny)]
    placed = 0
    for key, positions in bands.items():
        axis, fixed = key
        k = alloc[key]
        for idx in _evenly_spaced_indices(len(positions), k):
            p = positions[idx]
            ix, iy = (fixed, p) if axis == "x" else (p, fixed)
            grid[iy][ix] = "C"
            placed += 1
    assert placed == n_tiles, f"placed {placed} != requested {n_tiles}"
    return grid, nx, ny


def write_ceiling_map(layout_dir, n_tiles):
    grid, nx, ny = make_ceiling_grid(layout_dir, n_tiles)
    out_path = os.path.join(layout_dir, "ceiling_map.txt")
    with open(out_path, "w") as f:
        for row_index in range(ny - 1, -1, -1):
            f.write("".join(grid[row_index]) + "\n")
    return out_path


if __name__ == "__main__":
    layout_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "..", "layouts", "han_reference")
    )
    n_tiles = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    path = write_ceiling_map(layout_dir, n_tiles)
    print(f"wrote {path} ({n_tiles} ceiling tiles)")
