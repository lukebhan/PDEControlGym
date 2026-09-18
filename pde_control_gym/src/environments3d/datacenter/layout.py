"""Layout dataclasses and loader/validator for data-center cases.

A layout is a directory `layouts/<name>/` holding four files:

  site.json       -- scalar/config fields (room size, plenum, supply, power)
  floor_map.txt   -- room_tiles[1] lines of room_tiles[0] chars; first line is
                     the TOP row (iy = ny-1), so the file reads like Fig 9.
                     '.' floor, 'T' perforated tile, 'R' rack cell (must be
                     covered 1:1 by racks.csv), 'P' PDU, 'S' stairs.
  racks.csv       -- id,row,number,tile_ix,tile_iy,facing,u_height,empty
  ceiling_map.txt -- same shape as floor_map.txt; 'C' = ceiling return tile,
                     '.' otherwise. Optional (ceiling_tiles = [] if absent).

Coordinates: tile (tile_ix, tile_iy) is 0-based, x = room_tiles[0] dimension,
y = room_tiles[1] dimension, iy increases from the bottom of Fig 9 upward.
"""

import csv
import json
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rack:
    id: str
    row: str
    number: int
    tile_ix: int
    tile_iy: int
    facing: str  # '+x', '-x', '+y', '-y': direction the inlet/front faces
    u_height: int  # 42 or 45
    empty: bool  # True for Han's G11/G13: present, unpowered


@dataclass(frozen=True)
class Tile:
    tile_ix: int
    tile_iy: int
    open_area: float


@dataclass(frozen=True)
class Block:
    tile_ix: int
    tile_iy: int
    kind: str  # 'pdu' or 'stairs'
    height_m: float


@dataclass(frozen=True)
class CeilingTile:
    tile_ix: int
    tile_iy: int
    open_area: float


@dataclass(frozen=True)
class PlenumInlet:
    face: str  # 'xlo', 'xhi', 'ylo', 'yhi'
    fraction: float


_FACING_DELTA = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}


@dataclass
class Layout:
    name: str
    room_tiles: tuple
    tile_m: float
    height_m: float
    plenum_depth_m: float
    plenum_margin_tiles: int
    plenum_inlets: list
    tile_open_area: float
    ceiling_tile_open_area: float
    supply_flow_m3h: float
    supply_T_C: float
    total_it_power_kW: float
    power_mode: str
    rack_height_m: dict
    rack_depth_tiles: int
    rack_width_tiles: int
    pdu_height_m: float
    racks: list
    tiles: list
    ceiling_tiles: list
    blocks: list

    def summary(self):
        nx, ny = self.room_tiles
        by_u = {}
        n_empty = 0
        for r in self.racks:
            by_u[r.u_height] = by_u.get(r.u_height, 0) + 1
            n_empty += int(r.empty)
        n_pdu = sum(1 for b in self.blocks if b.kind == "pdu")
        n_stairs = sum(1 for b in self.blocks if b.kind == "stairs")
        lines = [
            f"Layout '{self.name}': {nx}x{ny} tiles ({nx*self.tile_m:.1f} x {ny*self.tile_m:.1f} m)",
            f"  racks: {len(self.racks)} total, empty: {n_empty}",
        ]
        for u in sorted(by_u):
            lines.append(f"    {u}U: {by_u[u]}")
        lines.append(f"  floor tiles: {len(self.tiles)}")
        lines.append(f"  ceiling tiles: {len(self.ceiling_tiles)}")
        lines.append(f"  PDUs: {n_pdu} (blocks), stairs blocks: {n_stairs}")
        text = "\n".join(lines)
        print(text)
        return text


def _read_map(path):
    with open(path) as f:
        lines = [line.rstrip("\n") for line in f if line.strip("\n") != "" or True]
    lines = [line for line in lines if line != ""]
    ny = len(lines)
    nx = len(lines[0])
    for line in lines:
        if len(line) != nx:
            raise ValueError(
                f"{path}: ragged row (expected width {nx}, got {len(line)})"
            )
    grid = [[None] * nx for _ in range(ny)]
    for row_index, line in enumerate(lines):
        iy = (ny - 1) - row_index
        for ix, ch in enumerate(line):
            grid[iy][ix] = ch
    return grid, nx, ny


def load_layout(path):
    with open(os.path.join(path, "site.json")) as f:
        site = json.load(f)

    nx, ny = site["room_tiles"]
    grid, fnx, fny = _read_map(os.path.join(path, "floor_map.txt"))
    if (fnx, fny) != (nx, ny):
        raise ValueError(
            f"floor_map.txt is {fnx}x{fny}, site.json room_tiles says {nx}x{ny}"
        )

    tile_open_area = site["tile_open_area"]
    tiles = []
    blocks = []
    for iy in range(ny):
        for ix in range(nx):
            ch = grid[iy][ix]
            if ch == "T":
                tiles.append(Tile(ix, iy, tile_open_area))
            elif ch == "P":
                blocks.append(Block(ix, iy, "pdu", site["pdu_height_m"]))
            elif ch == "S":
                blocks.append(Block(ix, iy, "stairs", site["height_m"]))
            elif ch not in (".", "R"):
                raise ValueError(
                    f"floor_map.txt: unknown char {ch!r} at ix={ix},iy={iy}"
                )

    racks = []
    racks_csv = os.path.join(path, "racks.csv")
    with open(racks_csv, newline="") as f:
        for row in csv.DictReader(f):
            racks.append(
                Rack(
                    id=row["id"],
                    row=row["row"],
                    number=int(row["number"]),
                    tile_ix=int(row["tile_ix"]),
                    tile_iy=int(row["tile_iy"]),
                    facing=row["facing"],
                    u_height=int(row["u_height"]),
                    empty=bool(int(row["empty"])),
                )
            )

    ceiling_tiles = []
    ceiling_path = os.path.join(path, "ceiling_map.txt")
    if os.path.exists(ceiling_path):
        cgrid, cnx, cny = _read_map(ceiling_path)
        if (cnx, cny) != (nx, ny):
            raise ValueError(
                f"ceiling_map.txt is {cnx}x{cny}, site.json room_tiles says {nx}x{ny}"
            )
        for iy in range(ny):
            for ix in range(nx):
                if cgrid[iy][ix] == "C":
                    ceiling_tiles.append(
                        CeilingTile(ix, iy, site["ceiling_tile_open_area"])
                    )
                elif cgrid[iy][ix] != ".":
                    raise ValueError(
                        f"ceiling_map.txt: unknown char {cgrid[iy][ix]!r} at ix={ix},iy={iy}"
                    )

    plenum_inlets = [
        PlenumInlet(pi["face"], pi["fraction"]) for pi in site["plenum_inlets"]
    ]

    layout = Layout(
        name=site.get("name", os.path.basename(os.path.normpath(path))),
        room_tiles=(nx, ny),
        tile_m=site["tile_m"],
        height_m=site["height_m"],
        plenum_depth_m=site["plenum_depth_m"],
        plenum_margin_tiles=site["plenum_margin_tiles"],
        plenum_inlets=plenum_inlets,
        tile_open_area=tile_open_area,
        ceiling_tile_open_area=site["ceiling_tile_open_area"],
        supply_flow_m3h=site["supply_flow_m3h"],
        supply_T_C=site["supply_T_C"],
        total_it_power_kW=site["total_it_power_kW"],
        power_mode=site["power_mode"],
        rack_height_m={str(k): v for k, v in site["rack_height_m"].items()},
        rack_depth_tiles=site["rack_depth_tiles"],
        rack_width_tiles=site["rack_width_tiles"],
        pdu_height_m=site["pdu_height_m"],
        racks=racks,
        tiles=tiles,
        ceiling_tiles=ceiling_tiles,
        blocks=blocks,
    )
    # stash the raw floor grid for validate(); not part of the public dataclass fields
    layout._floor_grid = grid
    return layout


def validate(layout):
    """Check cross-file consistency; raise ValueError listing every violation."""
    violations = []
    nx, ny = layout.room_tiles
    grid = getattr(layout, "_floor_grid", None)

    if grid is not None:
        r_cells = {
            (ix, iy) for iy in range(ny) for ix in range(nx) if grid[iy][ix] == "R"
        }
        rack_cells = {}
        for r in layout.racks:
            key = (r.tile_ix, r.tile_iy)
            if key in rack_cells:
                violations.append(
                    f"racks.csv: {rack_cells[key]} and {r.id} both claim tile {key}"
                )
            rack_cells[key] = r.id

        for key in r_cells - rack_cells.keys():
            violations.append(
                f"floor_map.txt: 'R' cell at {key} has no matching rack in racks.csv"
            )
        for key, rid in rack_cells.items():
            if key not in r_cells:
                violations.append(
                    f"racks.csv: rack {rid} at {key} is not an 'R' cell in floor_map.txt"
                )

    solid_kinds = {(b.tile_ix, b.tile_iy) for b in layout.blocks}
    if grid is not None:
        solid_kinds |= {
            (ix, iy) for iy in range(ny) for ix in range(nx) if grid[iy][ix] == "R"
        }

    for r in layout.racks:
        dx, dy = _FACING_DELTA.get(r.facing, (None, None))
        if dx is None:
            violations.append(f"rack {r.id}: unknown facing {r.facing!r}")
            continue
        fx, fy = r.tile_ix + dx, r.tile_iy + dy
        if not (0 <= fx < nx and 0 <= fy < ny):
            violations.append(
                f"rack {r.id}: front cell ({fx},{fy}) is outside the room"
            )
        elif (fx, fy) in solid_kinds:
            violations.append(f"rack {r.id}: front cell ({fx},{fy}) is solid")

    if violations:
        raise ValueError(
            f"layout '{layout.name}' failed validation ({len(violations)} issue(s)):\n  "
            + "\n  ".join(violations)
        )
    return True
