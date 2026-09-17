"""Shared room mesh for the Han et al. (2021) Section 3.1 validation cases.

Case 1 (empty room) and Case 2 (room + box) model the exact same 2.44 m cube,
inlet/outlet slots, and (for Case 2) box footprint -- so both cases build
their grid from this one module and get identical face coordinates. Case 1
has no box, but still gets faces pinned at the box's location; those extra
cell boundaries are harmless (just some unused resolution there) and mean a
location that is fluid in both cases sits in the exact same cell in both,
with no case-to-case grid confound.

Two profiles, same graded scheme (refine toward every wall/slot/box face),
different cell counts:
  "fast"  26x26x26  -- fast iteration
  "han"   40x40x40  -- matches Han's Table 1; THE DEFAULT

Refinement matches Han (Section 3.1.3, confirmed from the PDF text): the bulk
mesh averages ~6 cm, the inlet/near-ceiling band is refined to a minimum of
~0.5 cm, and the outlet/near-floor band to ~1 cm. `N_OUT_BAND`/`N_IN_BAND`
graded cells (via `stretched_faces`) reach ~1.05 cm / ~0.45 cm at those walls.
"""
import numpy as np

from .grid import Grid, stretched_faces

L = 2.44                    # cube edge [m]
H_IN = 0.03                 # inlet slot height [m]  (top of west wall)
H_OUT = 0.08                # outlet slot height [m] (bottom of east wall)
Z_IN = L - H_IN             # 2.41  -> inlet spans [Z_IN, L]
Q = 0.10                    # supply flow rate [m^3/s]
U_IN = Q / (H_IN * L)       # 1.366 m/s
NU = 1.5e-5

BOX_LO, BOX_HI = 0.61, 1.83     # box horizontal span [0.61, 1.83] in x and y
BOX_Z_LO, BOX_Z_HI = 0.0, 1.22  # box is FLOOR-MOUNTED: z in [0, 1.22] (Han Fig 5a)

N_OUT_BAND = 4               # graded cells in the outlet/near-floor slot band
N_IN_BAND = 4                # graded cells in the inlet/near-ceiling slot band

PROFILES = {
    "fast": dict(xy_side=8,  xy_box=10, z_box=8,  z_above=10),   # 26,26,26
    "han":  dict(xy_side=13, xy_box=14, z_box=12, z_above=20),   # 40,40,40
}


def _seg(a, b, n, min_dx, lo=True, hi=True):
    """Stretched faces on [a, b] with n cells, refined toward chosen ends."""
    return stretched_faces(b - a, n, min_dx=min_dx, refine_lo=lo, refine_hi=hi,
                            origin=a)


def _concat(*segs):
    """Concatenate face segments that share endpoints into one face array."""
    out = [segs[0]]
    for s in segs[1:]:
        out.append(s[1:])          # drop the shared start point
    return np.concatenate(out)


def build_x_faces(p):
    """x (and y) faces: west wall -> box -> east wall, refined at every face."""
    return _concat(
        _seg(0.0, BOX_LO, p["xy_side"], 0.03),       # west wall .. box, refine both
        _seg(BOX_LO, BOX_HI, p["xy_box"], 0.06),     # across the box
        _seg(BOX_HI, L, p["xy_side"], 0.03),         # box .. east wall
    )


def build_z_faces(p):
    """z faces: outlet band -> box top -> inlet band, box footprint on floor."""
    return _concat(
        stretched_faces(H_OUT, N_OUT_BAND, min_dx=0.01,
                        refine_lo=True, refine_hi=False, origin=0.0),
        _seg(H_OUT, BOX_Z_HI, p["z_box"], 0.04),     # up through the box to its top
        _seg(BOX_Z_HI, Z_IN, p["z_above"], 0.05),    # box top .. inlet
        stretched_faces(H_IN, N_IN_BAND, min_dx=0.005,
                        refine_lo=False, refine_hi=True, origin=Z_IN),
    )


def build_grid(profile="han"):
    """Build the shared room grid. Default profile is "han" (40^3)."""
    p = PROFILES[profile]
    xf = build_x_faces(p)
    return Grid(xf, xf, build_z_faces(p))
