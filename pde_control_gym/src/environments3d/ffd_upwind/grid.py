"""Structured, non-uniform, staggered (MAC) grid for the FFD-Upwind solver.
Scalars (p, T) live at cell center. (u, v, w) lives on (x, y, z)-faces
Coordinate convention for the room cases: x = length, y = width, z = height.
"""
from __future__ import annotations

import numpy as np


class Axis1D:
    """Geometry of a single non-uniform axis defined by its face coordinates."""

    def __init__(self, faces: np.ndarray):
        faces = np.asarray(faces, dtype=np.float64)
        if faces.ndim != 1 or faces.size < 2:
            raise ValueError("faces must be a 1-D array with >= 2 entries")
        if np.any(np.diff(faces) <= 0):
            raise ValueError("face coordinates must be strictly increasing")
        self.f = faces                           # (n+1,) face coordinates
        self.n = faces.size - 1                  # number of cells
        self.c = 0.5 * (faces[:-1] + faces[1:])
        self.d = np.diff(faces)                

        centers_ext = np.empty(self.n + 2)
        centers_ext[0] = faces[0]
        centers_ext[1:-1] = self.c
        centers_ext[-1] = faces[-1]
        self.dc = np.diff(centers_ext)           # (n+1,) center-to-center distances

    @property
    def length(self) -> float:
        return float(self.f[-1] - self.f[0])


def uniform_faces(length: float, n: int, origin: float = 0.0) -> np.ndarray:
    """Face coordinates for a uniform axis of `n` cells over [origin, origin+length]."""
    return origin + np.linspace(0.0, length, n + 1)


def stretched_faces(
    length: float,
    n: int,
    min_dx: float,
    refine_lo: bool = True,
    refine_hi: bool = True,
    origin: float = 0.0,
) -> np.ndarray:
    """Face coordinates clustered toward one or both ends.

    Builds a smooth cell-width distribution that reaches ~`min_dx` at the
    refined end(s) and expands toward the interior, then rescales so the widths
    sum exactly to `length`. Used to reproduce Han et al.'s near-inlet / near-
    wall refinement (min mesh ~0.5-1 cm) for the validation case mesh.
    """
    if not (refine_lo or refine_hi):
        return uniform_faces(length, n, origin)

    # Normalized coordinate of each cell center in [0, 1].
    s = (np.arange(n) + 0.5) / n
    # Weight -> 1 near a refined end, larger toward the interior.
    dist = np.ones(n)
    if refine_lo:
        dist = np.minimum(dist, s)
    if refine_hi:
        dist = np.minimum(dist, 1.0 - s)
    dist /= dist.max()  # in [0, 1], 0 at refined ends

    uniform_dx = length / n
    # Blend from min_dx at the refined ends up to a larger interior width.
    widths = min_dx + (uniform_dx * 2.0 - min_dx) * dist
    widths *= length / widths.sum()  # rescale to exact length

    faces = np.empty(n + 1)
    faces[0] = origin
    faces[1:] = origin + np.cumsum(widths)
    faces[-1] = origin + length  # kill rounding drift
    return faces


class Grid:
    """3-D staggered grid assembled from three `Axis1D` objects."""

    def __init__(self, xf: np.ndarray, yf: np.ndarray, zf: np.ndarray):
        self.x = Axis1D(xf)
        self.y = Axis1D(yf)
        self.z = Axis1D(zf)
        self.nx, self.ny, self.nz = self.x.n, self.y.n, self.z.n
        self.shape = (self.nx, self.ny, self.nz)

    @classmethod
    def uniform(cls, lengths, ncells, origin=(0.0, 0.0, 0.0)) -> "Grid":
        lx, ly, lz = lengths
        nx, ny, nz = ncells
        ox, oy, oz = origin
        return cls(
            uniform_faces(lx, nx, ox),
            uniform_faces(ly, ny, oy),
            uniform_faces(lz, nz, oz),
        )

    # --- cell-center coordinate meshes ----------------------------------
    def centers(self):
        """Return (Xc, Yc, Zc) broadcastable cell-center coordinate arrays."""
        return np.meshgrid(self.x.c, self.y.c, self.z.c, indexing="ij")

    def cell_volumes(self) -> np.ndarray:
        """(nx, ny, nz) array of cell volumes dx*dy*dz."""
        return (
            self.x.d[:, None, None]
            * self.y.d[None, :, None]
            * self.z.d[None, None, :]
        )

    def __repr__(self) -> str:
        return (
            f"Grid({self.nx}x{self.ny}x{self.nz}, "
            f"L=[{self.x.length:.3g},{self.y.length:.3g},{self.z.length:.3g}])"
        )
