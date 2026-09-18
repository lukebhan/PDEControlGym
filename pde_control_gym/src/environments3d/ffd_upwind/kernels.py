"""Numba-compiled inner loops for the FFD-Upwind solver.

Coefficient convention for a cell (i, j, k):

    aP * psi = aE*psi[i+1] + aW*psi[i-1]
             + aN*psi[i,j+1] + aS*psi[i,j-1]
             + aF*psi[i,j,k+1] + aB*psi[i,j,k-1] + b

Neighbor coefficients that point outside the array (or at a wall) must be set to
0 by the assembler; the sweep additionally guards the array bounds. `fixed` marks
Dirichlet cells (boundary velocity nodes, solid cells) whose value is held.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True)
def _jacobi_pass(psi, out, aP, aE, aW, aN, aS, aF, aB, b, fixed):
    nx, ny, nz = psi.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                if fixed[i, j, k]:
                    out[i, j, k] = psi[i, j, k]
                    continue
                s = b[i, j, k]
                if i + 1 < nx:
                    s += aE[i, j, k] * psi[i + 1, j, k]
                if i > 0:
                    s += aW[i, j, k] * psi[i - 1, j, k]
                if j + 1 < ny:
                    s += aN[i, j, k] * psi[i, j + 1, k]
                if j > 0:
                    s += aS[i, j, k] * psi[i, j - 1, k]
                if k + 1 < nz:
                    s += aF[i, j, k] * psi[i, j, k + 1]
                if k > 0:
                    s += aB[i, j, k] * psi[i, j, k - 1]
                out[i, j, k] = s / aP[i, j, k]


def jacobi(psi, aP, aE, aW, aN, aS, aF, aB, b, fixed, n_sweeps):
    """Run `n_sweeps` point-Jacobi passes in place; returns the updated array.
    The result is written back into `psi` so callers keep their reference.
    """
    tmp = np.empty_like(psi)
    a, bff = psi, tmp
    for _ in range(n_sweeps):
        _jacobi_pass(a, bff, aP, aE, aW, aN, aS, aF, aB, b, fixed)
        a, bff = bff, a
    if a is not psi:
        psi[...] = a
    return psi


@njit(parallel=True, cache=True, fastmath=True)
def linf_residual(psi, aP, aE, aW, aN, aS, aF, aB, b, fixed):
    """Max-norm of the linear-system residual b - A psi over non-fixed cells."""
    nx, ny, nz = psi.shape
    part = np.zeros(nx)
    for i in prange(nx):
        ri = 0.0
        for j in range(ny):
            for k in range(nz):
                if fixed[i, j, k]:
                    continue
                s = b[i, j, k] - aP[i, j, k] * psi[i, j, k]
                if i + 1 < nx:
                    s += aE[i, j, k] * psi[i + 1, j, k]
                if i > 0:
                    s += aW[i, j, k] * psi[i - 1, j, k]
                if j + 1 < ny:
                    s += aN[i, j, k] * psi[i, j + 1, k]
                if j > 0:
                    s += aS[i, j, k] * psi[i, j - 1, k]
                if k + 1 < nz:
                    s += aF[i, j, k] * psi[i, j, k + 1]
                if k > 0:
                    s += aB[i, j, k] * psi[i, j, k - 1]
                a = abs(s)
                if a > ri:
                    ri = a
        part[i] = ri
    return part.max()
