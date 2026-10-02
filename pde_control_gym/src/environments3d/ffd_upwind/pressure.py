"""Pressure Poisson operator and its reusable linear solver.

The discrete Laplacian is built once from the grid metrics and the boundary
layout, then factorized; `Solver.project` fills the RHS and back-solves every
step. Both functions are pure (no `Solver` state) so the numerics stay testable
in isolation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .config import FACES


def build_pressure_operator(grid, dx, dy, dz, dxc, dyc, dzc, pdir, solid):
    """Discrete Laplacian L (SPD) for the pressure, with a Neumann boundary
    condition at walls and inlets and a Dirichlet (p=0) ghost only where an
    outlet opening sits (`pdir`). RHS is filled each step from div(u*):
    L p = -(1/dt) div(u*) * Vol.

    Returns (L, pin_cell): a CSC matrix and, for a pure-Neumann system (no
    Dirichlet anywhere), the index of the fluid cell pinned to p=0 to make L
    nonsingular (else None). The pin zeros both its row and column, keeping L
    symmetric for the 'cholesky'/'amg' backends.
    """
    nx, ny, nz = grid.nx, grid.ny, grid.nz
    N = nx * ny * nz

    def idx(i, j, k):
        return (i * ny + j) * nz + k

    Ax = np.multiply.outer(dy, dz)  # (ny,nz)
    Ay = np.multiply.outer(dx, dz)  # (nx,nz)
    Az = np.multiply.outer(dx, dy)  # (nx,ny)

    # A face contributes a coefficient A/dist unless it's a domain-boundary
    # face with a Neumann pressure BC (wall/inlet) -> no term.
    rows, cols, vals = [], [], []
    diag = np.zeros(N)

    S = solid

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
                        c = Ax[j, k] / dxc[i + 1]
                        rows.append(P)
                        cols.append(idx(i + 1, j, k))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["xhi"][j, k]:
                    diag[P] += Ax[j, k] / (dx[i] * 0.5)
                # West (-x)
                if i - 1 >= 0:
                    if not S[i - 1, j, k]:
                        c = Ax[j, k] / dxc[i]
                        rows.append(P)
                        cols.append(idx(i - 1, j, k))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["xlo"][j, k]:
                    diag[P] += Ax[j, k] / (dx[i] * 0.5)
                # North (+y)
                if j + 1 < ny:
                    if not S[i, j + 1, k]:
                        c = Ay[i, k] / dyc[j + 1]
                        rows.append(P)
                        cols.append(idx(i, j + 1, k))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["yhi"][i, k]:
                    diag[P] += Ay[i, k] / (dy[j] * 0.5)
                # South (-y)
                if j - 1 >= 0:
                    if not S[i, j - 1, k]:
                        c = Ay[i, k] / dyc[j]
                        rows.append(P)
                        cols.append(idx(i, j - 1, k))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["ylo"][i, k]:
                    diag[P] += Ay[i, k] / (dy[j] * 0.5)
                # Front (+z)
                if k + 1 < nz:
                    if not S[i, j, k + 1]:
                        c = Az[i, j] / dzc[k + 1]
                        rows.append(P)
                        cols.append(idx(i, j, k + 1))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["zhi"][i, j]:
                    diag[P] += Az[i, j] / (dz[k] * 0.5)
                # Back (-z)
                if k - 1 >= 0:
                    if not S[i, j, k - 1]:
                        c = Az[i, j] / dzc[k]
                        rows.append(P)
                        cols.append(idx(i, j, k - 1))
                        vals.append(-c)
                        diag[P] += c
                elif pdir["zlo"][i, j]:
                    diag[P] += Az[i, j] / (dz[k] * 0.5)

    # Pure-Neumann system (no Dirichlet cell anywhere) is singular: pin the
    # first fluid cell to p=0.
    any_dirichlet = any(pdir[f].any() for f in FACES)
    rows.extend(range(N))
    cols.extend(range(N))
    vals.extend(diag)
    L = sp.csc_matrix((vals, (rows, cols)), shape=(N, N))
    if any_dirichlet:
        return L, None
    pin = int(np.argmin(S.reshape(-1)))  # first fluid cell
    L = L.tolil()
    L[pin, :] = 0.0
    L[:, pin] = 0.0
    L[pin, pin] = 1.0
    return L.tocsc(), pin


def make_pressure_solver(L, cfg):
    """Build the reusable pressure solve for cfg.pressure_solver.

    See Config.pressure_solver. The pin (if any) is applied symmetrically (see
    build_pressure_operator), so every pressure_solver option is available
    regardless of whether the domain has a pressure outlet. Returns
    (solve_fn, name) where name is the backend actually used (after any
    optional-dependency fallback).
    """
    want = cfg.pressure_solver
    N = L.shape[0]
    if want == "cholesky":
        try:
            from cholespy import CholeskySolverD, MatrixType
        except ImportError:
            want = "lu_mmd"
        else:
            coo = L.tocoo()
            chol = CholeskySolverD(
                N,
                coo.row.astype(np.int32),
                coo.col.astype(np.int32),
                coo.data.astype(np.float64),
                MatrixType.COO,
            )
            buf = np.empty(N)

            def solve(rhs, _c=chol, _b=buf):
                _c.solve(np.ascontiguousarray(rhs, dtype=np.float64), _b)
                return _b.copy()  # caller keeps the result as self.p

            return solve, "cholesky"
    if want == "amg":
        try:
            import pyamg
        except ImportError:
            # pyamg is the default backend but an optional dependency; when
            # it is not installed, fall back to the scipy-only sparse LU.
            want = "lu_mmd"
        else:
            ml = pyamg.ruge_stuben_solver(L.tocsr())
            tol = cfg.pressure_tol
            last = {"x": np.zeros(N)}

            def solve(rhs, _ml=ml, _t=tol, _s=last):
                x = _ml.solve(rhs, x0=_s["x"], tol=_t, accel="cg", maxiter=200)
                _s["x"] = x
                return x

            return solve, "amg"
    if want == "lu_mmd":
        return spla.splu(L.tocsc(), permc_spec="MMD_AT_PLUS_A").solve, "lu_mmd"
    return spla.factorized(L), "lu"
