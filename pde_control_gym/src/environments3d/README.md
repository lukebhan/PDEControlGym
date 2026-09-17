# 3D environments

3D PDE control environments for PDEControlGym. All concrete 3D environments
inherit `PDEEnv3D` (`base_env_3d.py`).

## Data-center thermal model (in progress)

A reinforcement-learning environment around the **Han et al. (2021)** Fast Fluid
Dynamics data-center model:

> Han, W., et al. (2021). *An open source fast fluid dynamics model for data
> center thermal management.* Energy & Buildings 230, 110599.

### Vendored solver

- `ffd_upwind/` — the FFD-Upwind solver core (staggered MAC grid, first-order
  upwind advection–diffusion, pressure projection, Chen–Xu zero-equation
  turbulence). A NumPy/Numba reimplementation of the Han et al. solver.
- `datacenter/` — data-center modules built on the solver: `plenum.py`
  (underfloor plenum + VanGilder pressure-shift tile-flow solve), `whitespace.py`
  (above-floor room), plus `layout/mesh/tiles/racks/metrics/units`, the
  `layouts/` case data, `studies/`, `tools/`, and PRD/RCI validation (`verify.py`,
  `metrics.py`).

The coupling is **one-way**: the plenum is solved to steady state → per-tile
flows → the whitespace consumes them. The plenum is hidden from the agent.

### Dependencies

The solver is not part of the base install (kept lightweight). Install with:

```
pip install pdecontrolgym[datacenter]
```

- **scipy** — required (sparse pressure Poisson solve, `cKDTree`, `brentq`).
- **numba** — JIT for the hot point-Jacobi sweeps.
- **pyamg** — the *default* pressure-solver backend, but **optional at runtime**:
  if it is not installed, the solver falls back automatically to the scipy-only
  sparse LU (`lu_mmd`) with identical results (see `ffd_upwind/solver.py`,
  `_make_pressure_solver`).

### Status

- [x] `PDEEnv3D` base class.
- [x] Vendored FFD-Upwind solver + data-center modules (importable as a
      sub-package; plenum solve verified end-to-end).
- [ ] Concrete `DataCenter3D` gym environment on `PDEEnv3D`
      (action = supply flow rate + supply temperature; observation = whitespace
      field).
