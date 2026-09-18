# 3D environments

3D PDE control environments for PDEControlGym. All concrete 3D environments
inherit `PDEEnv3D` (`base_env_3d.py`).

## Data-center thermal model

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

### `DataCenter3D` at a glance

One env step = one **steady** white-space solve for the current action (warm
started), so only converged fields are observed (see the rationale above).

- **Action** `Box(2)` in [-1, 1] → (supply flow, supply temperature), mapped to
  physical ranges (`flow_bounds_m3h`, `temp_bounds_C`).
- **Coupling** `plenum_mode="scaled"`: the plenum is solved once; per-tile flow
  *fractions* are cached and rescaled by supply flow each step. `"full"`
  re-solves the plenum every step.
- **Observation** `sensing="rack_inlet"` (default): per-rack inlet temps +
  setpoints + IT load; `"full"`: the whole `(nx, ny, nz, 4)` field.
- **Cost knobs**: `max_solve_steps` (per-step budget), `solve_tol`, `warm_start`.

```python
import gymnasium as gym
env = gym.make("PDEControlGym-DataCenter3D", layout="mini", cells_per_tile=2,
               max_solve_steps=300, episode_steps=3)
obs, info = env.reset(seed=0)
obs, reward, term, trunc, info = env.step(env.action_space.sample())
```

**Tractability (measured):** warm-started steady solves run ~seconds/step at
toy (`mini`) scale but ~minutes/step at `han_reference` (cpt=2, ~89k cells) —
tractable for classical control episodes and toy-scale RL, but realistic-scale
RL needs a coarser grid, looser tolerance, a bounded `max_solve_steps`, or a
surrogate.
