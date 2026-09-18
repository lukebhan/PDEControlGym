"""DataCenter3D: an RL environment around the Han et al. (2021) FFD data-center
thermal model, on top of :class:`PDEEnv3D`.

Design (see environments3d/README.md for the full rationale):

- **Action** = 2 scalars: supply flow rate and supply air temperature. The
  agent outputs them in [-1, 1]; the env maps each to its physical range.
- **Coupling** is one-way: the underfloor plenum is solved to steady state to
  get the per-tile flow split; the white space (room) consumes those flows.
  With a fixed tile open-area the split is self-similar in supply flow, so the
  plenum is solved once and the per-tile *fractions* are cached and rescaled by
  the supply flow each step (``plenum_mode="scaled"``). ``plenum_mode="full"``
  re-solves the plenum every step instead (slower, no self-similarity
  assumption).
- **One step = one steady white-space solve** for the current action, warm
  started from the previous step's field. Because Fast Fluid Dynamics is a
  steady-validated method, only converged fields are ever observed. A bounded
  ``max_solve_steps`` budget caps the per-step cost (see the tractability note
  in the README: warm-started steady solves are ~seconds/step at toy scale but
  ~minutes/step at ``han_reference`` resolution).
- **Observation**: ``"rack_inlet"`` (default) is a low-dim sensor vector
  (per-rack inlet temperatures + current setpoints + current IT load), matching
  real data-center instrumentation and the reward target; ``"full"`` exposes
  the whole 3D field.
- **Reward**: pluggable via ``reward_class`` (defaults to
  :class:`~pde_control_gym.src.rewards.dc_reward.DataCenterReward`).
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
from gymnasium import spaces

from pde_control_gym.src.environments3d.base_env_3d import PDEEnv3D
from pde_control_gym.src.rewards.dc_reward import DataCenterReward
from pde_control_gym.src.environments3d.datacenter.layout import load_layout
from pde_control_gym.src.environments3d.datacenter.plenum import run_plenum
from pde_control_gym.src.environments3d.datacenter.whitespace import build_whitespace
from pde_control_gym.src.environments3d.datacenter.mesh import (
    whitespace_grid,
    tile_cell_index,
    tile_cell_mask,
)
from pde_control_gym.src.environments3d.datacenter.tiles import body_force
from pde_control_gym.src.environments3d.datacenter.racks import (
    assign_powers,
    rack_flow_m3s,
)
from pde_control_gym.src.environments3d.datacenter.units import m3h_to_m3s
from pde_control_gym.src.environments3d.datacenter import metrics

LAYOUT_DIR = os.path.join(os.path.dirname(__file__), "datacenter", "layouts")


class DataCenter3D(PDEEnv3D):
    """
    See the module docstring for the design. Key parameters:

    :param layout: layout name under ``datacenter/layouts/`` (e.g. ``"mini"``,
        ``"han_reference"``) or an absolute path to a layout directory.
    :param cells_per_tile: white-space/plenum mesh resolution (Han uses 2 and 4).
    :param flow_bounds_m3h: (min, max) supply flow [m^3/h] for the action range.
        Defaults to (0.5, 1.5) x the layout's nominal supply flow.
    :param temp_bounds_C: (min, max) supply temperature [C] for the action range.
    :param plenum_mode: ``"scaled"`` (cache the tile-flow fractions once and
        rescale by supply flow) or ``"full"`` (re-solve the plenum each step).
    :param plenum_beta: tile open-area ratio for the plenum solve. Defaults to
        the layout's ``tile_open_area``.
    :param solve_tol: white-space steady-state tolerance (du/dt and dT/dt).
    :param max_solve_steps: cap on solver sub-steps per env step (the budget knob).
    :param min_solve_steps: minimum sub-steps before convergence may be declared.
    :param check_every: how often (in sub-steps) to test convergence.
    :param warm_start: reuse the previous step's field as the initial guess.
    :param sensing: ``"rack_inlet"`` (sensor vector) or ``"full"`` (whole field).
    :param sensing_noise_func: optional callable applied to the observation array.
    :param load_profile: optional callable ``step_index -> total_IT_power_kW``
        giving a time-varying IT load; defaults to the layout's constant load.
    :param episode_steps: number of control decisions per episode.
    :param reward_class: a :class:`BaseReward`; defaults to
        :class:`DataCenterReward` normalized to this layout.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layout: str = "mini",
        cells_per_tile: int = 2,
        flow_bounds_m3h: Optional[tuple] = None,
        temp_bounds_C: tuple = (14.0, 22.0),
        plenum_mode: str = "scaled",
        plenum_beta: Optional[float] = None,
        solve_tol: float = 5e-5,
        max_solve_steps: int = 1500,
        min_solve_steps: int = 100,
        check_every: int = 100,
        warm_start: bool = True,
        sensing: str = "rack_inlet",
        sensing_noise_func: Optional[Callable] = None,
        load_profile: Optional[Callable[[int], float]] = None,
        episode_steps: int = 100,
        reward_class=None,
    ):
        self.layout_name = layout
        layout_path = (
            layout if os.path.isdir(layout) else os.path.join(LAYOUT_DIR, layout)
        )
        self.layout = load_layout(layout_path)
        self.cells_per_tile = cells_per_tile
        self.plenum_mode = plenum_mode
        self.plenum_beta = (
            self.layout.tile_open_area if plenum_beta is None else plenum_beta
        )
        self.solve_tol = solve_tol
        self.max_solve_steps = max_solve_steps
        self.min_solve_steps = min_solve_steps
        self.check_every = check_every
        self.warm_start = warm_start
        self.sensing = sensing
        self.sensing_noise_func = sensing_noise_func
        self.episode_steps = episode_steps

        # Reference operating point + action ranges.
        self.Q_nom_m3s = m3h_to_m3s(self.layout.supply_flow_m3h)
        if flow_bounds_m3h is None:
            self.flow_bounds = (0.5 * self.Q_nom_m3s, 1.5 * self.Q_nom_m3s)
        else:
            self.flow_bounds = (
                m3h_to_m3s(flow_bounds_m3h[0]),
                m3h_to_m3s(flow_bounds_m3h[1]),
            )
        self.temp_bounds = tuple(float(t) for t in temp_bounds_C)

        # Base IT load (constant unless load_profile overrides it per step).
        self._base_powers = assign_powers(
            self.layout.racks, self.layout.total_it_power_kW, self.layout.power_mode
        )
        self.p_it_nom_kW = float(sum(self._base_powers.values()))
        self.load_profile = load_profile
        # Powered racks in the order the solver's flow-through racks are built
        # (non-empty racks in layout order -- see datacenter.mesh.solids_from_layout);
        # used to push a time-varying load into the live warm-started solver.
        self._powered_racks = [
            r for r in self.layout.racks if not getattr(r, "empty", False)
        ]

        # Precompute the plenum tile-flow fractions (self-similar in supply flow).
        pr = run_plenum(
            self.layout,
            cells_per_tile,
            self.layout.plenum_depth_m,
            self.plenum_beta,
            self.Q_nom_m3s,
            max_steps=2000,
        )
        self._tile_fractions = np.asarray(pr.Q_m3s, dtype=float) / float(
            np.sum(pr.Q_m3s)
        )
        self._n_tiles = len(self.layout.tiles)

        # White-space grid geometry (needed for observation_space; the solver
        # itself is built in reset()).
        grid = whitespace_grid(self.layout, cells_per_tile)
        gnx, gny, gnz = grid.nx, grid.ny, grid.nz

        # Initialize the PDEEnv3D interface with layout geometry, then reconfigure
        # for Option B: we do not keep the full (nt, ...) history buffer, and the
        # observation is a sensor vector, not the base full-field Box.
        X = self.layout.room_tiles[0] * self.layout.tile_m
        Y = self.layout.room_tiles[1] * self.layout.tile_m
        Z = self.layout.height_m
        super().__init__(
            T=float(episode_steps),
            dt=1.0,
            X=X,
            dx=self.layout.tile_m,
            Y=Y,
            dy=self.layout.tile_m,
            Z=Z,
            dz=Z,
            action_dim=2,
            reward_class=reward_class,
            normalize=False,
            state_dim=4,
        )
        self.U = None  # Option B holds the live solver, not a full time history.

        self.n_racks = len(self.layout.racks)
        if sensing == "full":
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(gnx, gny, gnz, 4), dtype=np.float32
            )
        elif sensing == "rack_inlet":
            # [per-rack inlet T, supply flow, supply T, total IT load]
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(self.n_racks + 3,), dtype=np.float32
            )
        else:
            raise ValueError("sensing must be 'rack_inlet' or 'full'")

        # Default reward normalized to this layout.
        if reward_class is None:
            self.reward_class = DataCenterReward(
                q_nom_m3s=self.Q_nom_m3s, p_it_nom_kW=self.p_it_nom_kW, nt=episode_steps
            )

        self._grid = None
        self._solver = None

    # ---- action / coupling helpers -------------------------------------------------

    def _denormalize(self, action):
        """Map an action in [-1, 1]^2 to (supply_flow_m3s, supply_T_C)."""
        a = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        Q = self.flow_bounds[0] + 0.5 * (a[0] + 1.0) * (
            self.flow_bounds[1] - self.flow_bounds[0]
        )
        T = self.temp_bounds[0] + 0.5 * (a[1] + 1.0) * (
            self.temp_bounds[1] - self.temp_bounds[0]
        )
        return float(Q), float(T)

    def _tile_flows(self, Q_sup_m3s):
        """Per-tile flow [m^3/s] for a supply flow, keyed like build_whitespace."""
        if self.plenum_mode == "full":
            pr = run_plenum(
                self.layout,
                self.cells_per_tile,
                self.layout.plenum_depth_m,
                self.plenum_beta,
                Q_sup_m3s,
                max_steps=2000,
            )
            q = np.asarray(pr.Q_m3s, dtype=float)
        else:  # "scaled"
            q = self._tile_fractions * Q_sup_m3s
        return {i: float(q[i]) for i in range(self._n_tiles)}

    def _apply_action(self, Q_sup, T_sup):
        """Update the live white-space solver's boundary conditions + tile body
        force for a new (supply flow, supply temperature) without rebuilding it."""
        solver, grid = self._solver, self._grid
        tile_flows = self._tile_flows(Q_sup)
        A_tile = self.layout.tile_m**2
        h = grid.z.dc[1]

        # Tile inlets (zlo): prescribed upward velocity per tile + supply temp.
        vel_map = solver.bc["zlo"].vel_map
        for i, _ in enumerate(self.layout.tiles):
            vel_map[self._tile_index == i] = tile_flows[i] / A_tile
        solver.bc["zlo"].temp_map[:] = T_sup

        # Ceiling return (zhi): uniform outflow that balances the supply.
        W_out = Q_sup / (self._n_ceil * A_tile)
        solver.bc["zhi"].vel_map[:] = W_out

        # Han Eq. 10 tile body force in the cell above each tile.
        src = np.zeros((grid.nx, grid.ny, grid.nz + 1))
        for i, tile in enumerate(self.layout.tiles):
            F = body_force(tile_flows[i], A_tile, h, tile.open_area)
            src[:, :, 1][self._tile_index == i] = F
        solver.w_source_extra = src

        self._U_ref = max(1e-9, max(tile_flows.values()) / A_tile)
        self.cur_Q, self.cur_T = Q_sup, T_sup

    def _march_to_steady(self):
        """March the white space to steady state (bounded by max_solve_steps)."""
        solver = self._solver
        u_prev = solver.u.copy()
        T_prev = solver.T.copy()
        converged = False
        step = 0
        for step in range(1, self.max_solve_steps + 1):
            solver.step()
            if step % self.check_every == 0:
                du = np.abs(solver.u - u_prev).max() / self._U_ref / self.check_every
                dT = np.abs(solver.T - T_prev).max() / 14.0 / self.check_every
                if (
                    du < self.solve_tol
                    and dT < self.solve_tol
                    and step >= self.min_solve_steps
                ):
                    converged = True
                    break
                u_prev = solver.u.copy()
                T_prev = solver.T.copy()
        return converged, step

    # ---- observation ---------------------------------------------------------------

    def _rack_inlet_temps(self):
        return np.array(
            [
                metrics.rack_inlet_temperature(self._solver, self._grid, self.layout, r)
                for r in self.layout.racks
            ],
            dtype=float,
        )

    def _get_obs(self):
        if self.sensing == "full":
            Uc, Vc, Wc = self._solver.velocity_at_centers()
            obs = np.stack([Uc, Vc, Wc, self._solver.T], axis=-1).astype(np.float32)
        else:
            obs = np.concatenate(
                [
                    self._rack_inlet_temps(),
                    [self.cur_Q, self.cur_T, self._p_it_kW],
                ]
            ).astype(np.float32)
        if self.sensing_noise_func is not None:
            obs = self.sensing_noise_func(obs)
        return obs

    def _current_powers(self, step_index):
        if self.load_profile is None:
            return self._base_powers, self.p_it_nom_kW
        total_kW = float(self.load_profile(step_index))
        scale = total_kW / self.p_it_nom_kW if self.p_it_nom_kW else 1.0
        return {k: v * scale for k, v in self._base_powers.items()}, total_kW

    def _apply_load(self, powers):
        """Push a new IT-load distribution into the live (warm-started) solver.

        A rack's airflow is power-proportional (Han Eq. 11) and its exhaust rise
        is ``P/(rho cp Q)`` (Eq. 12), so a time-varying load must update both the
        power and the intake flow in place; otherwise a warm-started step keeps
        the field frozen at reset's load and only the observation/reward scalar
        moves. (The cold-restart path rebuilds the solver from ``powers`` and so
        needs no in-place update.)"""
        powers_W = [powers[r.id] * 1000.0 for r in self._powered_racks]
        flows = [rack_flow_m3s(powers[r.id]) for r in self._powered_racks]
        self._solver.set_rack_powers(powers_W, flows)

    # ---- gym API -------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super(PDEEnv3D, self).reset(seed=seed)
        self.time_index = 0
        self.control_step = 0

        powers, self._p_it_kW = self._current_powers(0)
        Q_sup, T_sup = self.Q_nom_m3s, self.layout.supply_T_C
        tile_flows = self._tile_flows(Q_sup)

        self._grid, self._solver = build_whitespace(
            self.layout, self.cells_per_tile, tile_flows, T_sup, Q_sup, powers
        )
        self._tile_index = tile_cell_index(self._grid, self.layout)
        self._n_ceil = len(self.layout.ceiling_tiles)
        A_tile = self.layout.tile_m**2
        self._U_ref = max(1e-9, max(tile_flows.values()) / A_tile)
        self.cur_Q, self.cur_T = Q_sup, T_sup

        converged, nsteps = self._march_to_steady()
        obs = self._get_obs()
        info = {"converged": converged, "solve_steps": nsteps}
        return obs, info

    def step(self, action):
        Q_sup, T_sup = self._denormalize(action)
        self.control_step += 1
        self.time_index = self.control_step

        powers, self._p_it_kW = self._current_powers(self.control_step)
        if not self.warm_start:
            # Cold restart: rebuild from a uniform field each step.
            self._grid, self._solver = build_whitespace(
                self.layout,
                self.cells_per_tile,
                self._tile_flows(Q_sup),
                T_sup,
                Q_sup,
                powers,
            )
            self._tile_index = tile_cell_index(self._grid, self.layout)
        elif self.load_profile is not None:
            # Warm start: the solver is not rebuilt, so push the new load in place.
            self._apply_load(powers)

        self._apply_action(Q_sup, T_sup)
        converged, nsteps = self._march_to_steady()

        rack_T = self._rack_inlet_temps()
        terminate = self.control_step >= self.episode_steps
        truncate = False
        reward = self.reward_class.reward(
            rack_inlet_temps=rack_T,
            action=np.array([Q_sup, T_sup]),
            p_it_kW=self._p_it_kW,
            terminate=terminate,
            truncate=truncate,
            time_index=self.control_step,
        )

        obs = self._get_obs()
        info = {
            "converged": converged,
            "solve_steps": nsteps,
            "Q_sup_m3s": Q_sup,
            "T_sup_C": T_sup,
            "p_it_kW": self._p_it_kW,
            "t_max_in_C": float(metrics.t_max_in(rack_T)),
            "rci_hi": float(metrics.rci_hi(rack_T)),
        }
        return obs, reward, terminate, truncate, info
