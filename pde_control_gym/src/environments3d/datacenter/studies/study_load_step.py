"""Control-env check: a time-varying IT load must reach the thermal field.

`DataCenter3D` accepts a ``load_profile`` callable ``step_index ->
total_IT_power_kW``. Under the default ``warm_start=True`` the white-space
solver is *not* rebuilt between control steps, so a load change only reaches the
physics if the live solver's flow-through racks are updated in place. This script
exercises that path end-to-end on the fast ``mini`` layout and asserts the
thermal response is real, not just a moving observation/reward scalar.

Setup: supply flow and temperature are held fixed (a constant neutral action)
so the *only* thing changing between steps is the IT load. The load starts low
(reset + first step) and then triples. Because the racks reject their heat into
a fixed supply flow, the room-average temperature rise scales as
``P_total / (rho cp Q_sup)`` (Han Eqs. 11-12: rack airflow is power-proportional
and each rack's own rise is fixed, so the load shows up as room/inlet heating,
not as a bigger per-rack delta). Tripling the load must therefore raise the
rack-inlet temperatures by several kelvin and hold them there.

Run: python -m pde_control_gym.src.environments3d.datacenter.studies.study_load_step
(from the repo root), or just ``python <this file>`` -- it puts the repo root on
sys.path itself so both work.
"""
from __future__ import annotations

import os
import sys
import time

# Make the top-level `pde_control_gym` package importable when run as a script.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *([os.pardir] * 5)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from pde_control_gym.src.environments3d.datacenter3d import DataCenter3D

LOW_kW, HIGH_kW = 8.0, 24.0


def load_profile(step_index):
    """Low during reset (0) and the first control step (1); triples from step 2."""
    return LOW_kW if step_index < 2 else HIGH_kW


def main():
    t0 = time.perf_counter()
    env = DataCenter3D(
        layout="mini", cells_per_tile=2, warm_start=True, plenum_mode="scaled",
        load_profile=load_profile, max_solve_steps=1500, min_solve_steps=200,
        check_every=100, episode_steps=5)
    _, info = env.reset(seed=0)
    print(f"reset: converged={info['converged']} steps={info['solve_steps']} "
          f"({time.perf_counter() - t0:.1f}s)")

    # Fixed neutral action every step -> nominal supply flow, mid supply temp;
    # only the IT load varies between steps.
    action = np.array([0.0, 0.0], dtype=np.float32)
    rows = []
    for k in range(1, 5):
        _, reward, term, trunc, info = env.step(action)
        rows.append(info)
        print(f"step {k}: load={info['p_it_kW']:5.1f}kW  "
              f"t_max_in={info['t_max_in_C']:.3f}C  rci_hi={info['rci_hi']:.2f}  "
              f"solve_steps={info['solve_steps']} conv={info['converged']}")

    low, high = rows[0], rows[1]          # last LOW-load step, first HIGH-load step
    d_tmax = high["t_max_in_C"] - low["t_max_in_C"]
    print(f"\nload {low['p_it_kW']:.0f}->{high['p_it_kW']:.0f} kW  =>  "
          f"t_max_in {low['t_max_in_C']:.3f}->{high['t_max_in_C']:.3f}C "
          f"(delta {d_tmax:+.3f}C)")

    assert high["p_it_kW"] > low["p_it_kW"] + 1e-6, "load scalar did not increase"
    assert d_tmax > 0.2, (
        f"time-varying load did not reach the physics: t_max_in barely moved "
        f"({d_tmax:+.3f}C) when the load tripled")
    assert rows[2]["t_max_in_C"] >= high["t_max_in_C"] - 0.5, \
        "high-load temperature was not sustained after the load step"

    print("\nPASS: time-varying load propagates to the thermal field "
          f"(total {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
