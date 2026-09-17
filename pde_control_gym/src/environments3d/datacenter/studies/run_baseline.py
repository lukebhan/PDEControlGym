"""D7: baseline run of the han_reference layout at Han's baseline operating
point (AR = 2.07, T_sup = 22 C, beta = 0.25, plenum depth = 914 mm), at both
mesh resolutions -- the 1-ft sweep mesh (cells_per_tile=2, ~81k white-space
cells) and Han's own 6-in mesh (cells_per_tile=4, ~651k white-space cells) --
to record timing/memory and run the grid-check decision gate before the
D8/D9 parametric sweeps (plan milestone D7).
"""
from __future__ import annotations

import csv
import os
import resource
import time

import numpy as np

from ..layout import load_layout
from ..racks import assign_powers
from ..units import RACK_FLOW_M3H_PER_KW, m3h_to_m3s
from ..plenum import build_plenum, run_plenum
from ..whitespace import build_whitespace, run_whitespace

LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "layouts", "han_reference")
AR = 2.07
T_SUP_C = 22.0
BETA = 0.25
DEPTH_M = 0.9144
RMS_GATE_C = 0.7  # Han's ~5% PRD_T at the 14 C rack DeltaT reference


def peak_rss_mb():
    """Process peak RSS so far, in MB (Linux ru_maxrss is reported in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _read_rack_inlet_temps(path):
    """rack id -> T_inlet_mean [C] from a `metrics.rack_inlet_csv` file,
    excluding Han's empty (unpowered) racks G11/G13."""
    temps = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["empty"]) == 0:
                temps[row["id"]] = float(row["T_inlet_mean"])
    return temps


def run_profile(layout, cells_per_tile, powers_kW, Q_sup_m3s):
    profile = f"cpt{cells_per_tile}"
    nx, ny = layout.room_tiles
    print(f"\n=== {profile}: {nx * cells_per_tile}x{ny * cells_per_tile} tile grid ===")

    rss0 = peak_rss_mb()

    # `run_plenum`/`run_whitespace` rebuild the Grid/Solver internally rather
    # than accepting one, so AMG setup time (inside Solver.__init__) is timed
    # with a throwaway build call first, then folded into the real solve below.
    t0 = time.perf_counter()
    build_plenum(layout, cells_per_tile, DEPTH_M, BETA, Q_sup_m3s)
    t_setup_plenum = time.perf_counter() - t0

    t0 = time.perf_counter()
    plenum_result = run_plenum(layout, cells_per_tile, DEPTH_M, BETA, Q_sup_m3s)
    t_plenum = time.perf_counter() - t0
    print(f"  plenum:     setup {t_setup_plenum:5.1f}s, solve {t_plenum:7.1f}s "
          f"({plenum_result.steps:4d} steps, "
          f"{1000 * t_plenum / plenum_result.steps:6.1f} ms/step), "
          f"{'CONVERGED' if plenum_result.converged else 'NOT CONVERGED'}")

    tile_flows = {i: float(plenum_result.Q_m3s[i]) for i in range(len(layout.tiles))}

    t0 = time.perf_counter()
    build_whitespace(layout, cells_per_tile, tile_flows, T_SUP_C, Q_sup_m3s, powers_kW)
    t_setup_ws = time.perf_counter() - t0

    t0 = time.perf_counter()
    ws_result = run_whitespace(layout, cells_per_tile, tile_flows, T_SUP_C, Q_sup_m3s,
                                powers_kW, profile=profile)
    t_ws = time.perf_counter() - t0
    print(f"  whitespace: setup {t_setup_ws:5.1f}s, solve {t_ws:7.1f}s "
          f"({ws_result.steps:4d} steps, "
          f"{1000 * t_ws / ws_result.steps:6.1f} ms/step), "
          f"{'CONVERGED' if ws_result.converged else 'NOT CONVERGED'}")
    print(f"  peak RSS after {profile}: {peak_rss_mb():.0f} MB "
          f"(delta this profile: {peak_rss_mb() - rss0:+.0f} MB)")

    return plenum_result, ws_result


def main():
    layout = load_layout(LAYOUT_PATH)
    layout.summary()

    Q_IT_m3h = RACK_FLOW_M3H_PER_KW * layout.total_it_power_kW
    Q_sup_m3h = AR * Q_IT_m3h
    Q_sup_m3s = m3h_to_m3s(Q_sup_m3h)
    powers_kW = assign_powers(layout.racks, layout.total_it_power_kW, layout.power_mode)

    print(f"\nBaseline: AR={AR:g} (Q_IT={Q_IT_m3h:,.0f} m3/h, Q_sup={Q_sup_m3h:,.0f} m3/h "
          f"= {Q_sup_m3s:.3f} m3/s), T_sup={T_SUP_C:g} C, beta={BETA:g}, "
          f"depth={DEPTH_M * 1000:.0f} mm")

    t0 = time.perf_counter()
    _, ws2 = run_profile(layout, 2, powers_kW, Q_sup_m3s)
    _, ws4 = run_profile(layout, 4, powers_kW, Q_sup_m3s)
    wall_total = time.perf_counter() - t0

    T2 = _read_rack_inlet_temps(ws2.rack_csv_path)
    T4 = _read_rack_inlet_temps(ws4.rack_csv_path)
    ids = sorted(set(T2) & set(T4))
    missing = set(T2) ^ set(T4)
    if missing:
        print(f"\nWARNING: {len(missing)} rack ids present in only one mesh's CSV "
              f"(excluded from the comparison): {sorted(missing)}")

    diffs = np.array([T2[i] - T4[i] for i in ids])
    rms = float(np.sqrt(np.mean(diffs ** 2)))
    max_abs = float(np.max(np.abs(diffs)))

    print(f"\n=== Grid check: cpt=2 (1-ft) vs cpt=4 (Han's 6-in) rack-inlet "
          f"temperature, {len(ids)} racks ===")
    print(f"  RMS diff: {rms:.3f} C   max |diff|: {max_abs:.3f} C   "
          f"(gate: RMS <= {RMS_GATE_C} C)")
    if rms > RMS_GATE_C:
        print(f"  DECISION GATE TRIPPED: RMS {rms:.3f} C > {RMS_GATE_C} C -- "
              f"report to the user before running the D8/D9 sweeps at cpt=2.")
    else:
        print(f"  Grid check passed: cpt=2 (1-ft) is adequate for the D8/D9 sweeps.")

    print(f"\nTotal wall time: {wall_total:.1f}s   peak RSS: {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
