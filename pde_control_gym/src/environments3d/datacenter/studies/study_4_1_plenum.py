"""D8: Study 4.1 -- plenum depth x tile open-area-ratio parametric sweep
(25 plenum runs on `han_reference`), reproducing Han et al. (2021) Fig. 12
(plan milestone D8).

Fixed: supply flow Q_sup = 1.5e5 m3/h (Han's Section 4.1 value -- distinct
from the AR-based Q_sup `run_baseline.py` uses for the whitespace baseline).
Swept: plenum depth in {305,457,610,762,914} mm x tile open-area ratio beta
in {0.15,0.25,0.35,0.45,0.56} (Han's Table 4), at the D7-adopted cpt=2
(1-ft) mesh, AMG pressure solver (`build_plenum`'s default).

Writes one row per run to `results/dc/study41.csv`
(`depth,beta,max,min,std,steps,converged,seconds`, depth in m) and skips rows
already present so a partial/interrupted sweep resumes; then calls
`plots.fig12()` and `verify.report_agreement()` for every (depth, series)
against the digitized `data/han_dc/fig12_*` curves -- added after the
original D8 landed with only three hand-picked verification points and a
user-requested full-grid check afterwards found a systematic (not noise-like)
disagreement the spot-check missed entirely (median 18.9%, mean 21.2%, max
57.7% relative error). See `datacenter/verify.py` for why this is a standing
check now rather than an on-request one.

Run: ../../.venv/bin/python -m datacenter.studies.study_4_1_plenum
"""
from __future__ import annotations

import csv
import os
import time

from ..layout import load_layout
from ..plenum import run_plenum
from ..units import m3h_to_m3s
from .. import plots, verify

LAYOUT_PATH = os.path.join(os.path.dirname(__file__), "..", "layouts", "han_reference")
CELLS_PER_TILE = 2
Q_SUP_M3H = 1.5e5
DEPTHS_MM = [305, 457, 610, 762, 914]
BETAS = [0.15, 0.25, 0.35, 0.45, 0.56]
OUT_CSV = os.path.join("results", "dc", "study41.csv")
FIELDS = ["depth", "beta", "max", "min", "std", "steps", "converged", "seconds"]


def _existing_rows():
    if not os.path.exists(OUT_CSV):
        return {}
    with open(OUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    return {(round(float(r["depth"]), 6), round(float(r["beta"]), 6)): r for r in rows}


def _append_row(row):
    is_new = not os.path.exists(OUT_CSV)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def main():
    layout = load_layout(LAYOUT_PATH)
    layout.summary()
    Q_sup_m3s = m3h_to_m3s(Q_SUP_M3H)

    existing = _existing_rows()
    print(f"\nStudy 4.1: {len(DEPTHS_MM)}x{len(BETAS)} = "
          f"{len(DEPTHS_MM) * len(BETAS)} runs, {len(existing)} already done")

    for depth_mm in DEPTHS_MM:
        depth_m = depth_mm / 1000.0
        for beta in BETAS:
            key = (round(depth_m, 6), round(beta, 6))
            if key in existing:
                print(f"skip depth={depth_mm}mm beta={beta:.2f} (already in {OUT_CSV})")
                continue
            t0 = time.perf_counter()
            result = run_plenum(layout, CELLS_PER_TILE, depth_m, beta, Q_sup_m3s)
            seconds = time.perf_counter() - t0
            _append_row({
                "depth": depth_m, "beta": beta,
                "max": result.prd_max, "min": result.prd_min, "std": result.prd_std,
                "steps": result.steps, "converged": result.converged,
                "seconds": f"{seconds:.1f}",
            })

    print("\nAll runs done; rendering Fig 12.")
    path = plots.fig12(OUT_CSV)
    print(f"wrote {path}")

    print("\nAgreement vs digitized Han curves (data/han_dc/fig12_*):")
    with open(OUT_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for depth_mm in DEPTHS_MM:
        depth_m = depth_mm / 1000.0
        by_beta = {round(float(r["beta"]) * 100): r for r in rows
                   if round(float(r["depth"]), 6) == round(depth_m, 6)}
        for series in ("max", "min", "std"):
            sim = {beta_pct: float(r[series]) for beta_pct, r in by_beta.items()}
            verify.report_agreement(f"depth={depth_mm}mm {series}", sim,
                                     f"fig12_{series}_d{depth_mm}")


if __name__ == "__main__":
    main()
