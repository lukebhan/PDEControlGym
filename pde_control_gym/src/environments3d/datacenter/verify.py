"""Full-grid quantitative agreement check between a study script's simulated
output and Han et al. (2021)'s digitized figure curves (added after plan
milestone D8, as a standing check for every study script from here on).

D8 (Study 4.1 / Fig 12) originally shipped verified against only the three
points the plan's own Section 6 names explicitly: the two design-candidate
combinations (checked against Han's *fixed absolute* design thresholds, not
his curve) and one reference point (914 mm/25 %, reported as "within 2
percentage points" without also computing relative error). A full-grid check
run afterwards found a systematic disagreement the three-point check missed
entirely: median 18.9 %, mean 21.2 %, max 57.7 % relative error against the
digitized curves, with a consistent (not noise-like) sign pattern across all
five plenum depths -- plausibly the plenum-footprint guess the plan itself
already flagged ("revisit if Fig 12 uniformity is off"). Every study script
(`study_4_1_plenum.py`, and D9's `study_4_2_cooling.py`) should call
`report_agreement` once per digitized series after writing its CSV, so a gap
like this prints immediately instead of only surfacing on request.
"""
from __future__ import annotations

import csv
import os
import statistics

HAN_DIR = os.path.join("data", "han_dc")


def load_han_curve(name):
    """{x: y} from data/han_dc/<name>.csv, or None if not digitized yet."""
    path = os.path.join(HAN_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[round(float(row["x"]), 6)] = float(row["y"])
    return out


def relative_error_stats(sim, han, skip_below=0.1):
    """Pointwise |sim-han|/|han| stats (percent) at the x values common to both.

    `sim`/`han` are {x: y} dicts; callers must round x consistently so keys
    match exactly (e.g. beta as a whole percent, not a fraction). Points
    where |han| < `skip_below` are excluded -- the relative error there is
    dominated by digitization noise on a near-zero denominator, not a real
    disagreement (e.g. Fig 12's 914 mm/15 % point: both max and min digitize
    to ~0, so a ~0.3-percentage-point absolute difference reads as a
    meaningless 1000%+ relative error).
    """
    rels, worst = [], None
    for x, hv in han.items():
        if x not in sim or abs(hv) < skip_below:
            continue
        sv = sim[x]
        rel = abs(sv - hv) / abs(hv) * 100.0
        rels.append(rel)
        if worst is None or rel > worst[0]:
            worst = (rel, x, sv, hv)
    if not rels:
        return None
    return {
        "n": len(rels),
        "median": statistics.median(rels),
        "mean": statistics.mean(rels),
        "max": max(rels),
        "frac_over_20": sum(r > 20 for r in rels) / len(rels),
        "frac_over_30": sum(r > 30 for r in rels) / len(rels),
        "worst_x": worst[1], "worst_sim": worst[2], "worst_han": worst[3],
    }


def report_agreement(label, sim, han_name, warn_median=15.0, warn_max=40.0):
    """Print a one-line summary of `sim` ({x: y}) vs
    `data/han_dc/<han_name>.csv`, with a loud WARNING (not a quiet pass/fail)
    if median or max relative error exceeds the given thresholds -- a
    systematic-but-plausible-looking figure is exactly what D8's spot-check
    missed. Returns the stats dict, or None if there's no digitized curve or
    no overlapping x values."""
    han = load_han_curve(han_name)
    if han is None:
        print(f"  [{label}] no digitized curve at data/han_dc/{han_name}.csv -- skipped")
        return None
    stats = relative_error_stats(sim, han)
    if stats is None:
        print(f"  [{label}] digitized curve has no overlapping x with sim -- skipped")
        return None
    flag = ("  <-- WARNING: exceeds tolerance"
             if (stats["median"] > warn_median or stats["max"] > warn_max) else "")
    print(f"  [{label}] n={stats['n']} median={stats['median']:.1f}% "
          f"mean={stats['mean']:.1f}% max={stats['max']:.1f}% "
          f"(worst: x={stats['worst_x']:g} sim={stats['worst_sim']:.2f} "
          f"han={stats['worst_han']:.2f}){flag}")
    return stats
