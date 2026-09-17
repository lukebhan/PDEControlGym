"""Fig 12/13/14/15/16 reproductions from the data-center studies (plan
milestone D6).

Consumes CSVs the D8/D9 study scripts write (`results/dc/study41.csv`,
`study42.csv`) and, for `fig14`, the per-run `results/dc/ws_*.npz` snapshots
`whitespace.run_whitespace` saves; none of that data exists yet as of D6, so
these functions cannot be exercised end-to-end until D8/D9 run. Every function
is defensive about a missing input file/column and about a missing digitized
overlay (skipped, not an error) so it is safe to call as soon as partial data
appears. Column-name contracts assumed here (not yet fixed by an actual study
script) are documented per function -- flag to the user if a later milestone
picks something different.

Overlays Han's digitized curves from `data/han_dc/<name>.csv` (columns
`x,y`) where present. Every figure is written to `results/dc/fig<N>.png` and
the path is returned. Same palette as `plot_profiles.py`/`m4_nrmsd.py`.
"""
from __future__ import annotations

import csv
import os

import numpy as np

OUT_DIR = os.path.join("results", "dc")
HAN_DIR = os.path.join("data", "han_dc")

C_OURS, C_HAN = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(alpha=.25, lw=.6, color=MUTED, ls=":")
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _han_curve(name):
    """(x, y) arrays from `data/han_dc/<name>.csv`, or None if not digitized yet."""
    path = os.path.join(HAN_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    rows = _read_csv(path)
    return (np.array([float(r["x"]) for r in rows]),
            np.array([float(r["y"]) for r in rows]))


def _savefig(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    return path


def fig12(study41_csv=None):
    """Fig 12 (study 4.1): PRD_m max/min (left panel) and std (right panel) vs.
    tile open-area ratio beta, one line per plenum depth -- reproducing Han's
    actual two-panel layout (checked against Han21.pdf page 11 directly, not
    guessed: x-axis is beta in %, lines are depth, not the other way round).
    Expects `study41.csv` columns `depth,beta,max,min,std` (m, dimensionless,
    and Eq. 16 percent -- `study_4_1_plenum.py`'s row shape per plan section
    4.10). Overlays `data/han_dc/fig12_{max,min,std}_d<depth_mm>.csv` where
    digitized (`datacenter/tools/digitize_fig12.py`)."""
    plt = _plt()
    study41_csv = study41_csv or os.path.join(OUT_DIR, "study41.csv")
    rows = _read_csv(study41_csv)
    depths = sorted({float(r["depth"]) for r in rows})

    fig, (ax_mm, ax_std) = plt.subplots(1, 2, figsize=(10, 4.5))
    for depth in depths:
        depth_mm = round(depth * 1000)
        pts = sorted((float(r["beta"]), float(r["max"]), float(r["min"]))
                     for r in rows if float(r["depth"]) == depth)
        betas = np.array([p[0] for p in pts]) * 100
        alpha = 0.4 + 0.6 * (depth - depths[0]) / max(1e-9, depths[-1] - depths[0])
        ax_mm.plot(betas, [p[1] for p in pts], "-o", ms=4, lw=1.6, color=C_OURS,
                   alpha=alpha, label=f"{depth_mm} mm")
        ax_mm.plot(betas, [p[2] for p in pts], "-o", ms=4, lw=1.6, color=C_OURS, alpha=alpha)
        for series in ("max", "min"):
            han = _han_curve(f"fig12_{series}_d{depth_mm}")
            if han is not None:
                ax_mm.plot(han[0], han[1], "--", color=C_HAN, lw=1.2, alpha=alpha)

        std_pts = sorted((float(r["beta"]) * 100, float(r["std"]))
                          for r in rows if float(r["depth"]) == depth)
        ax_std.plot([p[0] for p in std_pts], [p[1] for p in std_pts], "-o", ms=4,
                    lw=1.6, color=C_OURS, alpha=alpha, label=f"{depth_mm} mm")
        han_std = _han_curve(f"fig12_std_d{depth_mm}")
        if han_std is not None:
            ax_std.plot(han_std[0], han_std[1], "--", color=C_HAN, lw=1.2, alpha=alpha)

    for ax, title in ((ax_mm, "Max. and Min. Values"), (ax_std, "Standard Deviations")):
        ax.set_xlabel("Tile Open Area Ratio [%]", fontsize=9, color=INK)
        ax.set_ylabel("PRD_m [%]", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    ax_std.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED,
                  title="depth (ours; Han dashed)", title_fontsize=7)
    fig.suptitle("Fig 12: tile-flow uniformity vs. open area / plenum depth", color=INK)
    fig.tight_layout()
    return _savefig(fig, "fig12.png")


def fig13(study42_csv=None):
    """Fig 13 (study 4.2): RCI_HI, RCI_LO, T_max_in vs. air ratio AR, one line
    per supply temperature. Expects `study42.csv` columns
    `AR,T_sup,RCI_HI,RCI_LO,T_max_in` (plan section 4.10)."""
    plt = _plt()
    study42_csv = study42_csv or os.path.join(OUT_DIR, "study42.csv")
    rows = _read_csv(study42_csv)
    t_sups = sorted({float(r["T_sup"]) for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for metric, ax, title in zip(("RCI_HI", "RCI_LO", "T_max_in"), axes,
                                  ("RCI_HI", "RCI_LO", "T_max_in [C]")):
        for t_sup in t_sups:
            pts = sorted((float(r["AR"]), float(r[metric]))
                         for r in rows if float(r["T_sup"]) == t_sup)
            xs, ys = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
            ax.plot(xs, ys, "-o", ms=4, lw=1.6, label=f"T_sup={t_sup:g} C")
        han = _han_curve(f"fig13_{metric}")
        if han is not None:
            ax.plot(han[0], han[1], "--", color=C_HAN, lw=1.4, label="Han (digitized)")
        ax.set_xlabel("air ratio AR", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        _style(ax)
    axes[-1].legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED)
    fig.suptitle("Fig 13: cooling performance vs. air ratio / supply temperature", color=INK)
    fig.tight_layout()
    return _savefig(fig, "fig13.png")


def fig14(layout_name, profile, ARs=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75), T_sup_C=16.0,
          Q_it_m3h=72928.0, rack_height_m=1.9812, plane_frac=2.0 / 3.0):
    """Fig 14: T field at z = plane_frac*rack_height, one panel per AR, at a
    fixed T_sup. Reads the `ws_<layout>_<profile>_q<round(AR*Q_it)>_t<T_sup>.npz`
    snapshot `whitespace.run_whitespace` saves for each (AR, T_sup); a missing
    run is shown as a blank "no data" panel rather than failing."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(ARs), figsize=(3.2 * len(ARs), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    vmin, vmax = T_sup_C, T_sup_C + 20.0
    im = None
    for ax, ar in zip(axes, ARs):
        Q_m3h = ar * Q_it_m3h
        fname = f"ws_{layout_name}_{profile}_q{round(Q_m3h):d}_t{round(T_sup_C):d}.npz"
        path = os.path.join(OUT_DIR, fname)
        ax.set_title(f"AR={ar:g}", fontsize=9, color=INK)
        _style(ax)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED, fontsize=8)
            continue
        data = np.load(path)
        zc = data["zc"]
        k = int(np.argmin(np.abs(zc - plane_frac * rack_height_m)))
        T_plane = data["T"][:, :, k]
        im = ax.pcolormesh(data["xc"], data["yc"], T_plane.T, vmin=vmin, vmax=vmax,
                            cmap="inferno", shading="nearest")
    if im is not None:
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="T [C]")
    fig.suptitle(f"Fig 14: T at z={plane_frac:.2f}x rack height, T_sup={T_sup_C:g} C",
                 color=INK)
    return _savefig(fig, "fig14.png")


def _bin_label(lo, hi):
    if np.isneginf(lo):
        return f"<{hi:g}"
    if np.isposinf(hi):
        return f">{lo:g}"
    return f"{lo:g}-{hi:g}"


def _fig15_panel(ax, rows, ars, edges, col_prefix, title):
    """Grouped bar chart: x = temperature-range bin, one bar group per AR --
    Han's actual Fig 15 layout (not bins-stacked-per-AR)."""
    labels = [_bin_label(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    n_bins = len(labels)
    x = np.arange(n_bins)
    width = 0.8 / max(1, len(ars))
    for j, ar in enumerate(ars):
        row = next((r for r in rows if abs(float(r["AR"]) - ar) < 1e-9), None)
        if row is None:
            continue
        counts = [float(row[f"{col_prefix}{i}"]) for i in range(n_bins)]
        ax.bar(x + (j - (len(ars) - 1) / 2) * width, counts, width=width, label=f"AR={ar:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("rack inlet temperature range [C]", fontsize=9, color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=MUTED)
    _style(ax)


def fig15(study42_csv=None, t_sup_C=16.0, hot_ars=(0.5, 0.75, 1.0), cold_ars=(1.0, 1.25, 1.5)):
    """Fig 15: two panels, reproducing Han's own bin scheme exactly rather than
    a single merged histogram -- **hot side** (overheating risk: `metrics.
    HOT_BIN_EDGES`, 1 C bins from 27-32 C, columns `n_hot_0..n_hot_6`) plotted
    against `hot_ars`, and **cold side** (overcooling risk: `metrics.
    cold_bin_edges(t_sup_C)`, 0.5 C bins from T_sup to T_sup+2, columns
    `n_cold_0..n_cold_4`) plotted against `cold_ars` -- matching the paper's own
    choice of AR ∈ {0.5,0.75,1.0} for the hot panel and {1.0,1.25,1.5} for the
    cold one (the AR ranges where each effect is visible; a study script is
    free to compute more ARs than either panel shows). `study42.csv` must
    carry both bin-count column sets; see `metrics.HOT_BIN_EDGES`/
    `metrics.cold_bin_edges` for the exact edges a study script should feed to
    `metrics.rack_count_by_range` to produce them."""
    from .metrics import HOT_BIN_EDGES, cold_bin_edges
    plt = _plt()
    study42_csv = study42_csv or os.path.join(OUT_DIR, "study42.csv")
    rows = [r for r in _read_csv(study42_csv) if abs(float(r["T_sup"]) - t_sup_C) < 1e-9]

    fig, (ax_hot, ax_cold) = plt.subplots(1, 2, figsize=(12, 4.5))
    _fig15_panel(ax_hot, rows, hot_ars, HOT_BIN_EDGES, "n_hot_",
                 "Hot side (overheating risk)")
    _fig15_panel(ax_cold, rows, cold_ars, cold_bin_edges(t_sup_C), "n_cold_",
                 "Cold side (overcooling risk)")
    ax_hot.set_ylabel("number of racks", fontsize=9, color=INK)
    fig.suptitle(f"Fig 15: rack inlet-temperature distribution, T_sup={t_sup_C:g} C",
                 color=INK)
    fig.tight_layout()
    return _savefig(fig, "fig15.png")


def fig16(study42_csv=None, criteria=None):
    """Fig 16: table of (AR, T_sup) pairs meeting Table 6's candidate criteria.
    `criteria(row: dict) -> bool` defaults to Han's headline gate, RCI_HI == 1
    and RCI_LO == 1; `study_4_3_candidates.py` (D10) is expected to pass its
    own, more complete Table 6 filter. Rendered as a table image, not a chart,
    since Fig 16 in the paper is itself a results table."""
    plt = _plt()
    study42_csv = study42_csv or os.path.join(OUT_DIR, "study42.csv")
    rows = _read_csv(study42_csv)
    if criteria is None:
        def criteria(r):
            return float(r["RCI_HI"]) >= 1.0 and float(r["RCI_LO"]) >= 1.0
    admissible = [(r["AR"], r["T_sup"]) for r in rows if criteria(r)]

    fig, ax = plt.subplots(figsize=(4, max(1.5, 0.35 * len(admissible) + 1)))
    ax.axis("off")
    ax.set_title("Fig 16: admissible (AR, T_sup) candidates", fontsize=10, color=INK)
    tbl = ax.table(cellText=[["AR", "T_sup [C]"]] + admissible,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    fig.tight_layout()
    return _savefig(fig, "fig16.png")
