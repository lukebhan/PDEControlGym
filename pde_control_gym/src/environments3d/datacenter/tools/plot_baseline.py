"""Floor-plan views of the D7 baseline run (han_reference, AR = 2.07, T_sup = 22 C).

`run_baseline.py` prints its report to stdout and leaves behind only the two
per-rack inlet-temperature CSVs (`ws_han_reference_cpt{2,4}_*_racks.csv`) --
the `.npz` field snapshots are gitignored. Those CSVs still carry the whole
white-space result at rack level, so this renders them as two floor plans:

  left   rack-inlet temperature on Han's own 6-in mesh (cells_per_tile=4)
  right  1-ft minus 6-in, i.e. the D7 grid check laid out spatially

Writes `results/dc/baseline_floorplan.png`.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from ..layout import load_layout
from ..plots import OUT_DIR, SURFACE, INK, MUTED, _plt, _savefig

LAYOUT = os.path.join("datacenter", "layouts", "han_reference")
CSV_FINE = os.path.join(OUT_DIR, "ws_han_reference_cpt4_q150961_t22_racks.csv")
CSV_COARSE = os.path.join(OUT_DIR, "ws_han_reference_cpt2_q150961_t22_racks.csv")
N_LABEL = 5  # how many extreme racks to annotate per panel


def _read(path):
    """rack id -> mean inlet temperature [C], powered racks only."""
    with open(path, newline="") as f:
        return {r["id"]: float(r["T_inlet_mean"])
                for r in csv.DictReader(f) if int(r["empty"]) == 0}


def _base(ax, layout, title):
    """Draw the room outline, perforated tiles and unpowered solids."""
    nx, ny = layout.room_tiles
    for t in layout.tiles:
        ax.add_patch(plt.Rectangle((t.tile_ix, t.tile_iy), 1, 1,
                                   facecolor="#cfe6ef", edgecolor="#9dc6d4", lw=.3, zorder=1))
    for b in layout.blocks:
        ax.add_patch(plt.Rectangle((b.tile_ix, b.tile_iy), 1, 1,
                                   facecolor="#e4e2dd", edgecolor=MUTED, lw=.3, zorder=1))
    for r in layout.racks:
        if r.empty:
            ax.add_patch(plt.Rectangle((r.tile_ix, r.tile_iy), 1, 1, facecolor="none",
                                       edgecolor=MUTED, lw=.6, hatch="///", zorder=3))
    ax.set_xlim(-1, nx + 1)
    ax.set_ylim(-1, ny + 5)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=11, pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(MUTED)
        sp.set_linewidth(.8)
    ax.add_patch(plt.Rectangle((0, 0), nx, ny, facecolor="none",
                               edgecolor=INK, lw=1.2, zorder=4))


def _racks(ax, layout, values, cmap, vmin, vmax, label_extreme="max"):
    """Colour each powered rack by `values[id]`; annotate the N_LABEL extremes."""
    norm = plt.Normalize(vmin, vmax)
    cm = plt.get_cmap(cmap)
    for r in layout.racks:
        if r.empty or r.id not in values:
            continue
        ax.add_patch(plt.Rectangle((r.tile_ix, r.tile_iy), 1, 1,
                                   facecolor=cm(norm(values[r.id])),
                                   edgecolor="#ffffff", lw=.25, zorder=2))
    pos = {r.id: (r.tile_ix, r.tile_iy) for r in layout.racks}
    key = (lambda i: -values[i]) if label_extreme == "max" else (lambda i: -abs(values[i]))
    # Fan the callouts out along the top edge so leaders never overlap: the
    # extremes cluster in a couple of rack rows, so in-place offsets collide.
    picked = sorted(values, key=key)[:N_LABEL]
    ny = layout.room_tiles[1]
    slots = np.linspace(.12, .88, len(picked)) * layout.room_tiles[0]
    for rid, sx in zip(picked, slots):
        x, y = pos[rid]
        ax.annotate(f"{rid} {values[rid]:+.2f}" if label_extreme == "abs"
                    else f"{rid} {values[rid]:.1f}",
                    (x + .5, y + .5), (sx, ny + 3.0),
                    color=INK, fontsize=7.5, ha="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=.55,
                                    shrinkA=0, shrinkB=2, alpha=.75),
                    bbox=dict(boxstyle="round,pad=.2", fc=SURFACE, ec=MUTED, lw=.4))
    return plt.cm.ScalarMappable(norm=norm, cmap=cm)


def baseline_floorplan(fine=CSV_FINE, coarse=CSV_COARSE):
    global plt
    plt = _plt()
    layout = load_layout(LAYOUT)
    T4, T2 = _read(fine), _read(coarse)
    ids = sorted(set(T4) & set(T2))
    diff = {i: T2[i] - T4[i] for i in ids}
    d = np.array([diff[i] for i in ids])
    rms, mx = float(np.sqrt((d ** 2).mean())), float(np.abs(d).max())

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), facecolor=SURFACE)

    vals = np.array([T4[i] for i in ids])
    _base(axes[0], layout, "Rack-inlet temperature, 6-in mesh (Han's own resolution)")
    sm = _racks(axes[0], layout, T4, "inferno_r", 22.0, float(vals.max()))
    cb = fig.colorbar(sm, ax=axes[0], fraction=.03, pad=.02)
    cb.set_label("T_inlet  [C]", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_xlabel(f"T_sup 22 C   mean {vals.mean():.2f}   max {vals.max():.2f} C"
                       f"     RCI_HI 1.000  RCI_LO 1.000",
                       color=MUTED, fontsize=9, labelpad=6)

    lim = float(np.abs(d).max())
    _base(axes[1], layout, "Grid check: 1-ft minus 6-in mesh")
    sm2 = _racks(axes[1], layout, diff, "RdBu_r", -lim, lim, label_extreme="abs")
    cb2 = fig.colorbar(sm2, ax=axes[1], fraction=.03, pad=.02)
    cb2.set_label("dT  [C]", color=MUTED, fontsize=9)
    cb2.ax.tick_params(colors=MUTED, labelsize=8)
    axes[1].set_xlabel(f"RMS {rms:.3f} C   max {mx:.3f} C   bias {d.mean():+.3f} C"
                       f"     (gate: RMS <= 0.7 C)",
                       color=MUTED, fontsize=9, labelpad=6)

    fig.suptitle("D7 baseline -- han_reference, AR 2.07, 149 powered racks",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, .02, 1, .95))
    return _savefig(fig, "baseline_floorplan.png")


if __name__ == "__main__":
    print(baseline_floorplan())
