#!/usr/bin/env python3
"""Digitize Han et al. (2021) Fig. 12 -- tile-flow uniformity (PRD_m) vs.
plenum depth x tile open-area ratio (plan milestone D8).

Source: /home/natet/ffd/Han21.pdf, page 11 (0-indexed page 10).

Unlike Figs 6-8 (traced smooth curves, see `data/experimental/
digitize_han_curves.py`), Fig. 12 is two panels of five 5-point marker series
(one per plenum depth: 305/457/610/762/914 mm), each series sampled at the
same five tile open-area ratios (15/25/35/45/56 %):
  - left panel  "Max. and Min. Values": two branches per depth (max, min)
  - right panel "Standard Deviations":  one branch per depth (std)
This is a marker-centroid digitization, not a row-wise trace: each series is
picked out by its plot color (sampled from the right panel's own legend
swatches, so no color is hand-guessed) and clustered into points at the five
known x positions.

Method
------
1. Render the page at an 8x zoom (~576 DPI) cropped to the two-panel region;
   matplotlib/whatever tool made this figure draws the axis spines exactly at
   the data limits, so the frame corners (found as the long solid dark
   rows/columns) calibrate both panels with no tick-reading: left panel
   x in [10,60]% maps to columns [557.5,1816.0], y in [-20,30]% to rows
   [1948.5,715.5]; right panel shares the same x columns shifted to
   [2194.0,3453.0] and y in [0,10]% to the same row range.
2. Legend swatch colors (sampled once from the right panel's legend box) give
   an exact target RGB per depth: 305mm=black line/text (handled as
   dark-pixel, not color-distance, since black is the frame/text color too),
   457mm=olive, 610mm=magenta, 762mm=teal, 914mm=blue.
3. At each of the five x positions, take a +/-14 px column window, mask pixels
   within a color tolerance, and split the matched rows into row-contiguous
   groups (gap > 25 px starts a new group). Two groups -> (max, min) directly.
   One wide group (span > 55 px) means an unresolved max/min crossing (the two
   markers visually overlap near PRD_m ~ 0, mostly at small beta for the
   deeper/less-restrictive curves) -- split it in half by row and average each
   half. One narrow group means the max and min markers are themselves
   coincident to sub-pixel precision -- use its centroid for both.
   The right (std) panel has only one branch per depth, but its legend box
   sits inside the axes at beta ~= 1-31%, std ~= 5.7-9.1% (both empirically
   from the swatch rows), which the low-beta/high-std corner of every curve
   never reaches -- checked, not assumed -- so it is masked out unconditionally
   rather than only where it would collide.

Outputs (data/han_dc/):
  fig12_max_d<depth>.csv, fig12_min_d<depth>.csv, fig12_std_d<depth>.csv
    for depth in {305,457,610,762,914} -- columns x,y (beta %, PRD_m %),
    consumed by `datacenter.plots.fig12`.
  digitize_fig12_notes.md -- calibration constants and the raw per-point table,
    for anyone who wants to re-check a value against the printed figure.

Run: ../../.venv/bin/python digitize_fig12.py
"""
from __future__ import annotations

import csv
import os

import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, "..", "..", "Han21.pdf")
OUT_DIR = os.path.join(HERE, "..", "..", "data", "han_dc")
PAGE_INDEX = 10
CLIP = pymupdf.Rect(60, 455, 540, 793)
ZOOM = 8

LEFT = dict(x0=557.5, x1=1816.0, xd0=10.0, xd1=60.0, y0=715.5, y1=1948.5, yd0=30.0, yd1=-20.0)
RIGHT = dict(x0=2194.0, x1=3453.0, xd0=10.0, xd1=60.0, y0=715.5, y1=1948.5, yd0=10.0, yd1=0.0)
# right-panel legend box (px, in the same frame) -- masked out unconditionally
LEGEND_BOX = dict(row0=820, row1=1310, col0=2194, col1=2800)

DEPTHS_MM = [305, 457, 610, 762, 914]
BETAS_PCT = [15, 25, 35, 45, 56]
SERIES_COLOR = {
    305: None,               # dark-pixel mask (black line/marker/text)
    457: (191, 191, 0),      # olive
    610: (191, 0, 191),      # magenta
    762: (0, 190, 190),      # teal
    914: (0, 0, 191),        # blue
}
COLOR_TOL = 60
HALF_WIN = 14
GAP_PX = 25
WIDE_SPAN_PX = 55


def render():
    doc = pymupdf.open(PDF)
    pix = doc[PAGE_INDEX].get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), clip=CLIP)
    return np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].astype(int)


def _col_of(panel, xd):
    return panel["x0"] + (xd - panel["xd0"]) / (panel["xd1"] - panel["xd0"]) * (panel["x1"] - panel["x0"])


def _y_of_row(panel, row):
    return panel["yd0"] + (row - panel["y0"]) / (panel["y1"] - panel["y0"]) * (panel["yd1"] - panel["yd0"])


def _mask(region, depth_mm):
    r, g, b = region[..., 0], region[..., 1], region[..., 2]
    color = SERIES_COLOR[depth_mm]
    if color is None:
        return (r < 100) & (g < 100) & (b < 100)
    dist = np.sqrt((r - color[0]) ** 2 + (g - color[1]) ** 2 + (b - color[2]) ** 2)
    return dist < COLOR_TOL


def _row_groups(img, panel, depth_mm, beta_pct, mask_legend):
    c = int(round(_col_of(panel, beta_pct)))
    y0i, y1i = int(panel["y0"]) + 4, int(panel["y1"]) - 4
    col0, col1 = c - HALF_WIN, c + HALF_WIN
    region = img[y0i:y1i, col0:col1]
    m = _mask(region, depth_mm)
    if mask_legend:
        rows_abs = np.arange(y0i, y1i)
        cols_abs = np.arange(col0, col1)
        in_legend = (
            (rows_abs[:, None] >= LEGEND_BOX["row0"]) & (rows_abs[:, None] <= LEGEND_BOX["row1"]) &
            (cols_abs[None, :] >= LEGEND_BOX["col0"]) & (cols_abs[None, :] <= LEGEND_BOX["col1"])
        )
        m = m & ~in_legend
    rows = np.where(m.any(axis=1))[0]
    if rows.size == 0:
        return []
    rows = np.sort(rows + y0i)
    groups, cur = [], [rows[0]]
    for row in rows[1:]:
        if row - cur[-1] > GAP_PX:
            groups.append(cur)
            cur = [row]
        else:
            cur.append(row)
    groups.append(cur)
    return groups


def digitize_left(img, depth_mm, beta_pct):
    """-> (max_pct, min_pct)."""
    groups = _row_groups(img, LEFT, depth_mm, beta_pct, mask_legend=False)
    if not groups:
        raise RuntimeError(f"fig12 left panel: no pixels for depth={depth_mm} beta={beta_pct}")
    if len(groups) >= 2:
        # two or more groups: take the extreme (topmost=max, bottommost=min)
        top, bottom = groups[0], groups[-1]
        return _y_of_row(LEFT, np.mean(top)), _y_of_row(LEFT, np.mean(bottom))
    rows = np.array(groups[0])
    span = rows[-1] - rows[0]
    if span > WIDE_SPAN_PX:
        mid = rows[0] + span / 2.0
        top_half = rows[rows <= mid]
        bot_half = rows[rows > mid]
        return _y_of_row(LEFT, np.mean(top_half)), _y_of_row(LEFT, np.mean(bot_half))
    v = _y_of_row(LEFT, np.mean(rows))
    return v, v


def digitize_right(img, depth_mm, beta_pct):
    """-> std_pct."""
    groups = _row_groups(img, RIGHT, depth_mm, beta_pct, mask_legend=True)
    if not groups:
        raise RuntimeError(f"fig12 right panel: no pixels for depth={depth_mm} beta={beta_pct}")
    rows = np.concatenate(groups) if len(groups) > 1 else np.array(groups[0])
    return _y_of_row(RIGHT, np.mean(rows))


def main():
    img = render()
    os.makedirs(OUT_DIR, exist_ok=True)
    notes = ["# Fig. 12 digitization (marker centroids)\n",
             f"Calibration: LEFT={LEFT}\nRIGHT={RIGHT}\nLEGEND_BOX={LEGEND_BOX}\n\n",
             "| depth_mm | beta_pct | max_pct | min_pct | std_pct |\n",
             "|---|---|---|---|---|\n"]
    rows_by_depth = {d: {"max": [], "min": [], "std": []} for d in DEPTHS_MM}

    for depth_mm in DEPTHS_MM:
        for beta_pct in BETAS_PCT:
            mx, mn = digitize_left(img, depth_mm, beta_pct)
            sd = digitize_right(img, depth_mm, beta_pct)
            rows_by_depth[depth_mm]["max"].append((beta_pct, mx))
            rows_by_depth[depth_mm]["min"].append((beta_pct, mn))
            rows_by_depth[depth_mm]["std"].append((beta_pct, sd))
            notes.append(f"| {depth_mm} | {beta_pct} | {mx:.2f} | {mn:.2f} | {sd:.2f} |\n")

    for depth_mm in DEPTHS_MM:
        for series in ("max", "min", "std"):
            path = os.path.join(OUT_DIR, f"fig12_{series}_d{depth_mm}.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["x", "y"])
                for x, y in rows_by_depth[depth_mm][series]:
                    w.writerow([x, f"{y:.3f}"])
            print(f"wrote {path}")

    with open(os.path.join(HERE, "digitize_fig12_notes.md"), "w") as f:
        f.writelines(notes)
    print(f"wrote {os.path.join(HERE, 'digitize_fig12_notes.md')}")


if __name__ == "__main__":
    main()
