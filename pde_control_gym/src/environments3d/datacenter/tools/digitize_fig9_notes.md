# Digitizing Fig. 9 (the reference data center layout)

Source: `Han21.pdf` page 9 (PDF page index 8), the embedded raster image
(xref 224, 1777x1316 px, JPEG) behind Fig. 9. Extracted with `pymupdf`'s
`extract_image` (native embedded resolution, sharper than a page render at
300 dpi -- the two happen to coincide in scale here: the image's own pixel
grid maps 1:1 onto a 300 dpi render of its PDF bounding box).

## Method

1. **Grid detection.** The floor plan is drawn on a uniform tile grid.
   Peak-finding on the per-column/per-row count of near-black pixels (a
   `scipy.signal.find_peaks` sweep) found 51 vertical and 38 horizontal grid
   lines exactly -- a 50 x 37 tile grid, matching `100 ft / 2 ft = 50` and
   `74 ft / 2 ft = 37` (Han: "approximately 30.5 m (100 ft) long, 22.6 m
   (74 ft) wide", 2x2 ft tiles). This fixed `room_tiles = [50, 37]`.

2. **Color classification (tiles, PDUs).** Each grid cell's dominant color
   (median RGB over its central 50%, avoiding grid lines and label text) was
   matched to the four legend swatches, sampled directly from the legend
   band of the same image:
   - floor: `(255,255,255)`
   - Rack 42U (gray): `(188,188,188)`
   - Rack 45U (purple): `(148,122,185)`
   - Perforated tile 25% open: `(11,106,136)`
   - PDU: `(207,150,133)`

   This is reliable for **tiles and PDUs**, which are drawn flush to the
   tile grid: 183 tile cells and 12 PDUs (48 cells at 2x2 each) came out of
   the classifier with no manual correction, matching Han's stated "183
   perforated floor tiles" and "12 PDUs" exactly. A QA overlay (outlines
   drawn over the source image at the classified cell boundaries) confirmed
   every tile/PDU outline lands exactly on the drawn box.

3. **Racks: color classification does NOT work, read by eye instead.**
   Rack boxes are **not** drawn on the tile grid -- they're overlaid at
   their own pitch (a zoomed crop of Row A shows rack box borders offset by
   a half-cell from the tile grid lines, and packed slightly denser than
   1 rack per tile row). A first pass using per-cell color classification
   for racks (same method as tiles/PDUs) produced 350+ "rack" cells from
   color bleed at box edges -- nowhere near 151. This matches the plan's
   prior note that a low-DPI attempt over-counted racks; the cause is now
   understood: racks are simply not grid-aligned in the source figure, so
   there is no cell to classify.

   Instead, each row-letter's rack column (A, C, D, E, F, G, H, I, J, and
   the standalone column labeled L) was cropped full-height at 3x zoom and
   the rack ID numbers were read directly off the image, together with
   box color (42U gray / 45U purple). Row-letter x-positions were found the
   same way tile columns were (dominant classified type per candidate
   column, since tile columns ARE grid-aligned); the adjacent rack lane was
   taken as the immediately-neighboring column, snapping each rack's
   position to the tile grid (footprint 1 tile x 1 tile, per the plan's
   default -- Fig 9 gives no other geometric description, and does not
   itself resolve to a clean sub-tile footprint either).

   Within each row-letter, both the "top" cluster (image rows 4-16) and
   "bottom" cluster (image rows 20-32, separated by a blank center aisle)
   number their racks so the **largest number sits nearest the outer wall /
   cluster start, decreasing toward the aisle** (verified against every
   row's visible sequence, e.g. Row A top 18..11 top-to-bottom, bottom
   9..1 top-to-bottom). Racks were placed at consecutive tile rows in that
   order, i.e. gaps in Han's *numbering* (e.g. Row G's top cluster has only
   "13" and "11", skipping "12" -- see below) are NOT reproduced as gaps in
   *position*; they'd have no physical meaning at our simplified 1x1-tile
   rack footprint.

4. **Facing direction.** Derived from which side of its tile column a row's
   racks sit on (column classification table below), not eyeballed --an
   initial eyeball pass on a couple of rows (E, I) was self-inconsistent and
   the classification table caught it.

## Row-by-row tally

| row | top cluster | bottom cluster | total | U | tile col | rack col | facing |
|---|---|---|---|---|---|---|---|
| A | 11-18 (8) | 1-9 (9) | 17 | top 42U, bottom **45U** | 5 | 4 | +x |
| C | 11-15 (5) | 1-10 (10) | 15 | 42U | 7 | 8 | -x |
| D | (none) | 1-10 (10) | 10 | 42U | 14 | 13 | +x |
| E | 11-20 (10) | 1-10 (10) | 20 | 42U | 16 | 17 | -x |
| F | 11-19 (9) | 1-10 (10) | 19 | 42U | 23 | 22 | +x |
| G | **11, 13** (2, both empty) | 1-10 (10) | 12 | 42U | 26 | 27 | -x |
| H | 15, 17 (2) | 1-10 (10) | 12 | 42U | 33 | 32 | +x |
| I | 12, 14-20 (8) | 1-10 (10) | 18 | 42U | 35 | 36 | -x |
| J | 11-20 (10) | 1-9 (9) | 19 | 42U | 42 | 41 | +x |
| L (standalone) | 11-19 (9) | (none) | 9 | **45U** | (none, uses J's col 42/44 area) | 45 | -x |

**Total: 151 racks** (133 42U + 18 45U), matching Han exactly ("151 racks
... 18 45U networking racks ... all the remaining racks ... 42U"). Rows A
and L are the two 45U rows -- the paper's OCR'd text reads "Rows 1 and 10",
which is a PDF text-extraction artifact (the font's glyph mapping for row
letters is garbled in `pymupdf`'s text layer); "1st and 10th" matches
exactly if A..J,L are counted as the 10 row-groups in left-to-right order
(A=1st, L=10th).

**Empty racks:** the paper states in prose (not visible in Fig 9's color
coding, since empty racks are drawn the same gray as any other 42U rack):
"Racks G11 and G13 (i.e., the 11th and 12th cabinets in Row G) are empty."
Row G's top cluster visibly contains only racks 11 and 13 (12 is skipped
entirely in the drawing) -- i.e. the two racks Han calls out are exactly
Row G's entire top cluster. `racks.csv` marks both `empty=1`;
`datacenter.racks.assign_powers` excludes `empty` racks from the power
distribution (0 kW, and excluded from the U-proportional denominator), so
the powered-rack total U works out to 131x42 + 18x45 = 6312, matching the
149-powered-rack figure Han uses for Fig. 10 ("149 racks (excluding 2 empty
racks)").

## PDU and stairs positions

12 PDU footprints (2x2 tiles each) read directly from the classified grid,
labeled 105/205/103/203/102/202 (top wall) and 104/204/101/201/106/206
(bottom wall) in the figure -- labels not stored (racks.csv/floor_map.txt
have no PDU-numbering field; add one later if a study needs it).

"Stairs" appears twice as a text label (top-left, bottom-right) with no
drawn footprint in Fig 9 (unlike racks/tiles/PDUs it isn't color-coded).
Approximated as a 2x2-tile block at each corner (`floor_map.txt` 'S'); this
is a guess, not a digitization, and has no heat/flow role beyond being
adiabatic solid (per the plan's defaults).

## Known discrepancies / approximations

- **Rack sub-tile position**: every rack snaps to a 1x1 tile footprint
  immediately adjacent to its row's tile column. Han's own drawing doesn't
  resolve to a clean sub-tile grid for racks either (see point 3 above), so
  this is not expected to be recoverable more precisely from Fig 9.
- **Ceiling tile positions**: not digitized -- Fig 9 doesn't show them.
  `tools/make_ceiling_map.py` places 42 tiles evenly along the hot-aisle
  lines it derives from `racks.csv` (the column/row directly behind each
  rack's rear face); see that file's docstring.
- **Tile/PDU footprint vs. Han's own grid**: Han's FFD grid is 200x148 for
  the white space (4 cells/tile) and 224x161 for the plenum -- both bigger
  than the 50x37 tile grid recovered here in the plenum's case (224x161 at
  6 cells/tile would need a plenum footprint larger than the white space by
  `plenum_margin_tiles` per side; the plan already flags this "unexplained"
  and defaults `plenum_margin_tiles=0"). Not resolved by this digitization.
