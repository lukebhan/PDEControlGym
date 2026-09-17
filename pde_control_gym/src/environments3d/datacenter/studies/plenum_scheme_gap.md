# Plenum PRD: what drives the converged ~55 vs Han 28, and why D8 (Chen+ns4) works

**Status (2026-09-15):** investigated hard. Net result: the D8 empirical match
(Chen + ns=4) is confirmed as the best corner-uniform reproducer; the "converged
solve gives ~55 at 305/56" puzzle is explained (Chen over-predicts eddy viscosity
at shallow/fast corners); but **no single-model converged solve we tried
reproduces Han across all corners** — so there is no clean drop-in "fix" that
beats D8, and the honest remaining task is to identify Han's actual plenum
turbulence model. Do NOT switch the plenum to a flat nu_t=100*nu (it breaks the
deep corners — see the table).

## The full corner table (dt=1.0, 305-914mm, beta as noted)

| corner | Han | Chen ns=4 (D8 default) | Chen converged (ns80) | const nu_t=100nu converged |
|---|---:|---:|---:|---:|
| 305mm/56% | 28.01 | 26.65 | 55.28 | 29.33 |
| 305mm/45% | 18.97 | 18.15 | 40.60 | 20.58 |
| 610mm/56% | 6.22  | 6.07  | 9.99 (did not converge) | 12.28 |
| 914mm/56% | 4.06  | 4.43  | 5.54  | 6.48 |
| mean\|err\| | — | **0.67** | large | ~2-3 (worse at depth) |

Chen+ns=4 matches Han across every corner. Converged-Chen overshoots badly at the
shallow/fast corners. Constant-100nu matches at shallow but **over-predicts ~2x at
depth**, so it is uniformly worse than D8.

## What the experiments established

1. **Scheme is not the cause.** Our `_assemble_u` is a faithful FFD-Upwind — its
   upwind coefficients, face fluxes and AP convective term match Han's kernels
   (`adve_VX_im`+`ap_im_coeff`) term-by-term; diffusion matches `diff_VX`. The
   turbulence experiments below change the answer with the scheme held fixed.

2. **Turbulence is the lever at the shallow/fast corners.** Single-variable at
   305/56, converged momentum: Chen (nu_t/nu max = **2092**) -> 55.28; constant
   nu_t=100nu -> 29.33 ≈ Han. So the converged-Chen 55 is inflated by Chen's
   excessive eddy viscosity there. (Sign, per prior work: more nu_t RAISES tile
   non-uniformity.)

3. **But Chen's nu_t is strongly corner-variable, and no flat value fixes it.**
   nu_t = C*|V|*l is huge at shallow/fast corners (|V| and l both large) and
   *below* 100nu at deep/slow corners. That is why constant-100nu over-predicts at
   depth (610/914): there the real (and Han's) eddy viscosity is *lower* than
   100nu, so flat-100nu under-mixes and inflates PRD.

4. **Why D8 (Chen+ns4) nonetheless matches Han everywhere — mixed reasons:**
   - Deep/low-CFL corners (610,914): ns=4 is essentially converged (low CFL needs
     few sweeps) and Chen's nu_t is moderate/low there, so Chen+ns4 ≈ converged ≈
     Han *legitimately*.
   - Shallow/high-CFL corners (305): converged Chen would give ~55 (excess nu_t),
     but ns=4 under-converges the momentum and pulls it back to ~27 ≈ Han. That
     match is partly **error cancellation** (Chen's excess nu_t up, under-converged
     momentum down). It is real and reproducible across the D8 mesh, but it is not
     a converged, sweep-independent result at those corners.

## Independent reference (plan test #4) — found

Tian, VanGilder, Healey, Condor, Han, Zuo, *A New Fast Fluid Dynamics Model for
Data-Center Floor Plenums* (Schneider Electric + CU Boulder; NSF PAR 10180923).
Same group, same FFD, same plenum problem. States explicitly:
- **first-order upwind** advection, chosen over semi-Lagrangian for rigorous mass
  & energy conservation (confirms Han's scheme; ours matches);
- turbulence: constant zero-equation **nu_t = 100*nu** for the *hypothetical*
  plenum, and a **zero-equation (Dhoot et al. 2017)** for the *real DC* plenum;
- **dt = 0.05-0.1 s**, marched 100 s to steady; mesh 6in horizontal x 2in vertical
  (nz=6);
- validated against **commercial CFD**, max tile-flow difference 3.6%.

So the group's plenum turbulence is a *zero-equation*, not our Chen calibration
and not necessarily a flat 100nu. Our Chen (room-case calibrated) over-predicts
nu_t at the plenum's high-velocity corners.

## Supporting: isat_ffd (shipped build) runs FFD-SL, no working upwind

Call-graph audit: the `_im` implicit-upwind kernels are dead (never
`clCreateKernel`'d, never called, `advection_solver` never read in the `.cl`, and
`diff_VX`/`adve_VX_im` both `=`-assign the same arrays so cannot combine). Live
path is semi-Lagrangian `adve_VX` + diffusion-only `diff_VX`/`ap_coeff` + 30
Jacobi. isat_ffd is the group's FFD-SL *baseline* build, not a runnable upwind
reference; its IT_MAX=30 is the SL diffusion solve. Does not contradict the paper.

## Turbulence / dt / mesh exploration (2026-09-15) — what it ruled out

Chased the "principled converged reproduction" through the paper's known knobs.
None gives a clean corner-uniform converged match; the exploration's value is in
what it eliminates:

- **Constant nu_t=100nu — ruled out.** Matches at shallow (305/56 conv 29.3 ≈ Han
  28) but over-predicts ~2x at depth (610 12.3 vs 6.2; 914 6.5 vs 4.1).
- **Mesh — ruled out.** Repeating const-100nu at cpt=4 (6in, the paper's
  resolution) + cholesky + converged: 305/56 25.7, 610/56 11.7, 914/56 6.2 — same
  depth over-prediction. Finer mesh doesn't fix it.
- **dt / CFL — ruled out.** Converged Chen at 305/56 is 54.2 / 56.0 / 54.2 at
  dt = 1.0 / 0.5 / 0.25. The shallow converged overshoot is dt-independent (and
  mesh-independent, per prior work). (The VanGilder outer loop also stops
  converging below dt~0.25 — a coupling artifact.)
- **Chen nu_t is ~370*nu (mean) at ALL depths** (peaks 1500-2100), i.e. not lower
  at depth — so the depth behaviour is not a Chen-magnitude effect.
- **Dhoot-2017 exact formula: not obtainable** (paywalled/403). Literature
  summaries describe it as a constant-100nu-with-wall-function, i.e. essentially
  the constant model already tested (which fails at depth).

**The real structure exposed:** the CONVERGED solve over-predicts Han at *every*
corner, with either turbulence model, any mesh, any dt (Chen conv 55/~10/5.5;
const conv 29/12/6.5 vs Han 28/6/4). Only the **low-per-step-iteration regime**
(our standard ns=4) brings it down to Han across all corners (Chen+ns4
26.65/18.15/6.07/4.43). The gap between converged and ns=4 is large at the
shallow/high-CFL corner (55 vs 27) and small at the deep/low-CFL corners (5.5 vs
4.4) — consistent with a per-step fractional-step under-resolution that scales
with CFL.

## Honest bottom line + recommendation

- Not the scheme (faithful upwind), not a turbulence-model swap (const-100nu fails
  at depth), not the mesh, not dt. D8 (Chen + our standard ns=4) reproduces Han
  across all corners (mean|err| 0.67) and **stands as the best reproduction.**
- Han's Fig-12 corresponds to the **low-per-step-iteration FFD regime** (FFD's
  speed premise; his upwind sweep count is unstated — isat_ffd's IT_MAX=30 is the
  SL path, not upwind). Our standard ns=4 matches it with no plenum tuning. The
  honest ambiguity: whether that low-sweep match is "legitimate FFD" or a
  fortuitous under-resolution can't be settled from Han's paper alone.
- The one untried lever that could still explain the converged over-prediction is
  the **plenum SETUP** (plan test #3: our VanGilder pressure-shift coupling,
  full-short-wall inlet, loss-coefficient-vs-depth) — not turbulence, dt, mesh, or
  scheme, which are now eliminated.
- SL cross-check (plan test #a) is moot: the driver is not the advection scheme.
