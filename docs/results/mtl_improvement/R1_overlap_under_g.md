# R1 — Overlap-under-G: does the private reg tower absorb dense supervision? (2026-06-08)

**Tier 3, AL seed=42, clean paired 2×2 in the CURRENT harness.** Driver
`scripts/mtl_improvement/r1_overlap_under_g.sh`; manifest
`scripts/mtl_improvement/r1_overlap_manifest.tsv`.

The overlapping-window study (2026-06-03) found dense (stride-1) supervision lifts **STL reg
+5.13pp** but, under the **OLD regime** (class-weighted CE + the SHARED `next_stan` reg backbone),
**MTL reg lifted only +0.50pp → the STL→MTL reg gap WIDENED** (the shared backbone could not absorb
the extra supervision). R1 re-runs the SAME windowing contrast under **champion G** (dual-tower
private STAN reg pathway, post-C25 unweighted, aux + prior-OFF) to test the dual-tower thesis:
does the private tower — a private pathway like STL — now absorb the dense lift?

Reg is scored matched-metric (R0 method): G-full = `indist·(1−ood_fraction)` at the indist-best
epoch; STL reg ceiling = p1 FULL `top10_acc` at the top10-best epoch. log_T is inert (G prior-OFF +
KD-off); the v14 seed-42 log_T satisfies the trainer's existence/freshness guards.

## Result (AL seed=42)

| windowing | STL reg ceil (full) | G reg-full | **ΔG−ceil reg** | G cat F1 | G reg-indist |
|---|---|---|---|---|---|
| non-overlap (v14) | 62.88 ± 4.05 | 62.77 ± 4.0 | **−0.11** | 52.79 | 64.79 |
| overlap (dk_ovl, stride-1) | 68.01 ± 4.22 | 67.67 ± 3.6 | **−0.34** | 61.18 | 68.09 |

- **STL reg ceiling lift** (overlap − non-overlap) = **+5.12** (reproduces the prior study's +5.13 exactly).
- **G reg-full lift = +4.89** — essentially the SAME magnitude as the STL lift.
- **ΔG−ceil reg shift = −0.23pp** (−0.11 → −0.34) — within fold σ (~4pp); the matched bar does NOT move.
- cat: G overlap lift **+8.39** (rising tide; STL cat lift was +9.77).
- Cross-checks: non-overlap reg ceiling 62.88 reproduces the frozen §2 seed-42 ceiling; non-overlap
  G ΔG−ceil −0.11 reproduces R0's multi-seed −0.09 → the seed-42 cell is sound.

## R1b de-confound (2026-06-08, advisor-required) — the absorber is C25, NOT the dual-tower

⚠ **The initial R1 verdict ("the PRIVATE tower absorbs the dense lift the shared backbone wasted")
was an OVER-READ — corrected here.** The +0.50→+4.89 swing changed TWO things at once: the loss
(class-weighted → C25-unweighted) AND the backbone (shared → private dual-tower). The advisor flagged
the confound; R1b isolates it by running the **SHARED backbone (mtlnet_crossattn + next_stan_flow,
prior-OFF) under the SAME C25-unweighted onecycle recipe** — i.e. G *minus* the private tower —
overlap vs non-overlap, AL seed42. Driver `scripts/mtl_improvement/r1b_shared_overlap_deconfound.sh`.

| AL seed42, overlap reg lift | regime | lift |
|---|---|---|
| OLD overlap study | class-weighted CE + SHARED backbone | **+0.50** |
| R1b | **C25-unweighted + SHARED backbone** | **+4.32** |
| R1 (G) | C25-unweighted + PRIVATE tower | +4.89 |

(R1b: shared reg-full 62.84 → 67.16; cat 53.05 → 60.98.)

**The shared backbone, once unweighted, ALSO absorbs the dense supervision (+4.32 ≈ the tower's
+4.89, within fold σ ~4).** The private tower adds only ~+0.57 (inside noise). So the dominant cause
of "overlap now registers in MTL reg" is **C25 unweighting** (the class-weighted loss optimised away
from top-K, suppressing the lift to +0.50) — **NOT the dual-tower architecture.** The old "shared
backbone can't absorb more data → gap widens" reading was the class-weighting confound, exactly the
pattern C25 fixed everywhere else.

## Verdict — rising tide on the matched bar; the "gap-widening" was the C25 confound (not the tower)

1. **G−ceiling NULL (rising tide) — solid, unchanged.** Overlap lifts G and the STL reg ceiling ≈
   equally (G reg +4.89, ceiling +5.12) → the matched G−ceiling bar is unchanged (−0.11 → −0.34,
   within σ). Not a champion-improving number.
2. **Mechanism (corrected): the absorber is C25, not the dual-tower.** R1b shows the shared backbone
   absorbs the dense lift just as well (+4.32) once unweighted. R1 therefore is **further evidence for
   C25 as the dominant reg lever** (the old overlap "gap-widening" was the class-weighting confound),
   NOT independent evidence for the dual-tower. The dual-tower thesis stands on its OTHER evidence
   (the C25-era Tier-2 re-run, T2V.4 alt-arch fair re-rank, B-A1 STAN-load-bearing) — R1 neither adds
   to nor subtracts from it.
3. Non-overlap canon unchanged (the 2026-06-03 user decision stands); overlap stays documented
   future-work, now with the cleaner attribution (its MTL benefit is unlocked by C25, and it's a
   rising tide that doesn't beat the achievable ceiling).

## Scope / caveats
- AL only, single seed (42), matching the seed-42 overlap STL ceilings + frozen non-overlap ceilings.
  A single-state mechanism probe (the card's "AL first → AZ/GE/FL if it moves"); it does not move the
  matched bar, so multi-state escalation is not motivated for a *number*, but a 1-state confirm at
  FL (where the matched gap is largest, −0.31) would harden the mechanism claim if desired.
- Folds for the overlap engine are generated on-the-fly (not frozen) — fine for this rising-tide
  contrast (both arms compared to their own-windowing ceiling); a frozen-fold version is unnecessary
  given the −0.23pp shift sits well inside σ.
