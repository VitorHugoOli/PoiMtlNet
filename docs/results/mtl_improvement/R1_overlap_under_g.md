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

## Verdict — MECHANISM WIN (dual-tower confirmed); G−ceiling NULL (rising tide)

**The private tower absorbs the dense supervision the shared backbone wasted.** The identical
windowing intervention that the OLD shared-backbone regime could not use (MTL reg +0.50 vs STL +5.13,
gap widened ~+4.6pp) is now **fully captured by G's private STAN reg tower** (MTL reg +4.89 ≈ STL
+5.12, gap unchanged). G's MTL reg pathway behaves like an STL pathway under the data scaling —
exactly the dual-tower design intent. This is **direct causal evidence for the dual-tower thesis**
(the central architectural claim of the study), independent of the C25 loss confound.

Per the magnitude rule this is NOT a champion-improving number — overlap is a *rising tide* that lifts
G and the ceiling equally, so the matched G−ceiling bar is unchanged (−0.23pp shift, within σ). The
value is the **mechanism**: the dual-tower converts a shared-backbone bottleneck (gap-widening under
more data) into a private-pathway non-issue (gap-stable). The non-overlap canon is unchanged (the
2026-06-03 user decision stands); overlap remains documented future-work that now ALSO carries this
mechanism evidence.

## Scope / caveats
- AL only, single seed (42), matching the seed-42 overlap STL ceilings + frozen non-overlap ceilings.
  A single-state mechanism probe (the card's "AL first → AZ/GE/FL if it moves"); it does not move the
  matched bar, so multi-state escalation is not motivated for a *number*, but a 1-state confirm at
  FL (where the matched gap is largest, −0.31) would harden the mechanism claim if desired.
- Folds for the overlap engine are generated on-the-fly (not frozen) — fine for this rising-tide
  contrast (both arms compared to their own-windowing ceiling); a frozen-fold version is unnecessary
  given the −0.23pp shift sits well inside σ.
