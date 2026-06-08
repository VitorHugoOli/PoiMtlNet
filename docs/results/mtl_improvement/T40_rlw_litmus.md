# T4.0b — RLW litmus on champion G (2026-06-08)

**Tier 4, the cheap ungated pre-check.** Driver `scripts/mtl_improvement/t40_rlw_litmus.sh`;
manifest `t40_rlw_manifest.tsv`. Champion G arch/recipe, but `--mtl-loss random_weight` (RLW, Lin
TMLR'22 — Dirichlet per-step random task weights) instead of `static_weight --category-weight 0.75`.
Run via `--canon none` + explicit G flags + `--no-reg/cat-class-weights`. AL + FL, seed 0. Reg
scored matched-metric (R0 method). Comparand = G (static_weight) seed-0.

## Result (seed 0)

| state | RLW reg-full | G reg-full | Δreg | RLW cat F1 | G cat F1 | Δcat |
|---|---|---|---|---|---|---|
| AL | 62.31 | 62.64 | −0.33 | 54.25 | 52.75 | +1.49 |
| FL | 73.02 | 72.95 | +0.07 | 71.94 | 73.12 | −1.18 |

(c) reg ceilings: AL 62.67 / FL 73.27. G matched gap: AL −0.09 / FL −0.31.

## Verdict — the inter-task weight is NOT the bottleneck

Random per-step task weighting (which spans the whole weight simplex over training) **matches** the
tuned static_weight cw0.75 on reg (within ±0.33pp) and merely **trades** on cat (AL +1.49 / FL −1.18,
roughly zero-sum). This is the canonical RLW litmus signal (Lin et al., TMLR'22): *if random weighting
≈ your tuned static weight, the choice of inter-task weight is not a sensitive lever.*

**Implication for T4.1:** the full `src/losses` balancer registry (nash/cagrad/gradnorm/aligned_mtl/
dwa/db_mtl/uw/uw_so/stch/... — all of which manipulate the inter-task weighting or per-task gradient
combination) is **predictably low-EV** under G. It will not beat static_weight, consistent with:
- the prior P4 verdict ("balancing is low-EV; the gap is not interference"),
- T2V.6 (famo/cagrad/nash/uw all ≈ G at default params),
- and now the RLW litmus directly (the weight axis itself is flat).

Neither RLW nor the matched bar moves — G sits at the reg ceiling regardless of inter-task weighting.

**What this does NOT test:** intra-task scale (the ~4.3× CE-magnitude gap, ln(4703)≈8.46 reg vs
ln(7)≈1.95 cat). That is T4.0a (loss-scale normalization) — a distinct mechanism (it rescales the
per-task gradient magnitude, not just the scalar weight). RLW does randomize the scalar weight across
the simplex, which indirectly exercises extreme scale imbalances and still matches G — so scale-norm is
*also* predicted low-EV, but it is a genuinely-untested distinct lever and is the highest-remaining-EV
Tier-4 card. (Decision pending — see log.md.)
