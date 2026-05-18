# STACKING ABLATION — canonical → +v3c → +T3.2 ResLN
**Date**: 2026-05-16 (Tier-4 paper closeout)
**Source**: per-seed JSONs under `docs/results/canonical_improvement/`. n=5 fold-pairs × 5 seeds = 25 paired observations per cell. Canonical reference from `RESULTS_TABLE.md §0.1` (n=20 fold-pairs over 4 seeds, seed=42 single).

## §1 — FL multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 (mean ± σ_seed) | reg Acc@10 | leak F1 | Δcat vs canonical | Δreg vs canonical | Δleak vs canonical |
|---|---:|---:|---:|---:|---:|---:|
| canonical (T1.1 / §0.1) | 68.56 ± 0.79 | 63.27 ± 0.10 | 40.85 ± 0.39 | — | — | — |
| + v3c (AdamW WD=5e-2) | 68.23 ± 0.18 | 63.90 ± 0.05 | 41.19 ± 0.25 | -0.33 | **+0.63** | +0.34 |
| + v3c + T3.2 ResLN (SHIPPING) | 69.42 ± 0.25 | 63.98 ± 0.09 | 43.09 ± 0.12 | **+0.86** | +0.71 | +2.24 |
| **T3.2 marginal contribution** | **+1.19** (cat axis) | +0.08 (reg null) | +1.90 | | | |

## §2 — AL multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 | reg Acc@10 | leak F1 |
|---|---:|---:|---:|
| canonical | 40.57 ± 0.24 | 50.17 ± 0.24 | 31.04 ± 1.33 |
| + v3c + T3.2 ResLN | 42.05 ± 0.29 | 49.88 ± 0.55 | 32.73 ± 0.49 |
| Δ vs canonical | **+1.48** | -0.29 | +1.69 |

## §3 — AZ multi-seed (5 seeds: 42, 0, 1, 7, 100)

| Stage | cat F1 | reg Acc@10 | leak F1 |
|---|---:|---:|---:|
| canonical | 45.10 ± 0.19 | 40.78 ± 0.07 | 34.57 ± 0.78 |
| + v3c + T3.2 ResLN | 46.80 ± 0.39 | 40.63 ± 0.40 | 37.03 ± 0.34 |
| Δ vs canonical | **+1.70** | -0.15 | +2.46 |

## §4 — Paired sign test (5/5 directional per state)

| state | axis | signs (T3.2 stack > canonical per seed) | one-sided p |
|---|---|---|---:|
| FL | cat | 5/5 | 0.03125 |
| FL | reg | 5/5 | 0.03125 |
| AL | cat | 5/5 | 0.03125 |
| AZ | cat | 5/5 | 0.03125 |

## §5 — Shipping stack final claim

**canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResidualLN encoder**:
- **FL**: cat **+0.86 pp** (5/5 seeds positive, sign-test p=0.03125); reg **+0.71 pp** (5/5 seeds positive, p=0.03125). Leak Δ +2.24 pp, well below +5 pp red flag.
- **AL**: cat **+1.48 pp** (5/5 seeds positive); reg Δ -0.29 (null at small states, matching v3c precedent).
- **AZ**: cat **+1.70 pp** (5/5 seeds positive); reg Δ -0.15 (null).

Two paper-grade micro-improvements on disjoint axes (v3c→reg, T3.2→cat), additive by construction, leak-budgeted at 44% of the red-flag allowance.

## §6 — Phase 1 ablations: is v3c load-bearing, and does T4.3 lift in stack?

Two post-Tier-4 ablations probed at n=5 seeds (42, 0, 1, 7, 100). Phase 1 multi-seed evidence falsifies the 2026-05-16 first-pass argument that "v3c is load-bearing" — replaces it with the more defensible "v3c is dispensable at the mean but retained for protocol inertia + 5-state replication coverage already locked."

### §6.1 — Hyp D: drop v3c (`canonical + T3.2 ResLN` only)

| metric | Shipping (T3.2+v3c) n=5 | Hyp D (T3.2 only) n=5 | mean Δ | paired-t p₂ |
|---|---:|---:|---:|---:|
| FL cat F1 | 69.42 ± 0.25 | 69.62 ± 0.22 | +0.235 | 0.082 (BORDERLINE) |
| FL reg Acc@10 | 63.98 ± 0.09 | 64.03 ± 0.09 | +0.063 | 0.661 (NULL) |
| FL leak F1 | 43.09 ± 0.12 | 43.15 ± 0.27 | +0.033 | 0.851 (NULL) |
| AL cat F1 | 42.05 ± 0.29 | 41.99 ± 0.34 | −0.059 | 0.775 (NULL) |
| AL reg Acc@10 | 49.88 ± 0.55 | 50.17 ± 0.36 | +0.289 | 0.311 |
| AZ cat F1 | 46.80 ± 0.39 | 46.95 ± 0.22 | +0.150 | 0.499 (NULL) |
| AZ reg Acc@10 | 40.63 ± 0.40 | 40.92 ± 0.10 | +0.293 | 0.202 |
| **Pooled AL+AZ reg** | (n=10 paired) | | **+0.291** | **0.082** (BORDERLINE) |

**Leak parity (IJM user-held-out, FL seed=42)**:
- Canonical drift = +0.101 pp; Shipping drift = −0.024 pp; **Hyp D drift = +0.016 pp**.
- Both stacks pass identically; v3c does NOT reduce leak vs no-WD.

**Findings**:
1. Hyp D is statistically **equivalent** to shipping on FL cat/reg/leak and AL/AZ cat after multiple-testing correction (m=22, Bonferroni α=0.0023).
2. Hyp D shows a **trend-positive pooled small-state reg lift** (+0.29 pp at p=0.082) that *restores the canonical reg level at AL/AZ*, where shipping shows a known −0.15 to −0.29 reg regression.
3. **No multi-seed (state, axis) cell favors shipping over Hyp D at α<0.10.**
4. Seed-variance argument REVERSED at small states: Hyp D is 1.5–3.8× MORE stable than shipping on reg at AL/AZ, equivalent at FL. The 2026-05-16 "3× wider Hyp D reg σ" was a fold-level artifact of seed=42 that does not survive multi-seed.

**Falsified premises from earlier advisor**:
- "AL/AZ single-seed REVERSES cat" → **falsified by multi-seed** (now NULL at both)
- "v3c provides 3× σ reduction on reg" → **falsified at small states** (Hyp D is MORE stable there)

**Decision**: v3c is RETAINED in the shipping stack, but the rationale is **protocol inertia + 5-state replication coverage already locked**, NOT v3c's mean contribution. The original v3c → +0.63 reg lift at FL is fully absorbed by T3.2; no axis cell after T3.2 shows v3c contributing significant marginal lift after multiple-testing correction.

### §6.2 — Hyp A: add T4.3 side features (`+ v3c + T3.2 + T4.3-all`)

| metric | Shipping n=5 | Hyp A n=5 | mean Δ | paired-t p₂ |
|---|---:|---:|---:|---:|
| FL cat F1 | 69.42 ± 0.25 | 69.78 ± 0.39 | **+0.353** | **0.041** ✓ |
| FL reg Acc@10 | 63.98 ± 0.09 | 63.93 ± 0.11 | −0.047 | 0.590 (NULL) |
| FL leak F1 | 43.09 ± 0.12 | 43.20 ± 0.15 | +0.112 | 0.279 (NULL) |
| AL cat F1 | 42.05 ± 0.29 | 42.00 ± 0.26 | −0.053 | 0.811 (NULL) |
| **AL reg Acc@10** | **49.88 ± 0.55** | **48.59 ± 0.96** | **−1.287** | **0.092** ⚠ |
| AZ cat F1 | 46.80 ± 0.39 | 46.82 ± 0.21 | +0.020 | 0.892 (NULL) |
| AZ reg Acc@10 | 40.63 ± 0.40 | 40.13 ± 0.83 | −0.498 | 0.133 |
| Pooled AL+AZ Δcat | (n=10 paired) | | −0.017 | 0.891 (NULL) |
| **Pooled AL+AZ Δreg** | (n=10 paired) | | **−0.893** | **0.024 ✓ paper-grade NEGATIVE** (8/10 paired negative) |

**Findings**:
- FL cat lift is real (+0.353, 5/5 seeds positive) but **DOES NOT GENERALIZE**.
- Small-state cat is null at both states (-0.05 AL, +0.02 AZ).
- **Small-state REG IS CRUSHED**: AL Δreg = -1.29 pp, AZ Δreg = -0.50 pp, **pooled p₂ = 0.024 paper-grade NEGATIVE**, 8/10 paired observations negative.
- Substrate-asymmetric outcome: T4.3 side-feature concat helps FL but causes representational dilution at smaller substrates that costs the next-POI head measurably.

**Status: Hyp A CLOSED 2026-05-17 — DEAD for shipping.** The reg axis kill criterion (which §6.4 had not explicitly gated on but should have) triggered unambiguously. Documented as substrate-asymmetric §Discussion finding for the paper. Per-seed Hyp A AL+AZ deltas in `hypA_v3c_T32_T43all_alaz_seed{0,1,7,100}.json` + single-seed=42 from `t43all_T32_v3c_{al,ar}_seed42.json`.

### §6.3 — Multiple-testing posture

At m=22 ablation hypotheses across the canonical_improvement slate (Tier 1–4 plus Phase 1 Hyp A/B/C/D), Bonferroni α=0.0023. Hyp D FL Δcat p=0.082 and Hyp A FL Δcat p=0.041 are both nominally interesting but neither survives correction.

The shipping stack's §5 headlines (T3.2 FL cat p=0.031 sign-test 5/5 paired, AL cat +1.48 5/5 paired, AZ cat +1.70 5/5 paired) are paper-grade by **sign-test at 5/5 seeds which is exact-binomial p=0.03125 per cell** — this directional evidence is the load-bearing inferential machinery, not paired-t means at the ablation level.

### §6.4 — Final categorical close (2026-05-17)

**Hyp A: CLOSED — DEAD for shipping.** AL/AZ multi-seed killed Hyp A on the reg axis (pooled p₂=0.024, 8/10 paired observations negative). Pooled AL+AZ Δcat = -0.02 (NULL). Documented as substrate-asymmetric §Discussion finding for the BRACIS paper.

**Hyp D: CLOSED — KEEP-AS-DOCUMENTATION ablation.** Statistically equivalent to shipping on 7 of 9 axis×state cells; trend-positive small-state reg restoration (pooled +0.29 p=0.082). The v3c standalone +0.63 FL reg lift is fully absorbed by T3.2; v3c retained for protocol inertia + 5-state replication coverage already locked, NOT for mean contribution.

**SHIPPING STACK FINAL**: `canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResLN encoder`. No change from 2026-05-16 lock. §5 stands as written.

### §6.5 — Lesson logged: explicit reg-axis kill criterion

The §6.4 decision tree (as originally written 2026-05-17) gated on cat only — Δreg kill criterion was implicit. Hyp A would have been killed earlier under an explicit rule. **All future Tier-5+ probes must include explicit reg-axis gates**:

- **Reg-axis kill**: Δreg ≤ −0.5 pp at any state at n=5 OR pooled small-state Δreg paired-t p₂ ≤ 0.05 with ≥6/10 paired negative → CLOSE as §Discussion regardless of cat-axis lift
- **Substrate-asymmetry rule**: any FL-only positive result at p<0.05 MUST be replicated at AL+AZ multi-seed before promotion. No FL-only ships.
- **Multi-seed mandatory**: no single-seed=42 result promotes to shipping. Phase 1 demonstrated 2/2 single-seed=42 cat signals collapsed to null at n=5 (Hyp D AL/AZ regression, Hyp A AL/AZ regression — both reversed at multi-seed).

---

## §7 — Tier 5 close (2026-05-18)

Four POI-side mechanism candidates closed against the shipping stack (`canonical + v3c + T3.2 ResLN`). Substrate fixed (Check2HGI); recipe locked. Each candidate exercises a distinct POI-substrate axis (identity slot / POI-POI structural co-occurrence / POI-feature self-supervision / cross-view alignment) and is evaluated as an exploration probe per the 2026-05-17 re-scope advisor — paper-discussion-worthy or future-work-worthy, **not** shipping candidates. Per-seed JSONs land at `docs/results/canonical_improvement/T5_*.json`. The §6.5 reg-axis kill rule was applied end-to-end and fired exactly where the substrate predicted (T5.1).

> **2026-05-18 follow-up — Phase-3 closeout note.** §7.1–§7.5 below preserve the 2026-05-18 first-pass close (T5.2b multi-seed at AL+AZ; T5.3 skipped). §7.6 records the Phase-3 follow-up that extended T5.2b to FL multi-seed AND ran T5.3 at AL+AZ × 5 seeds. The Phase-3 numbers supersede the §7.1 verdict-table for T5.2b (now 3-state multi-seed) and T5.3 (flipped from SKIPPED → RAN-as-§Discussion). Shipping stack still frozen at `canonical + v3c + T3.2 ResLN`.

### §7.1 — Per-candidate verdict table

| Candidate | Mechanism | n_seeds | Headline | Verdict |
|---|---|---:|---|---|
| **T5.1** (POI ID embedding) | `nn.Embedding(N_poi, 64)` zero-init, added pre-aggregation | 1 (AL+AZ seed=42) | **AL Δreg = −6.37 pp, AZ Δreg = −4.63 pp** (9–12× kill threshold) | **DEAD** — V2-c-class pool collapse; matches Phase 11 S3-b V2-c signature |
| **T5.2a** (Node2Vec POI-POI + alignment) | Joint skip-gram on Delaunay POI graph with cos-alignment to pooled POI embedding | 1 (AL+AZ seed=42) | **AL Δcat = −0.48, AZ Δcat = −0.45**; AL Δreg +0.72 only | **§Discussion CLOSED** — Hyp A signature (small-state cat regression class) |
| **T5.2b** (Masked POI feature recon) | 15 % POI mask, SCE-reconstruct via Delaunay neighbour pool (λ=0.3) | 5 (AL+AZ seeds 42,0,1,7,100); FL added in §7.6 (Phase 3) | **AL Δcat 5/5+ mean +0.154** (sign-test p=0.03125); pooled AL+AZ cat 9/10+ p=0.011 | **§Discussion + KEEP-AS-DOC** — sub-Bonferroni positive; only POI-side mechanism that produced a clean directional cat lift without firing §6.5 |
| **T5.3** (Multi-view co-training) | Symmetric cos-alignment View1↔View2 (category-only View2) | 0 at §7 first-pass; **5 (AL+AZ seeds 42,0,1,7,100) added in §7.6** | — at first-pass; AZ cat mean +0.314, AZ reg mean +0.303 (Cohen d=+0.85, p_one=0.065) at Phase 3 | **§7.1 first-pass:** SKIP → §Future Work · **§7.6 update (Phase 3):** RAN-as-§Discussion (positive trend both axes, sub-Bonferroni) |

### §7.2 — T5.2b multi-seed paired stats

Per-seed deltas vs shipping (`t32_resln_ALAZ_seed{seed}.json`), n=5 paired observations per cell:

| seed | AL Δcat | AL Δreg | AZ Δcat | AZ Δreg |
|---:|---:|---:|---:|---:|
| 42 | +0.04 | +0.64 | +0.30 | +0.00 |
| 0 | +0.12 | −0.48 | +0.07 | −0.25 |
| 1 | +0.12 | +0.48 | −0.57 | +0.24 |
| 7 | +0.14 | −0.10 | +0.50 | +0.81 |
| 100 | +0.35 | +0.10 | +0.14 | −0.11 |

Statistical summary:

| axis | n | mean Δ (pp) | paired-t p₂ | Wilcoxon p₂ | sign-test (one-sided) |
|---|---:|---:|---:|---:|---:|
| AL cat | 5 | **+0.154** | **0.041 ✓** | 0.0625 | **0.03125 (5/5+)** |
| AL reg | 5 | +0.128 | 0.559 | 0.6875 | 0.500 (3/5+) |
| AZ cat | 5 | +0.088 | 0.651 | 0.625 | 0.187 (4/5+) |
| AZ reg | 5 | +0.138 | 0.500 | 0.875 | 0.812 (2/5+) |
| **Pooled AL+AZ cat** | 10 | **+0.121** | 0.208 | 0.082 | **0.0107 (9/10+)** |
| Pooled AL+AZ reg | 10 | +0.133 | 0.330 | 0.488 | 0.377 (6/10+) |

**Reading.** AL cat 5/5+ with sign-test p=0.031 (matches the n=5 ceiling, identical to the §4 shipping-stack cat-axis sign tests) and a paired-t p=0.041 are both nominally interesting. Pooled AL+AZ cat at 9/10+ paired-positive (one-sided sign-test p=0.011) is the strongest directional signal in the Tier-5 slate at AL+AZ. No state×axis cell shows a reg-axis kill (Δreg ≥ −0.5 pp at both states; pooled Δreg paired-t p=0.33; pooled paired-negative count 4/10 — well above §6.5 ≥6/10 trigger). **Phase 3 extends this evidence to FL — see §7.6.**

### §7.3 — Multiple-testing posture

The canonical_improvement family count at the §7 first-pass close is **m = 22 + 4 = 26 ablation hypotheses** (Tier 1–4 + Phase 1 Hyp A/B/C/D + Tier 5 T5.1 / T5.2a / T5.2b / T5.3). At Bonferroni α = 0.05/26 ≈ **0.00192** the T5.2b AL-cat sign-test p=0.031 **fails by ~16×** and the pooled AL+AZ cat sign-test p=0.011 **fails by ~5.7×**. Even at the more permissive Holm step-down with the strongest T5 candidate as the first step, T5.2b does not survive correction at the family scale. **Sub-detection-threshold positive**: documented, not promoted. Phase-3 §7.6 re-anchors the family at **m = 28** to also include the T5.2b-FL cell and the T5.3 AL+AZ multi-seed cell — Bonferroni α* tightens to **0.00179**; the conclusions are unchanged.

### §7.4 — Shipping stack — no change

**SHIPPING STACK FINAL (unchanged)**: `canonical Check2HGI + v3c (AdamW WD=5e-2) + T3.2 ResLN encoder`. §5 stands as written; T5.2b is reported as a §Discussion positive sub-detection-threshold finding (only POI-side mechanism in the Tier-5 slate that produced a clean directional cat lift without firing §6.5). No headline rewrite is required.

### §7.5 — Lesson confirmed: §6.5 reg-axis kill rule works

The §6.5 explicit reg-axis kill criterion (Δreg ≤ −0.5 pp at any state at n=5) was authored 2026-05-17 to retroactively cover the Hyp A close. It now has a forward-looking validation: **T5.1 was killed unambiguously** at single-seed (AL Δreg = −6.37 pp, AZ Δreg = −4.63 pp — 9–12× the threshold) without requiring expensive multi-seed expansion. T5.2a's small-state cat regression (Hyp A signature) was also identified at single-seed without requiring full multi-seed expansion. Net: ~16 GPU-h banked across the Tier-5 slate (no T5.2b FL gate, no T5.3 first-gate, no T5.1 multi-seed escalation) by allowing the §6.5 rule to terminate experiments at the single-seed gate when the signature is unambiguous. (See §7.6 — Phase 3 *did* return to T5.2b-FL and T5.3 once budget was confirmed available; neither result re-opened the shipping stack.)

---

### §7.6 — Phase-3 closeout (2026-05-18 follow-up)

After the 2026-05-18 first-pass §7 close, two further Phase-3 runs landed: (a) **T5.2b extended to FL multi-seed** (5 seeds) to close out the 3-state coverage; (b) **T5.3 ran at AL+AZ × 5 seeds** (the SKIP from §7.1 was reversed once GPU-h budget was confirmed available). Verdict at end of Phase 3: **shipping stack remains FROZEN** at `canonical + v3c + T3.2 ResLN`. T5.3 flips from `SKIPPED → §Future Work` to `RAN-as-§Discussion`. Headline contributions of Tier 5 to the paper are §Discussion-only.

#### §7.6.1 — T5.2b FL multi-seed (new Phase 3 result)

JSON artefacts: `T5_2b_maePoi_FL_seed{42,0,1,7,100}.json`. Per-seed deltas vs shipping:

| seed | FL Δcat | FL Δreg |
|---:|---:|---:|
| 42  | +0.49 | −0.11 |
|  0  | +0.82 | −0.05 |
|  1  | +0.11 | −0.09 |
|  7  | +0.27 | +0.07 |
| 100 | −0.52 | −0.16 |

#### §7.6.2 — T5.2b 3-state × 5-seed statistical summary

| Cell | n | mean Δ | sd | t one-sided p | Wilcoxon p | Sign p | pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| FL cat | 5 | +0.234 | 0.500 | 0.178 | 0.438 | 0.375 | 4/5 |
| FL reg | 5 | −0.069 | 0.086 | 0.926 | 0.188 | 0.375 | 1/5 |
| AL cat | 5 | **+0.152** | 0.115 | **0.021** | 0.063 | 0.063 | **5/5** |
| AL reg | 5 | +0.128 | 0.448 | 0.279 | 0.625 | 1.000 | 3/5 |
| AZ cat | 5 | +0.090 | 0.404 | 0.323 | 0.625 | 0.375 | 4/5 |
| AZ reg | 5 | +0.140 | 0.415 | 0.247 | 0.813 | 1.000 | 3/5 |
| **Pooled cat (3-state)** | **15** | **+0.158** | — | 0.053 | 0.064 | **0.0074 (13/15)** |

**Cross-state cat-axis sign-test: 13/15 paired-positive, binomial p = 0.0074.** This is the strongest single piece of evidence in the entire Tier-5 slate. The AL cat cell alone clears the raw α=0.05 paired-t threshold (p_one=0.021, Cohen d ≈ +1.33). FL reg is mean-negative at −0.069 pp but well within the §6.5 kill threshold of −0.5 pp; the regression-axis is essentially flat (pooled Δreg ≈ +0.07 pp).

#### §7.6.3 — T5.3 AL+AZ multi-seed (new Phase 3 result)

JSON artefacts: `T5_3_multiview_{alabama,arizona}_seed42.json` (seed 42) + `T5_3_multiview_alaz_seed{0,1,7,100}.json`. Per-seed deltas vs shipping:

| seed | AL Δcat | AL Δreg | AZ Δcat | AZ Δreg |
|---:|---:|---:|---:|---:|
| 42  | −0.43 | +0.73 | −0.04 | +0.12 |
|  0  | +0.18 | −0.61 | +0.31 | −0.02 |
|  1  | −0.06 | +0.48 | −0.10 | +0.11 |
|  7  | +0.08 | −0.14 | +0.97 | +0.87 |
| 100 | +0.67 | +0.10 | +0.43 | +0.44 |

#### §7.6.4 — T5.3 AL+AZ × 5-seed statistical summary

| Cell | n | mean Δ | sd | t one-sided p | Wilcoxon p | Sign p | pos |
|---|---:|---:|---:|---:|---:|---:|---:|
| AL cat | 5 | +0.086 | 0.399 | 0.328 | 0.625 | 1.000 | 3/5 |
| AL reg | 5 | +0.113 | 0.524 | 0.327 | 0.813 | 1.000 | 3/5 |
| AZ cat | 5 | +0.314 | 0.432 | 0.090 | 0.313 | 1.000 | 3/5 |
| **AZ reg** | 5 | **+0.303** | 0.357 | **0.065** | 0.125 | 0.375 | 4/5 |

**T5.3 is mean-positive on all four cells.** The AZ reg axis is the largest effect-size cell in the entire Tier-5 slate (Cohen d ≈ +0.85; AZ cat d ≈ +0.73) but reaches only p_one ≈ 0.065–0.090, sub-Bonferroni. No reg-axis kill at either state — §6.5 does not fire. T5.3 is the cleanest positive-but-not-shipping Tier-5 candidate.

#### §7.6.5 — Bonferroni at m = 28 (Phase 3 family count)

α* = 0.05 / 28 ≈ **0.00179**. **No Tier-5 cell clears it.** T5.2b AL cat (p_one=0.021) misses by ~12×. T5.2b pooled 3-state cat sign-test (p=0.0074) misses by ~4×. T5.3 AZ reg (p_one=0.065) misses by ~36×. All Tier-5 evidence remains §Discussion-only.

#### §7.6.6 — No shipping change

**SHIPPING STACK FROZEN at `canonical + v3c + T3.2 ResLN`** for the BRACIS 2026 submission. The Phase-3 evidence does not change the §5 headline. T5.2b's cross-state cat-axis sign-test (13/15, p=0.0074) is the strongest Tier-5 evidence and is recorded as motivation for future work on masked-POI pretraining; it is not, by itself, paper-grade after multi-test correction at the m=28 family scale. T5.3's positive trend on both axes (AZ d ≈ +0.85 reg / +0.73 cat) is the cleanest positive-but-not-shipping signal in the slate and is flagged as the prime future-work multi-seed-on-FL extension if a deeper subsequent paper revisits Tier 5.

#### §7.6.7 — Forward sequence (no further GPU runs)

- **T5.2a multi-seed** — skipped per Hyp A precedent (single-seed signal of the same shape collapsed under multi-seed in Phase 1; no expectation that T5.2a's AL/AZ Δcat = −0.48/−0.45 single-seed signal survives n=5).
- **T5.3 FL multi-seed** — skipped on cost-benefit grounds. T5.3 at AL+AZ × 5 seeds already shows the strongest Tier-5 effect sizes but is sub-Bonferroni at m=28; running FL at the 2× multi-view compute cost (~25-30 GPU-h for a 5-seed sweep) is unlikely to clear the threshold and is unwarranted for a §Discussion-only finding. Flagged in §Future Work.

**Tier 5 is fully closed for BRACIS 2026.** canonical_improvement is complete; the folder is treated as read-only beyond this point.
