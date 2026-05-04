# F37 — FL STL baselines complete (P1+P2)

**Date:** 2026-04-28
**Status:** **NARRATIVE-CHANGING** at FL — CH18 (MTL > STL on reg) flips at scale; F49 Layer 3 closes with strongly negative architectural Δ.
**Cost:** 53 min total on M4 Pro MPS (P1 cat 24.5 min + P2 reg 29 min). Far below the ~8-10h estimate.
**Tracker:** `FOLLOWUPS_TRACKER.md §F37`, `CONCERNS.md §C12 §C15`, `OBJECTIVES_STATUS_TABLE.md §2.5`.

---

## Headline numbers (5-fold × 50ep, seed 42, FL)

| Cell | Acc@10 (5f mean ± std) | Metric |
|---|---:|---|
| **STL `next_gru` cat F1** | **0.6698 ± 0.0061** | macro F1 |
| MTL H3-alt cat F1 (existing) | 0.6792 ± 0.0072 | macro F1 |
| **MTL > STL on cat (FL)** | **+0.94 pp** | survives at FL ✓ |
| ─── | ─── | ─── |
| **STL `next_getnext_hard` reg** | **0.8244 ± 0.0038** | top10_acc (full) |
| MTL H3-alt reg (existing) | 0.7365 ± 0.0125 | top10_acc_indist (per-task best) |
| MTL H3-alt reg (joint-best) | 0.7196 ± 0.0068 | top10_acc_indist (primary checkpoint) |
| **MTL < STL on reg (FL)** | **−8.78 pp** | **CH18 flips at FL** ✗ |
| F49 frozen-cat λ=0 (existing) | 0.6422 ± 0.1203 | top10_acc_indist (primary) |
| **Architectural Δ vs STL (Layer 3)** | **−16.16 pp** | F49 closes at FL |

**Source files:**
- P1: `results/check2hgi/florida/next_lr1.0e-04_bs2048_ep50_20260428_0931/summary/full_summary.json`
- P2: `docs/studies/check2hgi/results/B3_baselines/stl_getnext_hard_fl_5f50ep.json`
- Wilcoxon: `docs/studies/check2hgi/results/paired_tests/FL_layer3_after_f37.json`

## Paired Wilcoxon (n=5, exact)

Fold splits verified identical between MTL H3-alt FL and STL F21c FL (per-fold val sizes match: 31836/31835/31835/31834/31835). Pairing valid.

| Comparison | Δ (pp) | n+/n− | Wilcoxon W+ | p (one-sided) | p (two-sided) |
|---|---:|:---:|:---:|:---:|:---:|
| MTL H3-alt vs STL F21c | **−8.78** | 0/5 | 0.0 | p_less = **0.0312** | 0.0625 |
| MTL H3-alt vs STL F21c (OOD-corrected lower bound) | −9.46 | 0/5 | 0.0 | p_less = 0.0312 | 0.0625 |
| F49 frozen-cat vs STL F21c (architectural Δ — Layer 3) | **−16.16** | 0/5 | 0.0 | p_less = **0.0312** | 0.0625 |
| F49 lossside-λ=0 vs STL F21c | −12.54 | 0/5 | 0.0 | p_less = 0.0312 | 0.0625 |

All four hit Wilcoxon's max-significance ceiling at n=5 (5/5 folds in the negative direction → p_less = 1/32 = 0.03125). Sign is unambiguous.

**Per-fold architectural deltas (frozen − STL F21c):** {−32.22, −31.05, −4.01, −5.42, −8.08}. Folds 0+1 collapse very low; folds 2-4 are mild. Consistent with the FL frozen-cat instability flagged earlier (per-fold reg-best epochs {2, 14, 9, 4, 2} → α-growth fails when cat features are random).

## Per-state pattern across all 3 headline states

| State | Architectural Δ (frozen − STL) | MTL H3-alt vs STL F21c | Verdict |
|-------|-------------------------------:|----------------------:|---------|
| **AL** | **+6.48 pp** (architecture wins) | **+6.25 pp** (MTL > STL) | architecture-dominant |
| **AZ** | −6.02 pp (architecture costs) | −3.29 pp (75% closed) | classical MTL pattern |
| **FL** | **−16.16 pp** (architecture costs heavily; n=5 sig) | **−8.78 pp** (MTL loses; n=5 sig) | STL ceiling ABOVE MTL |

The "architecture-dominant" headline does not generalise across scale. AL is the single state where MTL exceeds matched-head STL on reg.

## Implications for paper narrative

### Claims requiring revision

1. **CH18 (MTL substrate-specific MTL > STL gap closure)** — current Tier-A status: AL exceeds, AZ closes 75%, FL "TBD pending F37". With F37 done, **FL DOES NOT close the gap at all** — F21c FL beats MTL by ~9 pp. CH18 needs reframing:
   - Old: "H3-alt closes/exceeds the matched-head STL gap on reg, validated AL+AZ+FL."
   - New: "H3-alt closes the gap on AL (exceeding by +6.25 pp); on AZ closes 75%; **at FL scale, the matched-head STL ceiling rises above MTL**."

2. **CH20 (architecture-dominant on AL)** — still holds for AL. But the per-state mechanism table now reads:
   - AL: architecture +6.48 pp / co-adapt ≈ 0 / transfer ≈ 0 → architecture wins
   - AZ: architecture −6.02 pp / co-adapt +1.98 / transfer +0.75 → classical
   - FL: architecture **−16.16 pp** / co-adapt +8.27 / transfer −0.52 → architecture is a heavy cost; co-adapt rescues only ~half

3. **CH21 (joint claim — "interactional architecture × substrate, not transfer")** — the "architecture is the lift" framing is **AL-specific**. The joint claim must be restated:
   - Old: "MTL win is interactional architecture × substrate, not transfer."
   - New: "On AL, the MTL reg lift is purely architectural (+6.48 pp from cross-attn alone); on AZ and FL, the architecture costs reg, and cat-encoder co-adaptation only partially recovers. Substrate is necessary for the cat-side advantage at all 3 states; architecture-dominant lift on reg is AL-only."

### Concerns flipping

- **C12 ("hyperparameter mismatch")** — was "resolved 2026-04-27" conditionally on F37. Now F37 has landed; Layer 3 closes with **strongly negative architectural Δ at FL**. **Re-mark C12 as fully resolved** with the new FL evidence.
- **C15 (MTL coupling vs matched-head STL on reg)** — was "resolved 2026-04-26" with the H3-alt recipe. The re-open trigger explicitly stated: "F37 STL FL ceiling lands above MTL-H3-alt FL (71.96)". **F37 STL FL ceiling = 82.44 pp >> 71.96 → C15 RE-OPENS.**

### Limitations section update

The single biggest revision: §6.1 "Two dev-state regime (AL = 10K check-ins)" needs an explicit clause about scale-conditional architectural lift:

> "While the H3-alt recipe achieves an architecturally-dominant reg lift on Alabama (+6.48 pp from cross-attention alone, F49 paired Wilcoxon p=0.0312), this advantage does **not** generalise to FL scale: matched-head STL `next_getnext_hard` exceeds MTL H3-alt by 8.78 pp (5-fold paired Wilcoxon p=0.0312). The F49 architectural decomposition reveals that frozen-cat performance on FL collapses to −16.16 pp below STL, with per-fold variance σ = 12 pp consistent with α-growth failing when cat features are random at 4,702-region scale. We frame the AL result as a regime-specific architectural-dominant lift; the headline contribution at FL is the substrate (CH16), not the joint MTL pipeline."

## What this DOESN'T change

- **CH16 (Check2HGI > HGI on cat)** — still holds head-invariant at AL+AZ; FL Phase-2 grid (F36) will replicate.
- **CH18-substrate (MTL+HGI breaks the joint signal)** — still holds; this is a *substrate-substitution* claim, not a *state-replication* claim.
- **CH20 Layer 1+2 methodology** — loss-side λ=0 unsoundness under cross-attn still applies; encoder-frozen isolation still the clean ablation.
- **Cat-side MTL > STL** — survives at FL (+0.94 pp) although smaller margin than AL/AZ.

## What's now headline-blocking

Updated picture for paper-blocking work (was: F37 closure → narrative locked):

1. ~~F37 (FL STL F21c)~~ — **DONE 2026-04-28**.
2. **CH18 / CH20 / CH21 reframing in paper-facing docs** — narrative-critical edits to NORTH_STAR.md, CLAIMS_AND_HYPOTHESES.md, OBJECTIVES_STATUS_TABLE.md, PAPER_STRUCTURE.md.
3. **F36 (FL Phase-2 substrate grid)** — still queued. Now tests whether substrate findings (CH16, CH18-substrate) replicate at FL even though the architecture-dominant lift doesn't.
4. **CA + TX upstream pipelines (P3)** — still queued. Now tests whether the per-state pattern (AL only is architecture-dominant) generalises further or AL is the outlier.

## Next-step recommendations

- **Update tables in `paper/results.md`** — replace "TBD pending F37" cells with the FL F21c numbers. Especially the F49 3-way decomposition row for FL.
- **Update Limitations §6.1** — add the scale-conditional clause above.
- **Update C15 status** — re-open with FL evidence; mitigation = explicit per-state characterisation in paper.
- **Optional:** Run a multi-seed sweep on FL frozen-cat to nail down whether the −16 pp is a frozen-cat instability artefact or a robust architectural cost.
