# North-Star MTL Configuration

**Status (2026-04-24):** **B3 validated at 5-fold on AL + AZ and 1-fold × 2 on FL. Cat head refined via F27 from `NextHeadMTL` (Transformer) → `next_gru` (GRU). Paper-reshaping F21c finding noted in §§Caveats.** See §Committed config below.

## Committed config (2026-04-24, post-F27)

**B3 champion:**

```
architecture         : mtlnet_crossattn
mtl_loss             : static_weight(category_weight = 0.75)   # reg weight = 0.25
task_a head (cat)    : next_gru                                 # ← updated 2026-04-24 (F27)
task_b head (reg)    : next_getnext_hard                        # STAN + α · log_T[last_region_idx]
task_a input         : check-in embeddings (9-step window)
task_b input         : region embeddings (9-step window)
hparams              : d_model=256, 8 heads, max_lr=0.003, batch=2048, 50 epochs, seed 42
```

**F27 cat-head refinement — 5-fold AZ paired Wilcoxon** (B3 default `next_mtl` cat-head vs B3 `next_gru` cat-head):
| Metric | Δ mean | p_greater | Verdict |
|---|---:|---:|---|
| cat F1 | +2.37 pp | **0.0312** ✅ | significant |
| cat Acc@1 | +4.69 pp | **0.0312** ✅ | significant |
| reg MRR_indist | +1.50 pp | **0.0312** ✅ | significant |
| reg Acc@10_indist | +1.98 pp | 0.0625 | marginal |
| reg Acc@5_indist | +1.69 pp | 0.0625 | marginal |

Per-fold, all 5 cat folds positive on both cat F1 and cat Acc@1. See `research/F27_CATHEAD_FINDINGS.md` and `scripts/analysis/az_b3_cathead_wilcoxon.py`.

## AZ headline numbers under the new B3 (2026-04-24)

| Metric | Value |
|---|---:|
| Cat F1 | **0.4581 ± 0.0130** |
| Cat Acc@1 | **0.4930 ± 0.0067** |
| Reg Acc@10_indist | 0.5382 ± 0.0311 |
| Reg Acc@5_indist | 0.4054 ± 0.0340 |
| Reg MRR_indist | 0.2766 ± 0.0241 |

**vs STL Check2HGI cat (matched-class):** cat F1 0.4208 ± 0.0089 → **Δ = +3.73 pp** (much stronger than the pre-F27 +1.65 pp).
**vs STL STAN (reg ceiling):** reg Acc@10 0.5224 ± 0.0238 → Δ = +1.58 pp (tied within σ).
**vs STL GETNext-hard (F21c matched-head reg baseline):** reg Acc@10 0.6674 ± 0.0211 → Δ = **−12.92 pp** (MTL still trails on reg — F21c finding persists).

## Caveats — the F21c finding

**F21c (2026-04-24):** STL-with-the-graph-prior (`next_getnext_hard` single-task) **outperforms MTL-B3 on region** by 12–14 pp Acc@10 at AL + AZ. Full analysis: `research/F21C_FINDINGS.md`.

This does not invalidate B3 but reframes what MTL is buying:

- **Joint-task single-model deployment:** B3 gives both `next_category` and `next_region` predictions in one forward pass. Two STL models (one GETNext-hard for region + one matched STL cat head) would beat B3 on region by 12 pp but require running two separate models.
- **Cat F1 lift over STL:** MTL-B3 does lift STL cat F1 (AZ +3.73 pp, p=0.0312). This contribution survives F21c.
- **FL-scale PCGrad pathology:** F2's late-stage-handover finding is independent and paper-worthy.

## Validation status across states (post-F27)

| State | Protocol | cat F1 | reg Acc@10 | Status |
|---|---|---:|---:|---|
| AL | 5f × 50ep (pre-F27) | 0.3928 ± 0.0080 | 0.5633 ± 0.0816 | superseded by F31 |
| AL | **5f × 50ep (post-F27, next_gru)** | **0.4271 ± 0.0137** | **0.5960 ± 0.0409** | ✅ **F31 validated, +3.43 cat / +3.27 reg Acc@10** |
| AZ | 5f × 50ep (pre-F27) | 0.4362 ± 0.0074 | 0.5276 ± 0.0392 | superseded |
| AZ | **5f × 50ep (post-F27, next_gru)** ⭐ | **0.4581 ± 0.0130** | **0.5382 ± 0.0311** | ✅ **committed; Wilcoxon p=0.0312 on 3 metrics** |
| FL | 1f × 50ep (pre-F27, F2 + F17 fold 1 ×2) | 0.6623 / 0.6706 | 0.6582 / 0.6655 | prior n=1 |
| FL | **1f × 50ep (post-F27, next_gru)** | 0.6572 | 0.6526 | ⚠️ **F32 — cat F1 −0.93 vs pre-F27 mean**; within n=1 noise but direction flipped |
| FL | 5f × 50ep | 🔴 pending | 🔴 pending | headline run pending; see §F27 scale-dependence note |

## ⚠ F27 scale-dependence flag (2026-04-24)

The cat-head swap `NextHeadMTL → next_gru` **helps AL (+3.43 pp cat F1) and AZ (+2.37 pp, p=0.0312) but slightly hurts FL at n=1 (−0.93 pp cat F1)**. Three paths documented in `research/F27_CATHEAD_FINDINGS.md §Decision`:

- **A:** Commit `next_gru` universally (accept small FL cost for simpler narrative).
- **B:** Scale-dependent — `next_gru` for AL/AZ, `next_mtl` for FL/CA/TX.
- **C:** Run FL 5f B3+gru (~6 h MPS) to resolve definitively.

The NORTH_STAR config above currently reflects **A** pending user decision. If the user picks **B**, the cat head for FL/CA/TX reverts to `next_mtl` (MTLnet's historical default).

## History

### Post-F2 update (2026-04-23 evening)

F2 (`research/B5_FL_TASKWEIGHT.md`) completed all four phases. The Phase B3 configuration **`mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8h`** at n=1 fold on FL delivers:

| Metric | Soft B-M13 (prior north-star) | B3 | Δ |
|---|---:|---:|---:|
| cat F1 | 0.6601 | **0.6623** | **+0.22 pp** |
| reg Acc@10_indist | 0.6062 | **0.6582** | **+5.20 pp** |
| reg Acc@5_indist | 0.3601 | **0.3988** | **+3.87 pp** |
| reg MRR_indist | 0.2555 | **0.2794** | **+2.39 pp** |

B3 Pareto-dominates soft at n=1 on every joint-score metric. The mechanism: cat-heavy weighting triggers a **late-stage handover** — cat head converges fast in early epochs, then the shared backbone becomes available to the region head for the remaining epochs (cat training extends to epoch 42 vs ≤10 for soft/pcgrad/equal-weight).

**Interim policy:** soft remains the reported north-star until F2's follow-up validation lands (§Re-evaluation triggers). If both checks hold, B3 becomes the new universal north-star.

### Follow-up required before committing B3

| Check | Cost | Pass criterion |
|---|:-:|---|
| B3 on FL, 5-fold | ~5–6 h MPS | σ on cat F1 does not pull B3 below soft (soft cat = 66.01 n=1; B3 cat = 66.23 n=1 — σ could be decisive) |
| B3 on AL, 5-fold | ~1 h MPS | B3 does not break the AL cat head (current AL-hard+pcgrad is 38.50 cat F1) |
| B3 on AZ, 5-fold | ~1–1.5 h MPS | B3 preserves the AZ region lift from B-M9d (53.25 Acc@10 with pcgrad) |

Note: static_weight is a simpler optimizer than PCGrad, and AL/AZ already work under the harder PCGrad; low-risk that B3 breaks them.

## Interim choice (still current, pre-B3-validation)

**`mtlnet_crossattn + pcgrad + next_getnext (soft probe) d=256, 8 heads`** (B-M6b on AL, B-M9b on AZ, B-M13 on FL). All paper tables currently reference this config.

If F2 follow-up passes and B3 replaces soft, the migration is a single-string swap in every paper-facing table plus a 5-fold re-run on each state — same wall-clock cost as any scientific revision.

## Why (short version)

| State | soft joint Acc@10 / cat F1 | hard joint Acc@10 / cat F1 | Winner |
|:-:|:-:|:-:|:-:|
| AL 5f | 56.49 / 38.56 | 57.96 / 38.50 | tied within σ |
| AZ 5f | 46.66 / 42.82 | 53.25 / 42.22 | hard on reg (+6.59 outside σ), cat σ-tied |
| FL 1f | **60.62 / 66.01** | 58.88 / **55.43** | **soft** — hard's cat head fails to train |

- **FL is the headline state** (per `CONCERNS.md §C01` — the paper's primary table is FL + CA + TX).
- Hard has a **diagnosed training failure at FL scale** (see `research/B5_FL_SCALING.md` + the 2026-04-23 JSON comparison in `review/2026-04-23_critical_review.md`): cat head's best-val F1 over 50 epochs is 55.43 vs soft's 66.01 under the identical fold split. Not noise — gradient imbalance.
- Soft scales uniformly across AL / AZ / FL. Cat F1 is within σ of the cross-attn + GRU champion at every state.

## What this choice costs us

- The **AZ +1.01 pp MTL-over-STL-STAN** result (53.25 vs 52.24) that currently sits in hard. Soft on AZ lands at 46.66 Acc@10, which is +3.70 pp above Markov-1 but −5.58 below STL STAN. Under soft, AZ reg is framed as "MTL beats Markov" rather than "MTL beats STL".
- The "faithful Yang 2022 SIGIR" framing. Soft is an adaptation (learned probe) rather than the original hard-index formulation.

## What hard is still used for

Hard remains a **reported ablation row**, not retired. In the paper:

> We propose MTL-GETNext-soft as the joint-task model. We report a faithful hard-index variant as an ablation: at region-cardinality ≤ 1.5 K it matches (AL, within σ) or dominates (AZ, +6.59 pp Acc@10, +3.08 pp MRR) the soft adaptation. At 4.7 K-region scale (FL), hard over-dominates the MTL gradient through PCGrad and the category head fails to train (best-val cat F1 0.554 across 50 epochs vs soft's 0.660). We analyse the mechanism in §X and recommend soft as the scale-robust default.

## Re-evaluation triggers

This choice is revisited if **any** of the following lands:

1. **F2 (FL task-weight sweep).** If `task_b_weight < 1` restores FL-hard cat F1 to ≥ 60 while keeping reg Acc@5 lift, hard becomes scale-uniform and is re-promoted as north.
2. **F12 (FL 5-fold hard) with σ showing cat F1 within σ of soft.** Would argue the 10 pp cat gap was n=1 amplification, not training pathology — low likelihood given the `diagnostic_task_best` analysis but empirically checkable.
3. **A new MTL variant** (e.g., per-task weight clipping, prior-magnitude normalisation) that rescues FL-hard without a task-weight hack. Post-paper research direction.

Until one of those lands: **soft is the headline MTL config**.

## Pointers

- Joint-execution comparison: `OBJECTIVES_STATUS_TABLE.md §2`
- Cross-state deltas: `research/B5_MACRO_ANALYSIS.md`
- FL failure-mode diagnosis: `research/B5_FL_SCALING.md` + `review/2026-04-23_critical_review.md §FL-hard training pathology`
- Open follow-ups that can change this: `FOLLOWUPS_TRACKER.md` F2, F12
