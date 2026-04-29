# F50 — MTL Proposal Audit and Tiered Exploration Plan

**Date:** 2026-04-28. **Tracker:** see `FOLLOWUPS_TRACKER.md` §1 (F33 done; F50 sub-rows). **Status:** **Tier 0 + T1.1 + T1.2 STL LANDED 2026-04-28**; T1.2 MTL paused at 3/5 folds (HSM ≈ H3-alt — head-capacity hypothesis directionally refuted at n=3); T1.3/T1.4 launchers ready, NOT STARTED. **For pickup state see `F50_HANDOFF_2026-04-28.md`.** **Predecessor docs:** `F49_LAMBDA0_DECOMPOSITION_RESULTS.md`, `F37_FL_RESULTS.md`, `SUBSTRATE_COMPARISON_FINDINGS.md`. **Cross-refs:** `PAPER_PREP_TRACKER.md` §2, `PAPER_DRAFT.md` §1-2, `CONCERNS.md` §C12 §C13 §C14 §C15.

**Cost summary (planned):**
- Tier 0 — analysis-only — ~1 h CPU.
- Tier 1 — four cheap targeted tests — ~14 h Colab T4 (or ~36 h MPS) parallelisable.
- Tier 2 — three architectural alternatives — ~60 h Colab T4 — gated on Tier 1 outcome.
- Tier 3 — bigger pivots — ~60 h+ — gated on Tier 1+2 outcome.

**Review pass (2026-04-28):** advisor-stress-tested ahead of writing; four overshoots in an earlier draft were corrected before this version landed (compute estimates, "must pivot" register softened, F33 added to Tier 1, Δm computability via existing JSONs surfaced). The tiered structure below is what survived that review.

---

## 1 · Question

After F49 (3-way decomposition) and F37 (FL Layer 3 closing) we have a clean 3-state pattern: at 1.1K regions (AL) MTL exceeds matched-head STL on reg by +6.25 pp (paired Wilcoxon p=0.0312); at 1.5K (AZ) it ties; at 4.7K (FL) it loses by **−8.78 pp** (paired Wilcoxon p=0.0312, 5/5 folds negative). The architectural cost grows monotonically with region cardinality (architectural Δ +6.48 / −6.02 / **−16.16** pp).

The cat-side relation is more uniform: MTL > STL at every state (+0.94 to +3.64 pp). Substrate (Check2HGI per-visit context) carries the cat win across states.

This pattern is established. Three open questions remain:

1. **Does MTL Pareto-lose on the joint Δm metric at FL?** A back-of-envelope using the standard Maninis/Vandenhende formulation already suggests yes (see §3.3) but the formal computation with paired Wilcoxon p-values has not been done.
2. **Is the FL architectural cost a property of the cross-attention shared layer, or of the head choice (`next_getnext_hard`'s 4.7K-class flat softmax)?** The current decomposition cannot tell these apart.
3. **Does newer SOTA MTL machinery (FAMO 2023, Aligned-MTL 2023, PLE 2020) close the FL gap, or is the architectural cost fundamental at scale?** We tested PCGrad (NeurIPS 2020) and NashMTL (ICML 2022); the post-2022 negative-transfer literature has multiple drop-in alternatives we have not benchmarked.

This document defines a tiered experimental plan to answer these questions in order of cost-efficiency, with explicit decision routing at each tier.

---

## 2 · Why an audit now

Three current trackers (`FOLLOWUPS_TRACKER`, `PAPER_PREP_TRACKER`, `PHASE2_TRACKER`) have a shared dependency: **CA + TX 5f H3-alt** (~37 h Colab T4) inherits the cat-head + champion-config decisions made on AL/AZ/FL. If a Tier-1 finding changes the FL champion (e.g., FAMO recovers FL, or hierarchical softmax closes the architectural cost), CA + TX must run under the new champion; otherwise the headline table mixes recipes.

**Concretely:** Tier 1 should land *before* P3 launches CA + TX, not concurrently. P3's ~37 h is the longest single critical-path item and is wasted if rerun under a changed champion.

The audit also surfaces unrun ablations from `archive/phases_original/P0–P5` flagged by the 2026-04-27 + 2026-04-28 review docs (CH11 seed variance, CH09 head sweep, CH14 fclass-shuffle, original CH15 transductive-leakage). Some of these are already in `scope/ch14_ch10_p02_decisions.md` as run-or-retire memos; this doc references but does not duplicate that decision queue.

---

## 3 · Empirical state at audit time

### 3.1 Headline numbers — H3-alt champion (`static_weight + cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3, constant`, 5f × 50ep, seed 42)

| State | n_regions | cat F1 | reg Acc@10 | reg MRR | Source JSON |
|---|---:|---:|---:|---:|---|
| AL | 1,109 | 42.22 ± 1.00 | **74.62 ± 3.11** | … | `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1843` |
| AZ | 1,547 | 45.11 ± 0.32 | 63.45 ± 2.49 | … | `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260425_1853` |
| FL | 4,702 | **67.92 ± 0.72** | 71.96 ± 0.68 | … | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045` |

### 3.2 Matched-head STL ceilings (post-F37)

| State | cat STL `next_gru` | reg STL `next_getnext_hard` | Source |
|---|---:|---:|---|
| AL | 38.58 ± 1.23 | 68.37 ± 2.66 | `phase1_perfold/AL_check2hgi_cat_gru_5f50ep.json` + `B3_baselines/stl_getnext_hard_al_5f50ep.json` |
| AZ | 42.08 ± 0.89 | 66.74 ± 2.11 | `phase1_perfold/AZ_check2hgi_cat_gru_5f50ep.json` + `B3_baselines/stl_getnext_hard_az_5f50ep.json` |
| FL | 66.98 ± 0.61 | **82.44 ± 0.38** | F37 P1+P2 (M4 Pro 2026-04-28): `results/check2hgi/florida/next_lr1.0e-04_bs2048_ep50_20260428_0931/` + `results/B3_baselines/stl_getnext_hard_fl_5f50ep.json` |

### 3.3 Back-of-envelope Δm (Maninis et al. CVPR 2019 / Vandenhende TPAMI 2021)

The MTL-survey-standard Δm averages relative-improvement over baseline across tasks:

```
Δm = (1 / T) · Σ_t  (-1)^l_t · (M_{m,t} - M_{b,t}) / M_{b,t}    (× 100%)
```

For our 2-task case with `l_t=0` (higher-is-better) on both heads, M_b = matched-head STL, M_m = MTL H3-alt:

| State | Δ_cat (rel %) | Δ_reg Acc@10 (rel %) | Δm (mean of 2 tasks) | Verdict |
|---|---:|---:|---:|---|
| AL | (42.22−38.58)/38.58 = **+9.43%** | (74.62−68.37)/68.37 = **+9.14%** | **+9.29%** | MTL Pareto-positive |
| AZ | (45.11−42.08)/42.08 = **+7.20%** | (63.45−66.74)/66.74 = **−4.93%** | **+1.14%** | MTL marginal-positive |
| FL | (67.92−66.98)/66.98 = **+1.40%** | (71.96−82.44)/82.44 = **−12.71%** | **−5.65%** | **MTL Pareto-negative** |

These are point estimates without per-fold pairing or significance. **Tier 0 (§4) computes the formal version with paired Wilcoxon p-values from existing JSONs.** The back-of-envelope already shows the direction is unambiguous: MTL Pareto-loses at FL on Δm by ~5-6 percentage points.

If the formal Tier-0 confirms this with significance, the current PAPER_DRAFT framing — "a single MTL model substantially surpasses single-task and published baselines" — is supported only at AL+AZ on the joint metric. At FL the joint metric goes negative because the −12.7% relative reg cost overwhelms the +1.4% cat lift.

**This is the load-bearing missing analysis.** It does not require new compute.

### 3.4 Architectural-Δ pattern (F49 frozen-cat λ=0 vs STL F21c, 5-fold paired)

| State | n_regions | Architectural Δ (frozen − STL) | Wilcoxon | σ |
|---|---:|---:|---:|---:|
| AL | 1.1K | **+6.48 pp** | ~2.7σ from zero | 2.4 |
| AZ | 1.5K | **−6.02 pp** | ~3.7σ from zero | 1.6 |
| FL | 4.7K | **−16.16 pp** | **p=0.0312** (5/5 folds neg) | 12.0 (frozen-side) |

Note the FL frozen-cat reg path has σ=12 (vs 1.4 on loss-side and 0.4 on STL F21c). Per-fold reg-best epochs at FL frozen are {2, 14, 9, 4, 2} — α-growth fails to engage when cat features are random-frozen. This is methodologically subtle: the −16.16 pp magnitude has high variance, but the *sign* is unambiguous (5/5 paired folds negative).

### 3.5 The three structural-incompatibility symptoms (head + shared layer)

Three independent symptoms in the data, each diagnostic of MTL head incompatibility:

| Symptom | Evidence | Implication |
|---|---|---|
| **Disjoint optimal LR regimes (3× ratio)** | F40 / F48-H1 / F48-H2 / F45 — no single shared LR or schedule satisfies both heads. H3-alt's `cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3` is the only configuration that works at AL+AZ. | The shared backbone must serve two objectives whose optimisers have different optima. H3-alt decouples the optimisation as a workaround; it does not align the objectives. |
| **Divergent inductive biases** | Cat head = GRU recurrent + last-token softmax (sequential intent). Reg head = STAN attention + `α·log_T[last_region]` graph prior (positional structure). | Heads consume the shared representation through different mechanisms. AL's small reg cardinality lets one representation serve both; FL's 4.7K cardinality exposes the friction. |
| **Head-size mismatch (3 orders of magnitude)** | Cat softmax: 7 classes. Reg softmax: 1.1K (AL) / 1.5K (AZ) / 4.7K (FL). | Shared backbone's finite capacity is split between a tiny softmax and a huge softmax. Useful 7-way features have no a priori reason to be useful for 4.7K-way discrimination. |

The architectural-Δ pattern (§3.4) is *consistent with* these symptoms: incompatibility grows with reg-head size. This is not proof that "incompatibility" causes the FL flip — confounded with absolute cardinality, sample size, Markov saturation, etc. — but it is a coherent reading.

---

## 4 · Tier 0 — Formal Δm with paired Wilcoxon (analysis only, ~1 h)

### Hypothesis

H_0: MTL H3-alt is Pareto-positive on Δm at all three states (i.e., the cat lift carries the joint metric even at FL where the reg cost is large).

H_1 (per back-of-envelope §3.3): MTL H3-alt is Pareto-negative on Δm at FL (Δm < 0 at p < 0.05).

### Why Tier 0 first

The full per-fold JSONs already exist for AL+AZ+FL × {MTL H3-alt, STL `next_gru` cat, STL `next_getnext_hard` reg}. The Δm computation is per-fold pairwise arithmetic + paired Wilcoxon, no new training. **Cost: ~1 h.** Information value: decides whether `PAPER_DRAFT.md`'s headline "MTL substantially surpasses STL" survives at FL on the joint metric or only at AL+AZ.

If H_1 holds, the paper either:
- Reframes the headline to "MTL is Pareto-positive on the joint metric at small region cardinality (AL+AZ); at large cardinality (FL) the substrate carries the cat win and matched-head STL is the reg ceiling" — keeping CH21 *scale-conditional* per-state, which it already is.
- *Or* runs Tier 1 to test whether a Tier-1 alternative recovers FL Δm.

If H_0 holds (i.e., FL Δm > 0 at p < 0.05), the current framing reads cleanly and Tier 1 is downgraded to nice-to-have.

### Procedure

1. Reuse `scripts/analysis/p4_p5_wilcoxon_offline.py` (the no-scipy port from the overnight session).
2. Extract per-fold pairs (matched fold-i across cells via seed=42 + identical `StratifiedGroupKFold` splits): for each state and each task, `Δ_t,i = (M_{m,t,i} - M_{b,t,i}) / M_{b,t,i}`.
3. Per-fold Δm_i = (Δ_cat,i + Δ_reg,i) / 2.
4. Per-state: report mean(Δm), σ(Δm), n+ / n−, paired Wilcoxon p_two-sided and p_greater(MTL > STL).
5. Repeat with reg=MRR instead of Acc@10 for sensitivity.
6. **Bonus:** also compute the Pareto frontier directly — does the (cat, reg) pair (MTL, STL) dominate either way? If MTL strictly dominates one task and is strictly dominated on the other, frame as "trade-off" rather than "win/lose."

### Acceptance criteria

- **PASS:** FL Δm > 0 at p_greater < 0.10 (n=5 minimum significance is p=0.0312). Paper framing in PAPER_DRAFT.md §1 is supported on Δm at all 3 states.
- **MARGINAL:** FL Δm ≈ 0 within σ. Frame paper as "MTL preserves cat lift, trades reg for joint deployment efficiency" — already the spirit of `PAPER_DRAFT.md`.
- **FAIL:** FL Δm < 0 at p < 0.10 (matches §3.3 back-of-envelope). Paper headline at FL needs to qualify the joint claim or pivot to substrate-first; Tier 1 becomes the question of whether an alternative recipe rescues FL Δm.

### Output artefacts

- `results/paired_tests/F50_T0_delta_m.json` — per-state Δm + Wilcoxon.
- A short `research/F50_DELTA_M_FINDINGS.md` — paragraph-level summary of outcome.
- Update `OBJECTIVES_STATUS_TABLE.md §3` with the Δm row.

### Implementation note

The script needs an extractor function that pulls `mean(top10_acc_indist)` per fold from the joint-best-epoch checkpoint. Existing `p4_p5_wilcoxon_offline.py` has the structure; one new extractor (~30 LOC) suffices.

---

## 5 · Tier 1 — Cheap targeted tests (run *before* P3 launches CA+TX, ~14 h compute)

These four tests answer four specific load-bearing questions about why FL fails. **They are independent and parallelisable on Colab T4.** Total wall-clock if run in parallel: ~6 h. Total compute: ~14 h.

### Critical scheduling note

P3 (CA+TX upstream pipelines + 5f H3-alt) is the longest single critical-path item at ~37 h Colab. P3 inherits the cat-head + champion-config decisions made on AL/AZ/FL. **If any Tier-1 test changes the FL champion, P3 must run under the new champion.** Running Tier 1 and P3 concurrently risks ~37 h wasted compute. Therefore: **Tier 1 → wait for results → reconvene → P3.**

This delays P3 by ~1 day at most. The information value is high enough to justify the delay.

### T1.1 — F33: FL 5f×50ep B3 + `next_gru` cat head (Path A vs Path B decision)

**Status:** **already in `FOLLOWUPS_TRACKER.md` as P1 paper-blocking** but unrun. This audit elevates it to the top of Tier 1 because it gates the cat-head choice for CA+TX.

**Hypothesis:** the post-F27 swap `next_mtl → next_gru` cat head generalises beyond AL+AZ to FL scale (Path A — universal `next_gru`). Null hypothesis: at FL the original `next_mtl` Transformer head is better (Path B — scale-dependent cat head, F32 n=1 saw a −0.93 pp flip).

**Why it matters paperwise:** if Path B applies, the headline table needs a footnote "task_a head is scale-dependent" and CA+TX inherit `next_mtl`, not `next_gru`. The legacy CategoryHeadMTL (`next_mtl`) was a 4-layer × 4-head Transformer designed for the 7-class cat task; it may have more capacity at FL scale than the GRU last-token-softmax bottleneck.

**Cost:** ~6 h Colab T4 (or ~36 h MPS — gated on FL OOM at batch 2048 → use 1024).

**CLI:**

```bash
# On Colab T4, in notebooks/colab_check2hgi_mtl.ipynb
PYTHONPATH=src DATA_ROOT=$DRIVE/ingred OUTPUT_DIR=$DRIVE/output \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
python scripts/train.py \
  --task mtl --task-set check2hgi_next_region \
  --state florida --engine check2hgi \
  --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --cat-head next_gru --reg-head next_getnext_hard \
  --task-a-input-type checkin --task-b-input-type region \
  --folds 5 --epochs 50 --seed 42 --batch-size 1024 \
  --no-checkpoints
```

(This is **already** the H3-alt FL run that landed 2026-04-26 at `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045`. Re-read to confirm this *is* the F33 test under the H3-alt regime; if so, F33 may already be PASSED for cat F1 = 67.92 ± 0.72 vs pre-F27 1f envelope [65.72, 67.06]. Verify per-fold values fall above the pre-F27 envelope minus σ.)

**Acceptance criterion:**
- PASS (Path A): per-fold cat F1 5f mean ≥ 65.5 (within σ of pre-F27 envelope) → universal `next_gru` is committed; CA+TX inherit it.
- FAIL (Path B): per-fold cat F1 5f mean < 65.5 → re-run with `--cat-head next_mtl` ~6 h Colab T4; if it beats `next_gru` by > 1 pp at FL, paper documents scale-dependent cat head and CA+TX inherit `next_mtl`.

**Decision routing:**
- If F33 passes (Path A) → continue to T1.2/T1.3/T1.4.
- If F33 fails (Path B) → CA+TX inherit `next_mtl`; re-run AL+AZ MTL with `next_mtl` if claim of "joint paper recipe" is desired (or accept scale-dependent footnote).

### T1.2 — Hierarchical softmax on the reg head at FL

**Hypothesis:** the FL architectural cost is partly an artefact of the 4.7K-class flat softmax. A hierarchical softmax tree (county → tract) reduces head-side parameter explosion and may close some of the −16.16 pp architectural Δ.

**Mechanism:** replace `nn.Linear(d_model, 4702)` + `LogSoftmax` in the reg head with a 2-level hierarchical softmax. Florida has ~67 counties × ~70 tracts/county (≈ 4,690 — close to 4,702). The natural hierarchy is `state → county → tract` (we are already at state-level so it becomes `county → tract`). A 2-level tree has ~67 + 70 ≈ 137 effective output classes per prediction — a ~34× reduction.

Reference: Mikolov et al. (Hierarchical Softmax in word2vec, 2013); Le & Mikolov (Doc2Vec, 2014); Mnih & Hinton (NIPS 2008). The IEEE study "Effectiveness of Hierarchical Softmax in Large Scale Classification Tasks" (2018) reports performance "degrades as classes increase" but for 4.7K classes with natural hierarchy it remains competitive.

**Why this is informative:** if FL reg lifts under hierarchical softmax, the architectural cost is partly a head-capacity-mismatch artefact (symptom 2.3 above) — fixable. If it doesn't lift, the cost is structural to the cross-attention shared layer.

**Implementation:**

1. New head class `next_getnext_hard_hsm` in `src/models/next_heads/getnext_hard_hsm.py`. Wraps the existing `next_getnext_hard` STAN backbone but replaces the final `Linear → LogSoftmax` with a hierarchical softmax over a 2-level tree.
2. Tree construction utility `src/data/region_hierarchy.py` that reads the FL TIGER shapefile (`data/miscellaneous/tl_2022_*_tract_*.shp`), groups tracts by county, and emits a `(parent_idx, child_idxs)` mapping serialised to `output/check2hgi/florida/region_hierarchy.pt`.
3. Register in `src/models/heads/registry.py`.
4. Loss: hierarchical cross-entropy = `CE(parent) + CE(child | parent)`. For Acc@K evaluation, decompose top-K over the joint distribution `P(parent) · P(child | parent)`.
5. Unit test: round-trip `tract_idx → (county_idx, intra_county_idx) → tract_idx` for all 4,702 FL tracts.

**Cost:** ~6 h dev + ~3 h Colab T4 train.

**CLI:**

```bash
# After dev landing
python scripts/train.py \
  --task next --state florida --engine check2hgi \
  --task-input-type region \
  --next-head next_getnext_hard_hsm \
  --override-hparams transition_path="$OUTPUT_DIR/check2hgi/florida/region_transition_log.pt" \
                     hierarchy_path="$OUTPUT_DIR/check2hgi/florida/region_hierarchy.pt" \
  --folds 5 --epochs 50 --seed 42 --batch-size 2048 \
  --no-checkpoints

# Then MTL with hierarchical reg head at FL
python scripts/train.py \
  --task mtl --task-set check2hgi_next_region \
  --state florida --engine check2hgi \
  --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
  --cat-head next_gru --reg-head next_getnext_hard_hsm \
  --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --folds 5 --epochs 50 --seed 42 --batch-size 1024 --no-checkpoints
```

**Acceptance criterion:**
- STL hierarchical reg ≥ STL flat reg (82.44 ± 0.38) within σ → architecture preserved at the head level.
- MTL H3-alt with hierarchical reg head ≥ MTL H3-alt with flat reg head (71.96 ± 0.68) by Δ > +3 pp → head-capacity mismatch is a load-bearing factor; close ~half the architectural-Δ gap. **Run T1.2 on AL+AZ as well to confirm cross-state.**

**Decision routing:**
- If hierarchical reg head closes the FL gap → T1.2 becomes a paper-relevant finding; promote to a follow-up F-experiment with full 3-state evaluation.
- If hierarchical doesn't help → architectural cost is in the shared layer, not the head; pursue Tier 2 (PLE / Cross-Stitch / MTI-Net).

### T1.3 — FAMO drop-in replacement of `static_weight` at FL

**Hypothesis (speculative):** newer gradient-balancing methods (FAMO, NeurIPS 2023) may handle the FL negative-transfer regime better than `static_weight(0.75)` + per-head LR.

**Mechanism:** FAMO maintains a logit-parameterised task-weight vector `w` with `Σ w_i = 1`, updated via a single backward pass using observed loss-decrease history. O(1) memory and time per step (vs O(k) for PCGrad/NashMTL/CAGrad which need k separate gradient computations). Reference: Liu et al., FAMO: Fast Adaptive Multitask Optimization, NeurIPS 2023, https://arxiv.org/abs/2306.03792 + reference implementation https://github.com/Cranial-XIX/FAMO.

**Domain-gap caveat (critical):** FAMO's reported wins are on NYUv2 (3 dense vision tasks), CityScapes (2 tasks), CelebA (40 binary attributes), QM9 (11 molecular regression). None of these is a long-tail multi-class classification with 4.7K-class softmax. **Transfer of FAMO's claimed advantage to our regime is a guess.** Treat T1.3 as exploratory — *cheap to run and rules out one explanation* if it fails. Do not promise it as a fix.

**Implementation:**

1. New `src/losses/famo.py` based on the reference repo's `FAMO` class — ~120 LOC including the exponentially-weighted loss-history buffer and the closed-form `w` update.
2. Register in `src/losses/registry.py`.
3. Wire to `mtl_cv.py`'s loss-balancing path (existing `NashMTL` integration is the template).
4. Hyperparameter: FAMO has one knob `gamma` (loss-history smoothing); default 0.001 per the paper. Run as-is first.

**Cost:** ~3 h dev + ~3 h Colab T4 (FL only initially).

**CLI:**

```bash
python scripts/train.py \
  --task mtl --task-set check2hgi_next_region \
  --state florida --engine check2hgi \
  --model mtlnet_crossattn --mtl-loss famo --famo-gamma 0.001 \
  --cat-head next_gru --reg-head next_getnext_hard \
  --scheduler constant --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
  --folds 5 --epochs 50 --seed 42 --batch-size 1024 --no-checkpoints
```

**Acceptance criterion:**
- PASS: FAMO MTL FL reg Acc@10 ≥ 75.0 (closes ≥ 3 pp of the 8.78 pp gap) → FAMO is a paper-relevant alternative. Run AL+AZ for confirmation.
- FAIL: within σ of `static_weight` MTL → static_weight is not the bottleneck; cat-supervision-transfer is null *regardless of balancing method* — strengthens CH20.

**Decision routing:**
- PASS at FL → re-run AL+AZ FAMO. New champion candidate; CA+TX inherit FAMO not `static_weight`.
- FAIL → continue to T1.4 (Aligned-MTL); if both fail, gradient-balancing is exhausted and we move to Tier 2 architecture changes.

### T1.4 — Aligned-MTL drop-in replacement

**Hypothesis (also speculative, same domain caveat):** Aligned-MTL (Senushkin et al., CVPR 2023, "Independent Component Alignment for Multi-Task Learning") aligns task gradients via condition-number minimisation of the stacked task-gradient matrix. Designed for high-dimensional task-vectors; reports SOTA on NYUv2 / CityScapes / Pascal-Context.

**Mechanism:** at each step, stack task gradients into a matrix `G ∈ R^{P×T}` (P = parameter count, T = tasks). Compute the condition number κ(G) and rescale per-task gradients to minimise κ. Result: each task contributes "independent components" to the joint update.

**Cost:** ~2 h dev + ~2 h Colab T4. The reference repo provides a PyTorch hook.

**Acceptance criterion + decision routing:** symmetric to T1.3.

**Why both T1.3 and T1.4:** they target different aspects (FAMO = magnitude balancing; Aligned-MTL = direction alignment). If both fail, that triangulates that gradient surgery is not the locus.

### Tier 1 summary table

| ID | What | Status | Cost actual | Decision routing |
|---|---|:---:|:---:|---|
| **T0** | Δm + paired Wilcoxon from existing JSONs | ✅ **DONE 2026-04-28** | ~1 h | FL Δm = −1.63% p_two_sided=0.0625 (5/5 folds neg) — Pareto-loses at n=5 ceiling. AL+AZ Pareto-positive at p=0.0312. Backs CH21 scale-conditional reframing. See `F50_DELTA_M_FINDINGS.md`. |
| **T1.1 (F33)** | Cat-head Path A vs B at FL | ✅ **DONE 2026-04-28** | (no compute) | PASSED — FL 5f cat F1 = 68.21 ± 0.42, all folds above pre-F27 envelope. Universal `next_gru` cat head. Closes C14. See `F50_T1_1_CAT_HEAD_PATH_DECISION.md`. |
| **T1.2** | Hierarchical softmax reg head | dev ✅ DONE; FL STL ⏳ in flight | dev ~2 h actual; STL ~30 min MPS | Head + hierarchy + smoke test passed. STL FL run launched 2026-04-28 17:00 (PID 1385). After STL completes, run MTL with HSM head at FL. |
| **T1.3** | FAMO drop-in | dev ✅ ZERO-COST (already in src/losses/famo) | run ~5 h MPS | Launcher: `scripts/run_f50_t1_3_famo_fl.sh`. Sequenced after T1.2 MTL. |
| **T1.4** | Aligned-MTL drop-in | dev ✅ ZERO-COST (already in src/losses/aligned_mtl) | run ~5 h MPS | Launcher: `scripts/run_f50_t1_4_aligned_mtl_fl.sh`. Sequenced after T1.3. |

After Tier 1: reconvene. If anything in T1.1–T1.4 closes the FL gap, the new recipe is the candidate champion and CA+TX P3 launches under it. If everything in Tier 1 confirms the current finding (FL architectural cost is robust to head + balancer changes), Tier 2 is the next test bed.

---

## 6 · Tier 2 — Architecture changes (gated on Tier 1, ~60 h)

These tests change the shared-layer mechanism, not just the head or balancer. Run *only if* Tier 1 confirms the architectural cost is structural, not artifactual.

### T2.1 — PLE (Progressive Layered Extraction) backbone

**Reference:** Tang et al., "Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations", RecSys 2020 best paper. https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412236

**Mechanism:** L levels of Customised Gate Control (CGC) blocks. Each level has K_s shared experts + K_t task-specific experts per task. Gates select from `{shared experts ∪ own task-specific experts}`. Shared experts can route to any task; task-specific experts only route to their own task. Progressive separation: lower levels are more shared, higher levels more task-specific.

**Why for our problem:** PLE is *the* canonical architectural answer to MTL head incompatibility at scale. Industrial-scale validated (Tencent video recommendation). Directly addresses §3.5's three symptoms:
- Disjoint LR regimes → each task can effectively claim its task-specific experts.
- Divergent inductive biases → task-specific experts can specialise.
- Head-size mismatch → shared experts can be sized per the larger head.

**Hypothesis:** PLE on the same heads recovers FL reg without losing AL gains.

**Implementation hooks:**
- New `src/models/mtl/mtlnet_ple/model.py` (~400 LOC). The existing `mtlnet_dselectk` shares the expert-routing concept and is the closest template; PLE adds the progressive separation and the shared-vs-task-specific expert distinction.
- Register in `scripts/train.py --model mtlnet_ple`.
- Hparams: `n_layers=3`, `n_shared_experts=4`, `n_task_experts=2`, `expert_hidden=256` (default; sweep on AL).

**Cost:** ~15 h dev + ~10 h Colab T4 train (AL+AZ+FL × 5f).

**Acceptance:**
- PLE-MTL ≥ STL F21c on FL reg (closes ≥ 5 pp of −8.78 pp gap) → architectural-cost is fixable; new champion candidate.
- PLE-MTL ≈ H3-alt within σ at FL → architectural cost is structural to MTL-with-shared-features; supports the substrate-first reframing.

### T2.2 — Cross-Stitch Networks

**Reference:** Misra et al., "Cross-Stitch Networks for Multi-Task Learning", CVPR 2016.

**Mechanism:** at each layer, two task streams maintain separate feature maps. A learned per-layer 2×2 matrix `α` mixes them: `[h_a^{l+1}, h_b^{l+1}] = α · [h_a^l, h_b^l]`. Initialised to `α ≈ I` (high self-weight, low cross-weight); learned during training. Pure form of "learned share-vs-task-specific" — no fixed cross-attention K/V flow.

**Why for our problem:** directly tests whether *forced* cross-attention sharing (the K/V leakage from F49 Layer 2) is the FL bottleneck. If learned sharing recovers FL, the architecture issue is "how much to share at each depth", not "share or not". Older method but still a strong baseline on negative-transfer benchmarks.

**Implementation:**
- New `src/models/mtl/mtlnet_crossstitch/model.py` (~250 LOC; cross-stitch is mechanically simple).
- Register; default α-init = 0.9 self / 0.1 cross.
- Two parallel backbones (one per task), L cross-stitch layers between them.

**Cost:** ~10 h dev + ~5 h Colab T4 (AL+AZ).

**Acceptance:**
- Cross-Stitch MTL ≥ STL on FL reg → forced sharing is the bottleneck; PLE/Cross-Stitch is the architectural fix.
- Cross-Stitch fails → architectural cost is fundamental to multi-stream MTL on this task pair at scale.

### T2.3 — ROTAN reg head as STL ceiling reference

**Reference:** Wang et al., "ROTAN: A Rotation-based Temporal Attention Network for Time-Specific Next POI Recommendation", KDD 2024. https://dl.acm.org/doi/10.1145/3637528.3671809

**Mechanism:** rotation-based temporal encoding (analogous to RoPE in language models) on top of self-attention. Reports improvements over GETNext on standard POI benchmarks.

**Why for our problem:** GETNext-hard is currently the matched-head reg ceiling. If ROTAN beats GETNext-hard at single-task on AL+AZ+FL, the *reg ceiling itself* is higher than 82.44 (FL); the −8.78 pp MTL deficit may be larger than reported.

**Implementation:**
- New `src/models/next_heads/rotan.py` (~300 LOC; the reference architecture is in the KDD paper §3).
- Register in head registry.
- Run as STL only first (Tier 2.3a); MTL-with-ROTAN is Tier 3.

**Cost:** ~12 h dev + ~8 h Colab T4 (STL × 3 states × 5f).

**Acceptance:**
- ROTAN STL > GETNext-hard STL by ≥ 2 pp at any state → ceiling is higher; report ROTAN as the headline reg ceiling.
- ROTAN STL ≈ GETNext-hard within σ → GETNext-hard is the right ceiling; current paper unchanged.

### Tier 2 summary

| ID | What | Cost | Decision routing |
|---|---|:---:|---|
| **T2.1** | PLE backbone | ~25 h | If closes FL gap → new champion. |
| **T2.2** | Cross-Stitch | ~15 h | Tests forced sharing as bottleneck. |
| **T2.3** | ROTAN STL | ~20 h | Tests if reg ceiling is higher. |

Run T2.1 + T2.2 in parallel with T2.3 if Tier 1 motivates. T2.3 STL runs are pure single-task and don't share dependencies with T2.1/T2.2.

---

## 7 · Tier 3 — Bigger pivots (gated on Tier 1+2, ~60 h+)

These are the "if-multiple-fails" backstops. Run only if Tier 1 + Tier 2 confirm the FL architectural cost is structural and irreducible.

### T3.1 — Bi-Level GSL prototype reg head

**Reference:** "Bi-Level Graph Structure Learning for Next POI Recommendation", arXiv 2024. https://arxiv.org/html/2411.01169v1

**Mechanism:** two-level graph (POI-level + prototype-level). Cluster the 4.7K FL regions into ~100 prototypes; predict prototype first, then region within prototype. Long-tail mitigation via prototype aggregation. The paper reports significant gains over GETNext + STAN on Gowalla + Foursquare via four joint losses (CE + hierarchy + view-shared contrastive + view-specific orthogonality).

**Why for our problem:** explicitly addresses 4.7K-class long-tail. If hierarchical softmax (T1.2) closed *some* of the gap and prototypes close *more*, the FL architectural cost is largely a long-tail-distribution artefact — fixable.

**Cost:** ~25 h dev + ~10 h Colab T4. The cluster construction (k-means on region embeddings) is upstream and one-time.

### T3.2 — Distillation: STL teacher → MTL student

**Mechanism:** train STL `next_gru` (cat) and STL `next_getnext_hard` (reg) as teachers. Train MTL with a distillation loss `L = 0.5 · L_hard + 0.5 · KL(student_logits, teacher_logits)` per task. Captures STL ceiling but provides single-model deployment.

**Why for our problem:** if the paper truly wants "single-model MTL deployment that doesn't lose to STL", distillation is the most straightforward path. Reference: Hinton et al. (2015) for the basic mechanism; Pal et al. for MTL-specific applications.

**Cost:** ~15 h dev + ~25 h Colab T4 (3 states × {STL teacher × 2, MTL student × 1}).

### T3.3 — Substrate-first paper rewrite (no compute, ~10 h drafting)

**Trigger:** if Tier 1 + Tier 2 + T3.1 + T3.2 all fail to recover FL Δm.

**Reframe:** title pivots from *Beyond Cross-Task Transfer: Per-Head LR + Check-In Embeddings for MTL POI Prediction* to *Substrate Carries the Joint Task: Check-In-Level Embeddings for POI Category and Region Prediction*. MTL is demoted from "method" to "characterised joint deployment". F49 Layer 2 stays as a methodological contribution.

**Why this is a backstop:** the substrate gain is the most generalisable finding (+12-15 pp cat F1 across all 3 states; POI-RGNN external comparison +28-32 pp at FL). The paper has a publishable result with or without the MTL claim.

**Note:** this is *not* a recommendation to pivot now. It is a backstop if all empirical alternatives in Tier 1 + Tier 2 + T3.1 + T3.2 fail. The user's deliberate framing in `PAPER_DRAFT.md §1` already foregrounds the transfer-null finding; that's defensible.

---

## 8 · Decision tree (how to route Tier 0 / 1 outcomes)

```
Tier 0 (Δm + Wilcoxon, ~1 h)
├── PASS at all 3 states (FL Δm > 0 p < 0.10)
│   └── current PAPER_DRAFT framing supported on joint metric;
│       Tier 1 downgraded to nice-to-have insurance for P3.
├── MARGINAL at FL (Δm ≈ 0)
│   └── frame paper as "trade-off, not strict joint win at FL"
│       (already CH21 spirit). Run Tier 1 anyway for P3 insurance.
└── FAIL at FL (Δm < 0 at p < 0.10)
    └── Tier 1 becomes: "does any alternative recover FL Δm?"

Tier 1 (4 tests, ~14 h)
├── F33 PASS (Path A), Tier 1 alt PASS (T1.2/3/4 close FL gap)
│   └── new champion = (cat=next_gru) + new (head | balancer);
│       CA+TX P3 launches under new champion.
├── F33 FAIL (Path B), other Tier 1 PASS
│   └── champion = (cat=next_mtl) + new (head | balancer);
│       CA+TX P3 launches under combined.
├── all Tier 1 FAIL (no alternative recovers FL)
│   └── architectural cost is robust to head + balancer changes;
│       proceed to Tier 2 (PLE / Cross-Stitch / ROTAN).
└── F33 PASS, all alternatives FAIL
    └── current H3-alt champion confirmed; CA+TX P3 launches under it;
        Tier 2 deferred to camera-ready or skipped.

Tier 2 (3 tests, ~60 h)
├── PLE or Cross-Stitch closes FL gap
│   └── new champion architecture; major paper revision; CA+TX rerun.
├── ROTAN STL > GETNext-hard STL
│   └── reg ceiling revised upward; MTL deficit at FL is larger.
└── all Tier 2 FAIL
    └── architectural cost is structural to multi-stream MTL at scale;
        Tier 3 backstops are the only options.

Tier 3 (3 backstops)
├── Bi-Level GSL prototype reg head closes long-tail gap
│   └── reg head moves from GETNext-hard to prototype-aware;
│       paper reports substrate + new reg head; MTL is reported variant.
├── Distillation closes the deployment gap
│   └── paper claim becomes "single-model MTL via distillation";
│       compelling if FL Δm > 0 under distilled MTL.
└── all of Tier 1+2+3.1+3.2 fail
    └── execute T3.3: substrate-first paper rewrite.
```

---

## 9 · Critical-path schedule

```
Day 0 (today)
├── T0 Δm + Wilcoxon analysis (~1 h)         → first decision point
└── F33 verification — read existing FL JSON  → Path A confirmation if cat F1 ≥ 65.5

Day 1
├── T1.2 hierarchical softmax dev (~6 h)
├── T1.3 FAMO dev (~3 h)
├── T1.4 Aligned-MTL dev (~2 h)
└── start FAMO + Aligned-MTL FL training (Colab T4 in parallel)

Day 2
├── T1.2 STL hierarchical softmax FL (Colab T4)
└── T1.2 MTL hierarchical softmax FL (Colab T4)

Day 3 (reconvene)
└── results from T0 + T1.1 + T1.2 + T1.3 + T1.4
    → decide: launch CA+TX P3 OR escalate to Tier 2

Day 4-7 (if Tier 2 needed)
├── T2.1 PLE dev + train
├── T2.2 Cross-Stitch dev + train
└── T2.3 ROTAN STL dev + train

Day 8 onward
└── CA+TX P3 launches under whatever champion Tier 1+2 settled on
```

This delays P3 by ~3 days at most. Compared to the ~37 h of P3 compute, this is small overhead. If Tier 1 confirms the existing champion, the delay is ~1 day.

---

## 10 · F49 Layer 2 elevation recommendation

Per `CLAIMS_AND_HYPOTHESES.md §CH20 Layer 2` and the discussion in `2026-04-28_critical_analysis` (review folder), the methodological finding "loss-side `task_weight=0` ablation is unsound under cross-attention MTL" is broadly applicable to:

- MulT (Tsai et al., ACL 2019) — multimodal cross-attention
- InvPT (Ye et al., ECCV 2022) — inverted pyramid cross-task
- HMT-GRN (Lim et al., SIGIR 2022) — closest competitor in POI domain
- Any cross-task interaction MTL with `task_weight=0` ablation

Currently in `PAPER_DRAFT.md §3.1` it is contribution **C4 of 4** and lives in Methods §3.6 (Methodological Appendix or Limitations). I recommend promotion to **C2 or C3**, between the substrate finding and the architectural finding. Rationale:

1. **Reusable beyond this paper.** The substrate finding is POI-specific. The architectural finding is shared-attention-MTL-specific. The Layer 2 finding applies to any multi-stream MTL with attention.
2. **Refutes prior literature with significance.** F49b shows the legacy +14.2 pp transfer claim is null at ≥ 9σ on FL alone. That's an unusually strong refutation.
3. **Supports the proposed test infrastructure.** The 4 regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py` provide a reusable test pattern that the broader MTL community could adopt.

Concrete change: rename `paper/appendix_methodology.md` to `paper/methodological_note.md` and reference it from `paper/methods.md §3.4` (currently the optimiser regime section) as a forward-pointer rather than an afterthought. This is ~1 h of restructuring; no new compute.

---

## 11 · Risks and explicit speculation flags

### 11.1 Domain-gap caveats on T1.3 (FAMO) and T1.4 (Aligned-MTL)

Both methods report wins on:
- **NYUv2** (3 dense vision tasks: depth, normal, semantic seg)
- **CityScapes** (2 tasks: instance + semantic seg)
- **CelebA** (40 binary attribute classifications)
- **QM9** (11 molecular property regressions)

None of these is a long-tail multi-class classification with 4.7K classes and a graph-prior-augmented reg head. **There is no a priori guarantee these methods help our regime.** Run them because they are cheap; do not promise they will work.

### 11.2 FL frozen-cat instability (per F49c)

Per-fold reg-best epochs on FL frozen-cat λ=0 are {2, 14, 9, 4, 2}. Three of five folds picked very early epochs — symptom of α-growth not engaging when cat features are random. This makes the −16.16 pp architectural Δ noisy (σ_frozen = 12.0 vs σ_loss-side = 1.4). The *sign* is unambiguous (5/5 paired folds negative, p=0.0312) but the *magnitude* is uncertain.

**Implication for Tier 1:** if any Tier-1 alternative changes the FL frozen-cat reg-best epoch distribution, that's a methodological side-finding worth a sentence in the F50 results.

### 11.3 FL Markov-saturation re-emerging

Per `CONCERNS.md §C02`, FL Markov-1-region = 65.05 ± 0.93 Acc@10. STL `next_getnext_hard` = 82.44 ± 0.38 (= Markov + 17 pp). If hierarchical softmax + FAMO + Aligned-MTL each report MTL FL ≈ 65-70 reg Acc@10, the result is "MTL is no better than Markov-1 at FL" — a paper-pivotal finding regardless of the F50 outcome. We are not currently at this floor (MTL H3-alt FL = 71.96 = Markov + 6.91), but Tier 2/3 alternatives that *worsen* reg are possible.

### 11.4 CA + TX scale-curve unknowns

P3 (CA + TX) extends the scale curve from 3 points {AL, AZ, FL} = {1.1K, 1.5K, 4.7K} to 5 points adding CA (~6K) and TX (~5K). The architectural-Δ pattern's monotonicity hypothesis predicts CA architectural Δ ≈ −20 pp and TX ≈ −18 pp. **If CA + TX confirm this monotonicity, the paper has a clean scale-curve story regardless of Tier 1+2 outcomes.** This suggests one viable plan is: launch P3 immediately under H3-alt, accept the scale curve as the paper's story, and treat Tier 1+2 as camera-ready exploration. Tradeoff is the ~37 h Colab compute risk if Tier 1 finds a working alternative mid-flight.

### 11.5 Forgotten items from the original P0–P5 plan

Per the 2026-04-27 + 2026-04-28 reviews and `scope/ch14_ch10_p02_decisions.md`:

- **CH14** unconditional fclass-shuffle ablation — never run; ~30 min on AL.
- **Original CH15** transductive-leakage audit — never run; ~30 min on AL fold 0.
- **CH11** seed variance on champion configs — never run; F8 deferred 2026-04-23.
- **CH09** 5-head full sweep — only F27 binary swap done.
- **CH10** optimiser sweep on FL — only AL covered.

These are independent of F50 but should be resolved (run-or-retire) before paper submission. F50 does not block on them; they are tracked separately in `scope/`.

---

## 12 · Output artefacts (when each tier lands)

```
research/
├── F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md          this file
├── F50_DELTA_M_FINDINGS.md                        Tier 0 outcome (~1 page)
├── F50_T1_CAT_HEAD_PATH_DECISION.md               T1.1 outcome
├── F50_T1_HIERARCHICAL_SOFTMAX_FINDINGS.md        T1.2 outcome
├── F50_T1_FAMO_FINDINGS.md                        T1.3 outcome
├── F50_T1_ALIGNED_MTL_FINDINGS.md                 T1.4 outcome
└── F50_T2_*.md                                    only if Tier 2 runs

results/paired_tests/
└── F50_T0_delta_m.json                            Tier 0 numerical artefact

src/losses/
├── famo.py                                        if T1.3 implemented
└── aligned_mtl.py                                 if T1.4 implemented

src/models/next_heads/
└── getnext_hard_hsm.py                            if T1.2 implemented

src/models/mtl/
├── mtlnet_ple/                                    if T2.1 implemented
└── mtlnet_crossstitch/                            if T2.2 implemented

scripts/
├── analysis/f50_delta_m.py                        Tier 0 driver
└── run_f50_t1_<name>.sh                           Tier 1 launchers
```

---

## 13 · Update path for live trackers

When F50 lands artefacts, update:

- **`FOLLOWUPS_TRACKER.md`** §1 — add F50 multi-row block; mark T1.1 (= F33) as the same row already there.
- **`PAPER_PREP_TRACKER.md`** §2.1 — promote T0 (Δm) into the "headline-blocking" tier; add T1.2-T1.4 as sharpens-claims.
- **`PHASE2_TRACKER.md`** — note the dependency: P3 launches *after* Tier 1 reconvene.
- **`CLAIMS_AND_HYPOTHESES.md`** — add CH22 (or rename per scope/ch15_rename_proposal.md) for the Δm joint-metric claim, once Tier 0 lands.
- **`CONCERNS.md`** — re-open §C13 (AL extrapolation) referencing the scale-curve insurance; close §C14 (cat-head scale-dependence) when T1.1 verifies Path A.

---

## 14 · Cross-references

- **`PAPER_DRAFT.md`** — committed title + 130-word abstract; framing referenced §4 above.
- **`PAPER_PREP_TRACKER.md` §2** — paper-deliverable items P1-P12; P1+P2 done by F37, P3 awaits reconvene per F50 schedule.
- **`PAPER_STRUCTURE.md` §3** — baseline policy; STL `next_getnext_hard` is the matched-head reg ceiling.
- **`NORTH_STAR.md`** — H3-alt champion config; unchanged by F50 unless Tier 1+ reveals an alternative.
- **`FOLLOWUPS_TRACKER.md` §F33** — already-listed paper-blocker that F50 elevates to T1.1.
- **`CONCERNS.md` §C12 §C13 §C14 §C15** — open concerns this audit speaks to.
- **`research/F49_LAMBDA0_DECOMPOSITION_RESULTS.md`** — predecessor (F49 attribution).
- **`research/F37_FL_RESULTS.md`** — predecessor (F37 closing).
- **`review/2026-04-28_compiled_overview.md`** — compiled state at audit time.
- **`review/2026-04-28_critical_analysis.md`** *(if user moves it from /tmp)* — origin of the Tier-1/2/3 structure.
- **`scope/ch14_ch10_p02_decisions.md`** — run-or-retire memo for forgotten P0/P5 items.

## 15 · Sources cited

- FAMO (NeurIPS 2023): https://arxiv.org/abs/2306.03792 + https://github.com/Cranial-XIX/FAMO
- Aligned-MTL (CVPR 2023): "Independent Component Alignment for Multi-Task Learning", Senushkin et al.
- PLE (RecSys 2020 Best Paper): https://dl.acm.org/doi/fullHtml/10.1145/3383313.3412236
- Cross-Stitch (CVPR 2016): "Cross-Stitch Networks for Multi-Task Learning", Misra et al.
- Bi-Level GSL (arXiv 2024): https://arxiv.org/html/2411.01169v1
- ROTAN (KDD 2024): https://dl.acm.org/doi/10.1145/3637528.3671809
- ForkMerge (NeurIPS 2023): https://arxiv.org/abs/2301.12618
- DST / Dropped Scheduled Task (OpenReview 2024): https://openreview.net/forum?id=myjAVQrRxS
- Cross-task Attention (WACV 2023): https://arxiv.org/html/2206.08927
- Hierarchical softmax effectiveness study: https://arxiv.org/pdf/1812.05737
- Balanced Meta-Softmax (NeurIPS 2020): https://proceedings.neurips.cc/paper/2020/file/2ba61cc3a8f44143e1f2f13b2b729ab3-Paper.pdf
- Maninis et al. Δm formulation (CVPR 2019)
- Vandenhende et al. MTL survey (TPAMI 2021)
- Awesome MTL paper list: https://github.com/thuml/awesome-multi-task-learning
- Long-tailed learning systematic review (arXiv 2024): https://arxiv.org/html/2408.00483v1
- MGCL multi-granularity (Frontiers 2024): https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2024.1428785/full
- Selective Task Group Updates (arXiv 2025): https://arxiv.org/html/2502.11986
- Fantastic Multi-Task Gradient Updates in a Cone (arXiv 2025): https://arxiv.org/html/2502.00217
