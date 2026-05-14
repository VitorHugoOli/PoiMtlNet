# Methods section — draft v0

**Date:** 2026-04-28
**Target:** Submission paper §3 (Methods). ~3-4 pages camera-ready.
**Status:** First-pass draft. Numbers come from `OBJECTIVES_STATUS_TABLE.md`, `NORTH_STAR.md`, `F49_LAMBDA0_DECOMPOSITION_RESULTS.md`. CA+TX rows are placeholders.

---

## 3 Methods

### 3.1 Task definitions

We study joint multi-task learning (MTL) on the pair `(next_category, next_region)` over user check-in sequences from US-state Foursquare data.

- **`next_category`** — primary task. Given a sliding window of 9 prior check-ins (POI embeddings + per-visit context), predict the category of the next check-in over a 7-class flat label set: `{Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel}`. Primary metric: macro-F1 (class-balanced).
- **`next_region`** — auxiliary task. Same input window; predict the census tract of the next check-in. Class cardinality varies by state: AL ≈ 1,109; AZ ≈ 1,250; FL ≈ 4,702; CA ≈ TBD; TX ≈ TBD. Primary metric: Acc@10 on in-distribution regions; secondary: Acc@1, Acc@5, MRR.

We deliberately retire the POI-granularity `next_poi` task that earlier framings (e.g. P0–P2 of an internal roadmap) targeted. Pilot experiments showed that user-grouped 5-fold cross-validation leaves long-tail POIs with insufficient in-fold support, dominating macro-F1 with noise; flat category prediction at 7 classes is the granularity at which prior work (POI-RGNN [cite], MHA+PE [cite], HMT-GRN [cite], MGCL [cite]) reports comparable results. See appendix `SCOPE_DECISIONS.md` for the full rationale.

### 3.2 Embedding substrate — Check2HGI

We embed each check-in (i.e. each individual visit, not each unique POI) into a 256-dim vector using **Check2HGI** [cite], a check-in-level variant of Hierarchical Graph Infomax that augments the graph with per-visit context tokens. Two key properties distinguish Check2HGI from POI-stable embeddings such as HGI:

1. **Per-visit context.** A POI visited at different times (e.g. a Starbucks during commute vs. weekend leisure) receives different vectors. HGI assigns the same vector to all visits of the same POI.
2. **Region-aware aggregation.** Region embeddings (`region_embeddings.parquet`, also 256-dim) are produced jointly with check-in embeddings under the same hierarchical graph; this enables the auxiliary `next_region` task to use a learned region prior (the **STAN-Flow** (`next_stan_flow`) reg head — see §3.4).

**Substrate validation.** Phase-1 (§4.2) confirms that Check2HGI carries a head-invariant +11.58 to +15.50 pp macro-F1 advantage over HGI on `next_category` across two states (AL, AZ; n=5 paired Wilcoxon p=0.0312 each). A POI-pooled counterfactual (replacing per-visit Check2HGI with the per-POI mean) recovers ~28% of the gap; ~72% comes from per-visit context — the dominant mechanism (Phase-1 C4).

### 3.3 Backbone — MTLnet with cross-attention

We adopt **MTLnet** [cite framework paper] as the joint backbone, specialised here as `MTLnetCrossAttn`. The architecture is:

1. **Per-task input modality.** The cat stream consumes check-in embeddings (256-d) over the 9-step window; the reg stream consumes region embeddings (256-d) over the same window. Streams are not concatenated; they enter the backbone as parallel inputs.
2. **Bidirectional cross-attention block.** Stack of N=4 cross-attention blocks. Each block exchanges queries between streams: `cross_ab` (cat queries → reg keys/values) and `cross_ba` (reg queries → cat keys/values). Internal block-FFN (`ffn_a`, `ffn_b`) and LayerNorms (`ln_a*`, `ln_b*`) live in `shared_parameters()` and are trained jointly under MTL loss.
3. **Task-specific heads.** Cat head: `next_gru` (1-layer GRU + classifier; selected over `NextHeadMTL` Transformer default after the F27 ablation showed +2.69 pp macro-F1 on AZ at n=5 paired p=0.0312). Reg head: **STAN-Flow** (registry `next_stan_flow`; legacy alias `next_getnext_hard`) — a STAN-style attention block with a learned graph prior `α · log_T[r_last]` that scales the region transition matrix `T` (computed from training data via `pipelines/region_transition.py`). The graph-prior weight `α` is a learnable scalar in `next_specific_parameters()`. The pattern is inspired by GETNext (Yang et al. 2022); STAN-Flow is **not** a faithful reproduction (GETNext is a next-POI model with friendship + check-in graph priors).

Hidden width d_model=256, 8 attention heads, dropout 0.1.

### 3.4 Optimisation — per-head LR (the H3-alt recipe)

Single-LR schedules (OneCycleLR or constant) consistently fail to satisfy both heads simultaneously: high LR (≥3e-3) lifts reg via α growth in the graph prior, but destabilises the cat encoder; low LR (≤1e-3) preserves cat but starves α growth, leaving reg flat. We resolve this with **per-head learning-rate groups**:

```
cat_lr     = 1e-3  constant   (cat encoder + cat head)
reg_lr     = 3e-3  constant   (reg encoder + reg head, including α)
shared_lr  = 1e-3  constant   (cross-attention blocks + final layer norms)
```

Optimiser AdamW (`weight_decay=0.05`, `eps=1e-8`); 50 epochs; batch size 2048 on AL+AZ, 1024 on FL (memory).

Loss balancing uses `static_weight(category_weight=0.75)`; `L_total = 0.75 · L_cat + 0.25 · L_reg` (both cross-entropy). We do **not** use NashMTL, PCGrad, or GradNorm under this regime — see §3.7 (loss-side ablation methodology) for the soundness analysis that underpins this choice.

This recipe — formally `F48-H3-alt` in our internal nomenclature — exceeds matched-head single-task on `next_region` Acc@10 by **+6.25 pp on AL** (74.62 ± 3.11 vs STL 68.37 ± 2.66) and closes 75% of the gap on AZ (63.45 ± 2.49 vs 66.74 ± 2.11).

### 3.5 Datasets and folds

Five US states from Foursquare check-ins, segmented by census tract:

| State | N check-ins | N POIs | N regions | N users |
|-------|------------:|-------:|----------:|--------:|
| Alabama (AL)  | ~10K  | TBD | 1,109  | TBD |
| Arizona (AZ)  | ~26K  | TBD | 1,250  | TBD |
| Florida (FL)  | ~127K | TBD | 4,702  | TBD |
| California (CA) | TBD | TBD | TBD    | TBD |
| Texas (TX)      | TBD | TBD | TBD    | TBD |

AL+AZ are **dev/ablation** states (used for Phase-1 substrate validation, F27 cat-head ablation, F48 LR-regime sweep, F49 architecture attribution). FL+CA+TX are **headline** states.

**Folds.** 5-fold stratified group K-fold with `groups=user_id` (no user appears in both train and val). Seed=42 across all experiments; the `--no-folds-cache` CLI flag forces deterministic split regeneration so that every cell in the F49 decomposition table is paired across fold-i. This pairing is the basis for paired Wilcoxon tests reported in §4.

### 3.6 Single-task baselines (matched-head policy, post-F27)

To enable matched-head MTL-vs-STL comparison, we report:

- **STL `next_gru` cat 5f** — exact same cat head as MTL champion; F37 (4050 task in flight). Available: AL, AZ. Pending: FL, CA, TX.
- **STL **STAN-Flow** (`next_stan_flow`) reg 5f** — exact same reg head as MTL champion (with graph-prior α). Available: AL = 68.37 ± 2.66 Acc@10; AZ = 66.74 ± 2.11. Pending FL (F37 P2 4050).
- Auxiliary head-sensitivity rows (under STAN, `next_lstm`, `next_single`, head-free linear probe) reported in appendix table for substrate-claim head-invariance.

External baselines (faithful re-implementations in `baselines/{next_category,next_region}/`):
- POI-RGNN, MHA+PE — for `next_category`
- STAN, HMT-GRN, MGCL, ReHDM — for `next_region` (HMT-GRN/MGCL are concept-aligned but on different datasets; we cite for context)
- Markov-1-region (1-step transition probability) — classical floor on reg

### 3.7 Architecture attribution under MTL with cross-attention

A subtle methodological issue arises when ablating MTL contributions under cross-attention. Setting `category_weight = 0` in the MTL loss does not cleanly silence cat-supervision: under cross-attention, the reg-side gradient `∂L_reg/∂θ_cat` flows through the cross-attention K/V projections, training the cat encoder as a reg-helper. We confirm this with four passing regression tests in `tests/test_regression/test_mtlnet_crossattn_lambda0_gradflow.py`.

The clean isolation requires **encoder-frozen** ablation: explicitly setting `requires_grad=False` on the cat encoder + cat head while keeping `category_weight = 0`. This forces a strict architectural decomposition:

```
Full MTL − STL  =  (frozen_λ0 − STL)        ← architecture alone
                +  (loss_λ0 − frozen_λ0)    ← cat-encoder co-adaptation via K/V
                +  (Full MTL − loss_λ0)     ← cat-supervision transfer
```

We use this 3-way decomposition (§4.3, F49) to attribute the H3-alt reg lift to its components. The methodological note (§4.3) applies retroactively to any MTL architecture with cross-task gradient flow (MulT, InvPT, HMT-GRN-style cross-modal attention).

### 3.8 Statistical tests

For paired comparisons (same fold split), we report:
- **Paired Wilcoxon signed-rank** (one-sided greater for substrate/MTL claims; two-sided for null hypotheses).
- For substrate non-inferiority (CH15 reframing), **TOST** with margin δ = 2 pp Acc@10.
- Effect sizes reported in pp (percentage points) on the primary metric, with σ per fold mean.
- `n = 5` paired folds per cell. Exact Wilcoxon p-values (no asymptotic approximation needed at n=5).

---

## Open TODO for Methods section
- Fill in dataset table N check-ins / N POIs / N users per state (CA, TX TBD).
- Citations for MTLnet framework, Check2HGI, POI-RGNN, MHA+PE, HMT-GRN, MGCL, STAN.
- Confirm class label set for `next_category` (Foursquare 7-class consolidation; cite if there's a precedent).
- Add a small architecture diagram (figure 1 candidate).
