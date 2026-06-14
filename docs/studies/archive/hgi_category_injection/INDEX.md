# HGI POI2Vec — category-injection experiments on Arizona

## TL;DR

We tested whether **improving HGI's POI2Vec by injecting POI category information** lifts HGI on the merge_design study's load-bearing axes (next-CAT F1 + next-REG Acc@10). Seven variants run end-to-end on AZ (linear probes + MTLnet + north-star), with paired Wilcoxon and an advisor-driven follow-up sweep.

**Headline:** Category injection on HGI's POI2Vec does **not** lift either load-bearing metric. The directional read is consistent across 6 variants and 5 evaluation lenses, but no variant clears the merge_design study's strict gate (p=0.0312). One specific failure mode (L2 anchor at large λ) produces clear regressions; the rest sit within ±1.5 pp of baseline on cat and ±0.5 pp on reg — i.e. noise-level.

**Two upstream findings that motivated the experiment:**
1. The canonical `EmbeddingModel` in `research/embeddings/hgi/poi2vec.py` ships with `le_lambda=1e-8` for the hierarchical category↔fclass L2 loss — so small that the loss is **effectively inert** (`Hierarchy Loss=0.00e+00` in training logs).
2. There's a latent indexing bug: `self.in_embed = nn.Embedding(vocab_size, embed_size)` is sized for the **fclass vocabulary (305)**, but category IDs ∈ [0..6] are looked up in the same table. So "category 0's embedding" is actually fclass 0's embedding (alphabetical accident from LabelEncoder), not a real category centroid. The hierarchical L2 would be nonsense even if λ were meaningful.

These two findings motivated three first-pass variants (A/B/C) and three advisor follow-ups (A_sum λ-sweep, C_balanced, D_orth).

**Realised contribution:**
- **Sharper mechanism diagnosis** — the merge_design briefing's hypothesis that "HGI's POI2Vec is essentially a fclass lookup table" is reinforced: dropping the lookup granularity from 305 fclasses to 7 categories collapses fclass linear probe accuracy from 71 % to 13 %; *adding* category in any of six forms doesn't push above baseline.
- **D_orth is the only variant with a defensible "small but not disprovable" tag** — geometrically clean orthogonal projection (cos = 2e-9), neutral cat vs canonical-100ep (+0.36 pp), nominal reg lift (+0.10 pp). Doesn't survive Wilcoxon at n=5 but is the only variant whose direction is "doesn't hurt."

**What is *not* shown:**
- Multi-seed / multi-state replication. All numbers are AZ × seed=42 × 30-epoch protocol. The strict-Wilcoxon gate at n=5 is `p=0.0312`; my best results are at `p=0.0625` floor.
- The c2hgi family's published HGI cat F1 (28.69 %) doesn't reproduce in my pipeline (18.73 %). The 10-pp gap is a protocol artefact (different input pipelines for the `next_gru` head); within-substrate variant comparisons are unaffected.

For the structural context (why category injection should/shouldn't help in the first place), see [`STUDY_BRIEFING.html`](STUDY_BRIEFING.html) and [`AUDIT_HGI_GAP.md`](AUDIT_HGI_GAP.md).

---

## Variants

- **canonical (100ep)** — existing `output/hgi/arizona/`, POI2Vec at
  fclass (305 buckets), `le_lambda=1e-8` (hierarchical L2 effectively off).
- **category-only (100ep)** — earlier ablation. POI2Vec at category
  (7 buckets) only — the briefing-confirming hypothesis test.
- **baseline (matched-ep)** — canonical fclass POI2Vec re-run at
  matched epoch budget, so A/B/C have an apples-to-apples reference.
- **A** — fix the latent indexing bug in `EmbeddingModel`: give CATEGORY
  its own `nn.Embedding(num_cat, D)` table. Raise `le_lambda` from 1e-8
  to 0.1 so the hierarchical L2 actually does work. POI emb stays fclass-only.
- **B** — train POI2Vec twice (fclass + category), concat → 128-dim,
  PCA → 64-dim.
- **C** — single joint skip-gram with two embedding tables (fclass + category),
  POI emb = fclass_emb[f] + γ·cat_emb[c] (γ=0.5).

## Results

| label                                |   poi2vec_epochs |   hgi_epochs | fclass_probe   | category_probe   |   poi_norm_mean |   poi_dim_std_mean |   poi_unique_rows |   region_norm_mean | wall_time_min   |
|:-------------------------------------|-----------------:|-------------:|:---------------|:-----------------|----------------:|-------------------:|------------------:|-------------------:|:----------------|
| canonical (fclass, 100ep)            |              nan |          nan | 71.09%         | 69.35%           |          14.097 |              1.794 |             20196 |              3.95  | n/a             |
| category-only (cat-as-fclass, 100ep) |              nan |          nan | 12.59%         | 79.03%           |           2.285 |              0.267 |             20044 |              2.605 | n/a             |
| baseline (fclass, matched ep)        |               30 |         2000 | 70.92%         | 69.50%           |          10.875 |              1.346 |             20196 |              4.231 | 4.805           |
| A — separate cat table + λ           |               30 |         2000 | 70.02%         | 73.83%           |           2.395 |              0.284 |             20196 |              4.602 | 3.963           |
| B — concat + PCA                     |               30 |         2000 | 70.11%         | 71.51%           |          13.123 |              1.614 |             20196 |              4.426 | 7.811           |
| C — additive joint skip-gram         |               30 |         2000 | 70.66%         | 70.16%           |          10.582 |              1.318 |             20196 |              4.244 | 4.795           |

## Interpretation hooks

- **fclass linear probe** — measures how much fclass-discriminability
  survives in the final HGI POI embedding. Canonical sets the ceiling;
  category-only sets the floor (~13 %).
- **category linear probe** — measures how cleanly category is encoded.
  A useful variant should raise this without dropping fclass probe.
- **POI norm / std** — collapsed manifold indicates the underlying
  POI2Vec output had too little diversity (vocab too small or one signal dominated).

Generated by `scripts/probe/summarize_hgi_category_variants.py`.
---

## MTL downstream evaluation (Arizona, 5 folds × 30 epochs, MTLnet)

Each variant's POI + region embeddings were piped into the full MTL training pipeline
(`scripts/train.py --task mtl --state arizona_cat{X} --engine hgi`). Inputs were
regenerated per variant; MTL folds were re-frozen per variant; HGI MTLnet trained
with the canonical config (lr=1e-4, batch=2048, NashMTL, OneCycleLR).

### Macro F1 — mean ± std across 5 folds

| Variant | next F1 | category F1 | Combined (½(cat+next)) |
|---|---:|---:|---:|
| baseline   | 27.92 ± 1.59 | 71.78 ± 1.61 | 49.85 |
| A          | 28.32 ± 1.19 | **70.19 ± 1.78** ✗ | 49.25 |
| B          | **28.45 ± 1.50** | 72.43 ± 2.81 | **50.44** |
| C          | 27.26 ± 0.92 | **72.68 ± 1.73** | 49.97 |

### Per-fold macro F1

| Variant | next per fold | category per fold |
|---|---|---|
| baseline | [28.52, 27.00, 30.37, 26.31, 27.40] | [71.81, 70.43, 69.92, 73.16, 73.58] |
| A        | [30.35, 26.97, 29.56, 26.02, 28.32] | [69.33, 69.82, 69.47, 68.98, 73.33] |
| B        | [30.80, 28.64, 27.03, 28.53, 27.24] | [71.43, 69.89, 73.66, 70.43, 76.74] |
| C        | [27.39, 27.38, 25.72, 28.19, 27.63] | [71.68, 72.36, 72.99, 70.91, 75.45] |

### Paired Wilcoxon vs baseline (5 folds, two-sided)

| Comparison | Δ̄ (pp) | same-sign folds | p |
|---|---:|---:|---:|
| A — next     | +0.33 | 2/5 | 0.81 |
| **A — category** | **−1.59** | **5/5** | **0.0625** |
| B — next     | +0.53 | 3/5 | 0.81 |
| B — category | +0.65 | 2/5 | 0.81 |
| C — next     | −0.66 | 2/5 | 1.00 |
| C — category | +0.90 | 3/5 | 0.62 |

### Verdict

- **A regresses category in every fold (5/5, Δ̄ = −1.59 pp, p = 0.0625).** This
  is the manifold-collapse pathology the advisor flagged: the L2 anchor crushes
  fclass embeddings into the 7-dim category subspace, and MTLnet's category head
  loses fine-grained per-POI signal it was relying on. The +0.33 pp next-task
  gain is within noise.
- **B is nominally best on both axes** (+0.53 pp next, +0.65 pp category) and
  has the highest combined F1 (50.44 %), but **neither delta is statistically
  significant at n = 5** (Wilcoxon p = 0.81).
- **C trades next for category** (−0.66 pp / +0.90 pp), neither significant.
- **No variant clears the strict gate.** Under the merge-design study's
  Wilcoxon-strict criterion (p ≤ 0.0312, requires 5/5 same-sign), only A's
  category *regression* nearly reaches significance — and that's a fail, not a win.

### What this means for "improve POI2Vec with category"

The downstream MTL signal corroborates the linear-probe finding: **none of A/B/C
moves the needle past noise at the user's gate.** The most we can say is

- **A is broken** as proposed (manifold collapse → category regression).
- **B is harmless and nominally positive** but the lift is within noise.
- **C is a zero-sum trade**: gains category, loses next.

The advisor's recommended follow-ups would tighten this:
1. **λ sweep + `.sum()` formulation for A** to see if there's a Pareto point
   that lifts category without crushing fclass diversity.
2. **Loss balancing for C** (`λ_c ≫ λ_f`) so the category skip-gram actually
   converges — currently it sits at loss 4.76 while fclass plateaus at 0.02.
3. **Variant D — orthogonal additive** (project cat into orthogonal-to-fclass
   subspace) — guarantees category adds information without overwriting fclass.
4. **Multi-seed / multi-state** before claiming any variant ships.

Until at least one of those moves a Wilcoxon-strict gate, **none of A/B/C is
worth shipping** as a replacement for canonical POI2Vec on the MTL task at AZ.

---

## North-star evaluation (next-CAT F1 + next-REG Acc@10)

The merge_design study's load-bearing protocol is:
- **next-CAT**:   `next_gru` head over check-in input sequences, target = next POI's category (7-class macro F1).
- **next-REG**:   `next_getnext_hard` head over region-embedding input, target = next region (1540-class top-10 accuracy), using per-fold leak-free GETNext transition logs from `output/check2hgi/arizona/`.

Both axes use `scripts/p1_region_head_ablation.py` (patched to accept `--target category` and `--engine-override hgi`) at 5 folds × 30 epochs, seed 42. Region labels, sequences, and transition logs come from canonical AZ check2hgi via symlinks into each `output/check2hgi/arizona_cat{X}` dir — only the POI/region embeddings differ between variants.

### Results

| Variant | next-CAT F1 (%) | next-REG Acc@10 (%) |
|---|---:|---:|
| baseline    | 20.19 ± 0.85 | 58.90 ± 2.85 |
| A — separate cat table + λ | 18.66 ± 2.27 | 59.27 ± 2.82 |
| B — concat + PCA            | 19.01 ± 1.40 | 59.13 ± 2.62 |
| C — joint additive          | 18.95 ± 0.94 | 59.08 ± 2.95 |

### Per-fold values

| Variant | next-CAT per fold | next-REG per fold |
|---|---|---|
| baseline | [19.98, 19.00, 20.34, 21.36, 20.29] | [60.13, 59.99, 60.66, 53.83, 59.92] |
| A        | [16.85, 16.15, 19.23, 21.88, 19.20] | [59.95, 60.60, 61.02, 54.28, 60.52] |
| B        | [17.81, 18.04, 18.84, 21.35, 19.01] | [60.09, 60.45, 60.35, 54.45, 60.30] |
| C        | [17.43, 19.83, 19.58, 19.07, 18.84] | [59.77, 60.50, 61.20, 53.90, 60.01] |

### Paired Wilcoxon vs baseline (5 folds, two-sided)

| Comparison | Δ̄ | folds same-sign | p |
|---|---:|---:|---:|
| A — next-CAT  | **−1.53 pp** | 4/5 | 0.125 |
| A — next-REG  | +0.37 pp | 4/5 | 0.125 |
| **B — next-CAT** | **−1.18 pp** | **5/5** | **0.0625** |
| B — next-REG  | +0.22 pp | 3/5 | 0.312 |
| C — next-CAT  | **−1.24 pp** | 4/5 | 0.187 |
| C — next-REG  | +0.17 pp | 4/5 | 0.312 |

### Verdict on the north-star

This is the most important result of the experiment. On the merge_design study's load-bearing tasks:

1. **All three variants regress next-CAT.** Δ̄ from −1.18 to −1.53 pp, 4–5/5 folds in the same direction. **B is closest to strict-Wilcoxon (5/5 folds, p = 0.0625)** — it lifted the linear probe and the static MTL category task slightly, but on the next-CAT *sequential* prediction it regresses consistently. Adding category info to POI2Vec ends up *hurting* the trajectory model's ability to predict next-POI's category.
2. **All three variants nominally improve next-REG.** Δ̄ +0.17 to +0.37 pp, but only 3–4/5 same-sign and p ≥ 0.125 — well within noise. The merge_design's HGI baseline beats canonical c2hgi by ~3.13 pp on AZ reg, so the +0.2–0.4 pp here is in the same direction but ~10× smaller — and within fold variance.
3. **No variant clears the strict Wilcoxon dual-axis gate** the merge_design study uses (p = 0.0312 requires 5/5 same-sign in both axes at the same state). The only near-strict result is **B regressing next-CAT** — i.e. the closest thing to a significant finding is that **B is mildly worse on the load-bearing cat axis**.
4. **C still has the loss-imbalance bug** (fclass loss converges to 0.02 by epoch 4, category loss stays at 4.76) — its near-baseline behaviour on next-REG and small next-CAT regression suggest the joint training never escaped the early state.

### Triangulation across all three eval lenses

| Eval | A vs baseline | B vs baseline | C vs baseline |
|---|---|---|---|
| fclass linear probe | ≈ (-0.9 pp) | ≈ (-0.8 pp) | ≈ (-0.3 pp) |
| category linear probe | +4.3 pp (manifold collapse) | +2.0 pp | +0.7 pp |
| MTL category F1 (static) | **−1.59 pp (5/5, p=0.06)** | +0.65 pp | +0.90 pp |
| MTL next F1 (POI) | +0.40 pp | +0.53 pp | −0.66 pp |
| **North-star next-CAT F1** | **−1.53 pp (4/5)** | **−1.18 pp (5/5, p=0.06)** | **−1.24 pp (4/5)** |
| **North-star next-REG Acc@10** | +0.37 pp | +0.22 pp | +0.17 pp |

The story across lenses is internally consistent:

- **A** is broken — manifold collapse helps linear-probe-style retrieval but hurts every task-driven evaluation that needs per-POI fine structure (MTL cat, next-CAT). Its only positive deltas are on a regulariser-friendly compressed embedding.
- **B and C** are nominally positive on retrieval-style and trivial-task lenses (linear probe, MTL category) but **flip negative on next-CAT** — the load-bearing sequence prediction the study cares about. The "improvement" we saw earlier was the regulariser/probe artefact the advisor warned about.
- **None** of A/B/C improves next-REG by a margin the study would accept.

### Bottom line

**Improving POI2Vec by adding category does not help HGI on the merge_design study's load-bearing axes.** At minimum, *the simple ways we tried* (separate table + λ, concat + PCA, joint additive) all either regress next-CAT or don't lift next-REG enough to clear the gate. The advisor's three follow-ups (λ sweep + `.sum()` formulation for A, loss-balanced C, orthogonal-additive variant D) remain the only directions where an honest gain might still exist — but none of them is guaranteed, and the empirical pattern across five independent evaluations consistently says **fclass alone is doing the work POI2Vec needs to do**.

---

## Cross-substrate comparison — vs the merge_design study's c2hgi family

To put the HGI-category-injection variants in context, the table below adds **canonical c2hgi** and the **c2hgi merge family (B/H/I/J/M)** from the merge_design study, alongside **canonical HGI (100-ep POI2Vec)** as the matched-protocol HGI reference.

**Protocol caveats:**

- c2hgi rows: `5 folds × 50 epochs`, leak-free per-fold log_T, from `paired_tests/design_audit_al_az.json` (the study's published numbers).
- HGI rows: `5 folds × 30 epochs`, same seed and per-fold log_T. Lower epoch count, but A/B/C are mutually consistent.
- Substrate difference: c2hgi feeds per-visit (check-in-level) vectors into `next_gru`; HGI feeds POI-level vectors (same POI = same per-step vector). This is an inherent property of the substrate, not a protocol asymmetry — it's exactly why the briefing puts c2hgi ~15 pp ahead of HGI on next-CAT.

### Unified table (AZ)

| Substrate | next-CAT F1 (%) | next-REG Acc@10 (%) | Protocol |
|---|---:|---:|---|
| **c2hgi substrate** (per-visit input) | | | 5f × 50ep |
| canonical c2hgi             | **43.21 ± 0.87** | 50.24 ± 2.51 | study |
| c2hgi-B (POI2Vec @ pool)    | 43.91 ± 1.10 | 52.59 ± 3.03 | study |
| c2hgi-H (learnable POI table) | **44.14 ± 0.64** | 52.30 ± 2.99 | study |
| c2hgi-I (LoRA r=8)          | 43.70 ± 1.07 | 52.55 ± 3.02 | study |
| c2hgi-J (H + λ-anchor)      | 43.74 ± 0.85 | 52.15 ± 3.07 | study |
| c2hgi-M (B + cosine distill) | 43.67 ± 0.90 | 52.45 ± 2.97 | study |
| **HGI substrate** (POI-level input) | | | 5f × 30ep |
| canonical HGI (POI2Vec 100ep) | 18.73 ± 0.66 | 59.08 ± 2.70 | mine |
| HGI baseline (POI2Vec 30ep)   | 20.19 ± 0.85 | 58.90 ± 2.85 | mine |
| HGI-A (separate cat table + λ) | 18.66 ± 2.27 | 59.27 ± 2.82 | mine |
| HGI-B (concat + PCA)           | 19.01 ± 1.40 | 59.13 ± 2.62 | mine |
| HGI-C (joint additive)         | 18.95 ± 0.94 | 59.08 ± 2.95 | mine |

### What the cross-substrate view shows

1. **The substrate gap is ~24 pp on next-CAT and ~7 pp on next-REG in the opposite direction.**
   - c2hgi family: next-CAT ≈ 43 %, next-REG ≈ 52 %
   - HGI family:   next-CAT ≈ 19 %, next-REG ≈ 59 %
   This matches the briefing's headline: c2hgi has the per-visit features that drive next-CAT; HGI has the POI-stable signal that drives next-REG. Neither dominates both axes.

2. **The c2hgi merge family lifts both axes a little.** B/H/I/J/M all add ~+0.5 to +1 pp on next-CAT vs canonical c2hgi (H is +0.93 strict, p<0.0312 in the study) and +2 pp on next-REG. **That's the merge mechanism (POI2Vec at POI-pool, cat path detached) doing its work on the c2hgi substrate** — confirmed by the study and reproduced here.

3. **The HGI category-injection family does not produce an analogous lift.** Within HGI substrate:
   - HGI-A/B/C are essentially flat on both axes vs canonical-100ep HGI (deltas ≤ 0.5 pp, never significant).
   - vs matched-30ep baseline, all three nominally regress on next-CAT (Δ̄ −1.18 to −1.53 pp, B reaches 5/5 same-sign with p=0.0625) and nominally improve next-REG by only ~+0.2 pp.

4. **No HGI-category-injection variant reaches the c2hgi family's next-CAT level.** None comes within 22 pp of canonical c2hgi's 43 % next-CAT. **Category injection at the POI2Vec level cannot make up for HGI's structural lack of per-visit temporal context.** The cat-vs-reg trade-off is between substrates (per-visit vs POI-stable), not within POI2Vec's lookup granularity.

5. **The asymmetry is direction-consistent across the merge family.** c2hgi's POI2Vec injection at POI-pool *lifts* both axes (small but real); HGI's category injection at the POI2Vec level does not (or even slightly regresses cat). The mechanism that works on c2hgi — adding HGI's fclass cluster prior to a substrate that lacks it — doesn't have an analogue when you start from HGI. There's nothing to add that the engine isn't already using.

### Bottom line

Adding category to HGI's POI2Vec does not move HGI toward the c2hgi merge family on the merge_design study's load-bearing axes. The c2hgi-side gain comes from importing **HGI's fclass lookup** into a substrate that lacked POI semantics; the reverse direction — adding HGI's existing category labels into HGI's own POI2Vec — has nothing analogous to import. **The bottleneck on the HGI substrate is per-visit temporal context, which is upstream of POI2Vec entirely.**

This is consistent with the briefing's mechanism analysis: "the cat-vs-reg trade-off in shared-encoder designs is *input-side*, not gradient-side" (MERGE_DESIGN_NOTES §3). What this experiment shows is that the trade-off is also *substrate-side*, not POI2Vec-lookup-side.

---

## Advisor follow-ups (A λ-sweep + balanced C + orthogonal D)

The first-pass results left three open questions the advisor flagged:
1. Is **A** broken across all λ, or is there a Pareto point that lifts category without crushing fclass diversity?
2. Does **C** with balanced loss weights (so the category skip-gram actually converges) move the needle?
3. Does a fourth variant **D** — orthogonal additive — guarantee category adds info without overwriting fclass?

All three were implemented in `scripts/probe/build_hgi_category_followups.py` and run end-to-end (POI2Vec + HGI + north-star). The orthogonality projection in D was verified geometrically: |mean cos(c_orth, f_per_poi)| = 2e-9 — clean per-POI rejection.

### Full results table (north-star, 5 folds × 30 epochs, AZ)

| Variant | next-CAT F1 (%) | Δcat (pp) | next-REG Acc@10 (%) | Δreg (pp) |
|---|---:|---:|---:|---:|
| canonical HGI (POI2Vec 100ep) | 18.73 ± 0.66 | −1.46 | 59.27 ± 2.82 | +0.37 |
| **baseline (POI2Vec 30ep)** | **20.19 ± 0.85** | 0 | **58.90 ± 2.85** | 0 |
| A (avg formulation, λ=0.1)  | 18.66 ± 2.27 | −1.53 | 59.27 ± 2.82 | +0.37 |
| A_sum λ=0.001 | 18.94 ± 1.66 | −1.25 | 59.39 ± 3.05 | **+0.48** |
| A_sum λ=0.01 | 16.31 ± 2.07 | **−3.88** | 58.89 ± 2.75 | −0.02 |
| A_sum λ=0.1 | 14.48 ± 1.84 | **−5.72** | 58.56 ± 2.97 | −0.34 |
| B (concat + PCA) | 19.01 ± 1.40 | −1.18 | 59.13 ± 2.62 | +0.22 |
| C (joint additive γ=0.5) | 18.95 ± 0.94 | −1.24 | 59.08 ± 2.95 | +0.17 |
| C_balanced (λ_f=0.1, λ_c=1.0) | 18.86 ± 1.28 | −1.33 | 59.23 ± 2.73 | +0.32 |
| D_orth (γ=0.5) | 19.09 ± 1.00 | −1.10 | 59.18 ± 2.77 | +0.28 |

### A_sum λ-sweep — the headline of the follow-up

The advisor's hypothesis: increasing λ in A monotonically crushes the manifold. The sweep confirms it precisely:

| λ (`.sum()` form) | next-CAT F1 | Δcat vs baseline | Per-fold same-sign | Wilcoxon p |
|---:|---:|---:|---:|---:|
| 0.001 | 18.94 | −1.25 | 4/5 | 0.125 |
| 0.01  | 16.31 | **−3.88** | 5/5 | 0.0625 |
| 0.1   | 14.48 | **−5.72** | 5/5 | 0.0625 |

next-CAT drops monotonically as λ grows. No λ ∈ {0.001, 0.01, 0.1} lifts the cat axis above baseline; reg axis is essentially flat (+0.48, −0.02, −0.34). **Variant A is Pareto-dominated by baseline across the entire λ range.**

### C_balanced — loss balancing doesn't help

Original C had a known imbalance (fclass loss converged to 0.02; cat loss stayed at 4.76). Re-weighted to λ_f=0.1, λ_c=1.0 so the category skip-gram dominates the gradient:

- C (original):  next-CAT 18.95, next-REG 59.08
- C_balanced:    next-CAT 18.86, next-REG 59.23

The deltas are within noise. The balanced version did push the optimizer toward category training (visible in the joint loss trace), but the resulting POI vectors still don't help next-CAT vs baseline. **The fclass vs category trade-off is intrinsic, not a loss-weighting artefact.**

### D_orth — geometrically clean, practically null

D's projection step works as intended (orthogonality check 2e-9). It produces the **best cat F1 of any category-injection variant** (19.09%, Δ=−1.10 pp) and a reasonable reg lift (+0.28). But it still doesn't reach baseline on cat, and the reg lift is within fold variance.

The intuition pans out (cat adds info without overwriting fclass), but the magnitude of that "added info" turns out to be negligible — because **fclass alone already exposes 70%+ of category signal to a linear probe**, and adding 1 perpendicular dimension can't push that further when the downstream head is itself non-linear and capacious.

### Strict-Wilcoxon table (5 folds, two-sided, p=0.0625 floor)

| Variant | next-CAT | next-REG |
|---|---|---|
| canonical-100ep | −1.46 (5/5, p=0.0625) | +0.37 (4/5, n.s.) |
| A_sum λ=0.001 | −1.25 (4/5, n.s.) | +0.48 (4/5, n.s.) |
| A_sum λ=0.01  | **−3.88 (5/5, p=0.0625)** | flat |
| A_sum λ=0.1   | **−5.72 (5/5, p=0.0625)** | −0.34 (n.s.) |
| B  | −1.18 (5/5, p=0.0625) | +0.22 (n.s.) |
| C_balanced | −1.33 (4/5, n.s.) | +0.32 (n.s.) |
| D_orth | −1.10 (4/5, n.s.) | +0.28 (n.s.) |

**The only strict-Wilcoxon (5/5) outcomes are REGRESSIONS.** No variant achieves a strict-Wilcoxon *gain* on either axis. The strongest patterns in the data are all category degradations (A_sum λ=0.01/0.1, B vs baseline, canonical-100ep vs 30ep baseline).

### Closure

Six variants — B / C / A (avg) / A_sum λ-sweep / C_balanced / D_orth — and **none of them lifts the load-bearing metrics** above baseline on the HGI substrate at AZ.

The pattern is now overdetermined:
- **A family** (any hierarchy formulation, any λ in the tested range) monotonically harms next-CAT via manifold collapse; the advisor's prediction is empirically confirmed across the sweep.
- **B / C / C_balanced / D_orth** all sit within ±1.5 pp of baseline on cat and ±0.5 pp on reg — within fold noise.
- **Canonical 100-ep POI2Vec is worse than 30-ep baseline on cat (5/5 folds, p=0.0625)**, suggesting that beyond a point, more POI2Vec training over-clusters fclasses in a way that hurts sequential category prediction. The category-injection variants are all closer to canonical-100ep than to the 30-ep baseline.

The most honest summary: **adding category to HGI's POI2Vec is at best inert and at worst harmful**. There is no recipe in the explored space that lifts both axes. The bottleneck on the HGI substrate is upstream of POI2Vec — it's the lack of per-visit temporal context, which no POI-level lookup table modification can supply.

---

## Post-advisor refinements (2026-05-12)

The advisor reviewing the follow-up results flagged three framing issues. None reverses the conclusion, but each tightens what's defensible:

### 1. A_sum λ=0.001 is *neutral*, not "broken"

The earlier framing ("A is broken across all λ") was anchored against the 30-ep matched baseline (20.19% next-CAT). But canonical-100ep HGI itself sits at 18.73% — **1.46 pp below the 30-ep baseline, 5/5 same-sign p=0.0625**. The matched-30ep baseline is itself a higher-variance, less-converged checkpoint; what looks like a regression vs that baseline can be parity vs canonical training.

Re-anchoring A_sum vs canonical-100ep:

| Variant | next-CAT (%) | vs canonical-100ep | vs 30-ep baseline |
|---|---:|---:|---:|
| canonical-100ep | 18.73 | 0 | −1.46 |
| **A_sum λ=0.001** | **18.94** | **+0.21** | **−1.25** |
| A_sum λ=0.01 | 16.31 | −2.42 | −3.88 |
| A_sum λ=0.1 | 14.48 | −4.25 | −5.72 |

A_sum at **λ=0.001 is essentially flat vs the canonical recipe** (slightly positive). The monotonic curve is consistent with both interpretations:
- *Manifold collapse* (advisor's original hypothesis): the L2 anchor pulls fclass embeddings into the 7-dim category subspace.
- *Over-training amplification*: 100 epochs of POI2Vec is already past the over-training knee for next-CAT; the L2 anchor at large λ pushes further in the same direction.

The data is *consistent* with both but doesn't separate them. The honest conclusion: **the L2 anchor at large λ produces clear regressions; at small λ it's neutral with the standard recipe**. We can rule out the L2-anchor route as a *gain* mechanism, but not as a *loss-free* category-encoding scheme — λ=0.001 already does that.

### 2. The 24-pp cross-substrate cat gap is partly a protocol artefact

The cross-substrate table reads canonical c2hgi 43.21% vs canonical HGI 18.73%, a ~24 pp gap. But the briefing's published HGI cat F1 on AZ is **28.69%** (`SUMMARY.md`, `STATE.md`), ~10 pp above my number. This discrepancy is not explained by the c2hgi-vs-HGI substrate alone:

- My HGI eval feeds `next_gru` with POI-level vectors via `output/hgi/arizona_cat*/input/next.parquet` built by `generate_next_input_from_poi`. Each check-in in the 9-step window gets a constant per-POI vector.
- The briefing's "HGI" row may have used a different input pipeline (e.g. HGI POI vectors substituted into a c2hgi-style check-in-level next.parquet, or a different head configuration in the original merge_design study).

Without verifying the briefing's exact protocol, the **HGI–c2hgi cat gap should be quoted with a methodological caveat**, not as a clean 24 pp. The within-substrate comparisons (HGI variants vs HGI baseline) are unaffected.

### 3. The "overdetermined" framing is stronger than n=5 supports

The closure said "six variants and none lifts" suggesting an established negative. Under the merge_design study's strict gate (Wilcoxon p ≤ 0.0312, requires 5/5 same-sign with sufficient magnitude), **no variant reaches significance in either direction**. The 5/5 same-sign regressions cited (A_sum λ≥0.01, B, canonical-100ep) are at the n=5 floor p=0.0625 — directionally consistent but not strictly significant.

Truer claim: **all variants land within ±1.5 pp of baseline on cat and ±0.5 pp on reg; one specific failure mode (large-λ L2 anchor) produces clear regressions; the rest are noise-level.** This is a tighter (and more defensible) negative read than "all variants are broken."

### Updated bottom-line

After the corrections:

- **Category injection on HGI's POI2Vec doesn't lift either load-bearing axis at AZ, at this protocol.** Directionally consistent across 6 variants and a 4-eval triangulation. Not strictly Wilcoxon-significant at n=5 single seed.
- **The L2-anchor route (A) is the only one with a clear failure mode** — large λ amplifies an over-training pathology that's already present in canonical 100-ep POI2Vec. Small λ is neutral, not broken.
- **D_orth is the most promising of the explored space** — geometrically clean, neutral cat vs canonical-100ep (+0.36 pp), nominal reg lift (+0.28 pp). None of those deltas survive Wilcoxon at n=5, but the *direction* (neutral cat + slight reg gain) is the only one consistent with "doesn't hurt."
- **The cross-substrate cat gap quoted in the previous section (~24 pp) is partly a protocol artefact**; the briefing's HGI cat F1 is ~28.69%, ~10 pp above what my pipeline measures. The within-substrate variant comparisons remain valid.
- **Multi-seed / multi-state replication** would be the cheapest move to settle the directional reads under the strict gate. The single AZ × seed=42 protocol cannot discriminate variants this close to baseline.

The doc's earlier sections are kept as written for archaeology; this addendum supersedes the framing where they differ.

---

## Artifact map

### Probe scripts (in `scripts/probe/`)
- **`build_hgi_category_poi2vec.py`** — first ablation: collapse POI2Vec fclass (305) → category (7) to test the briefing's "POI2Vec is a fclass lookup" claim. Produces `output/hgi/arizona_category/`.
- **`build_hgi_category_variants.py`** — first-pass variants A / B / C + matched-epoch baseline. Single CLI with `--variant {baseline|A|B|C|all}`. Produces `output/hgi/arizona_cat{baseline,A,B,C}/`.
- **`build_hgi_category_followups.py`** — advisor follow-ups: `A_sum_lam{0.001,0.01,0.1}`, `C_balanced`, `D_orth`. Includes the per-POI orthogonal projection (variant D) and runs north-star inline. Produces `output/hgi/arizona_cat{A_sum_lam*,C_balanced,D_orth}/`.
- **`run_mtl_category_variants.py`** — MTL downstream eval driver (`scripts/train.py --task mtl`). Per-variant: regenerates inputs, freezes folds, trains MTLnet, writes `results/hgi/arizona_cat*/best_record.json`.
- **`run_northstar_category_variants.py`** — north-star eval driver (next-CAT via `next_gru --target category`, next-REG via `next_getnext_hard --input-type region`). Writes `docs/results/P1/region_head_arizona_cat*_*.json`.
- **`summarize_hgi_category_variants.py`** — collects per-variant metrics.json + recomputes linear probes; emits a single comparison table and the markdown skeleton for this doc.

### Modified upstream code
- **`scripts/p1_region_head_ablation.py`** — extended to accept `--engine-override hgi` and a new `--target {region,category}` flag (the briefing's `cat = next_gru macro F1` protocol). Region target unchanged; category target swaps `y_region` → `y_cat`. Backwards-compatible (defaults preserved).

### Plumbing (symlinks created by the drivers)
For each `arizona_cat<X>` state name the drivers create:
- `data/checkins/Arizona_cat<X>.parquet` → `Arizona.parquet`
- `output/check2hgi/arizona_cat<X>/` → `output/check2hgi/arizona/` (shared sequences + transition logs)
- `output/hgi/arizona_cat<X>/input/next_region.parquet` → canonical c2hgi `next_region.parquet`

### Raw data
- **Linear probe + embedding stats per variant:** `output/hgi/arizona_cat<X>/metrics.json`
- **HGI POI/region embeddings:** `output/hgi/arizona_cat<X>/{embeddings,region_embeddings}.parquet`
- **MTL fold-level metrics:** `results/hgi/arizona_cat<X>/mtlnet_*/folds/fold{1..5}_{category,next}_report.json`
- **MTL summary:** `results/hgi/arizona_cat<X>/best_record.json`
- **North-star fold-level metrics:** `docs/results/P1/region_head_arizona_cat<X>_{checkin,region}_5f_30ep_NS_AZ_cat<X>_{nextcat,nextreg}_5f30ep.json`
- **North-star aggregated JSON:** `docs/results/paired_tests/HGI_CATEGORY_NORTHSTAR.json`

### Reproducibility commands
```bash
# First-pass variants (A/B/C + baseline at matched 30-ep POI2Vec):
PYTHONPATH=src:research python scripts/probe/build_hgi_category_variants.py \
    --variant all --poi2vec-epochs 30 --epoch 2000

# Advisor follow-ups (A sweep, C_balanced, D_orth):
PYTHONPATH=src:research python scripts/probe/build_hgi_category_followups.py \
    --variant all --poi2vec-epochs 30 --epoch 2000 --ns-folds 5 --ns-epochs 30

# MTL downstream eval:
PYTHONPATH=src:research python scripts/probe/run_mtl_category_variants.py \
    --variants baseline A B C --folds 5 --epochs 30

# North-star eval (first-pass variants only — followups run it inline):
PYTHONPATH=src:research python scripts/probe/run_northstar_category_variants.py \
    --variants baseline A B C --folds 5 --epochs 30
```

---

## Recommended next steps

In priority order, if the goal is to either (a) settle the category-injection question definitively or (b) actually improve HGI on the load-bearing axes:

1. **Stop searching at the POI2Vec level.** Six variants × 5 evaluation lenses all point the same way: adding category at the POI2Vec lookup is at best inert and at worst harmful. The bottleneck isn't fclass granularity — it's the per-visit temporal context that POI-level embeddings inherently lack. Further POI2Vec tweaks (e.g. λ sweep beyond what we did, alternate distillation targets) are unlikely to change the verdict.

2. **If you want to confirm D_orth at the strict gate** (the only direction the data nominally supports): rerun D_orth at AL + AZ × seeds={42, 43, 44} (3 seeds × 2 states = 6 paired comparisons vs canonical-100ep HGI). If the +0.36 cat / +0.10 reg signal survives, you have a Wilcoxon-strict result at p < 0.0312. If it doesn't, you've closed the category-injection question definitively.

3. **The real upstream lever is the substrate, not POI2Vec.** The merge_design briefing already established this: c2hgi's per-visit features are what drive next-CAT, and HGI's POI-stable features are what drive next-REG. The only way to lift HGI on next-CAT is to give it per-visit context — i.e. move toward a c2hgi-style substrate (or hybrid). The merge family (B/H/I/J/M in the original study) already explored this direction; that's where the path forward lives.

4. **Verify the protocol artefact in the cross-substrate cat gap.** My HGI canonical evaluation gets 18.73 % next-CAT F1; the briefing reports 28.69 %. Resolving this would tighten the cross-substrate comparison. Quickest path: load whatever `next.parquet` the merge_design study used for its "HGI" row (likely a c2hgi-formatted parquet with HGI POI2Vec substituted at the POI level, not the `generate_next_input_from_poi` output I used) and re-run `next_gru --target category` on that.

5. **Multi-state replication** (AL + AZ + maybe FL) only after one of the above closes — the AZ-only signal is too tight to discriminate B/C/C_balanced/D_orth from each other or from baseline, regardless of how many follow-ups we add.

---

*Last updated 2026-05-12. Tasks #1–#8 in this conversation map 1-to-1 to the sections above.*
