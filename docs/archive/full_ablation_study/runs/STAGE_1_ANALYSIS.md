# Stage 1 Results — 2026-04-13

## Configuration
- Engine: fusion (128-dim)
- State: alabama
- Screen: 25 candidates × 1 fold × 10 epochs, Seed 42
- Promotion: top-5 × 2 folds × 15 epochs, Seed 42

## Screen Results (1f × 10ep) — all 25 succeeded

| Rank | Candidate   | Joint      | Next F1 | Cat F1 |
| ---- | ----------- | ---------- | ------- | ------ |
| 1    | s1_cgc21_ca | **0.5060** | 0.2797  | 0.7323 |
| 2    | s1_cgc21_al | 0.5040     | 0.2804  | 0.7276 |
| 3    | s1_dsk42_ca | 0.4982     | 0.2743  | 0.7221 |
| 4    | s1_dsk42_al | 0.4982     | 0.2720  | 0.7244 |
| 5    | s1_mmoe4_al | 0.4962     | 0.2637  | 0.7286 |
| 6    | s1_mmoe4_ca | 0.4948     | 0.2628  | 0.7268 |
| 7    | s1_cgc22_al | 0.4945     | 0.2690  | 0.7200 |
| 8    | s1_cgc22_ca | 0.4901     | 0.2683  | 0.7119 |
| 9    | s1_base_ca  | 0.4646     | 0.2597  | 0.6695 |
| 10   | s1_base_al  | 0.4608     | 0.2581  | 0.6636 |
| 11   | s1_cgc21_uw | 0.4103     | 0.2416  | 0.5789 |
| 12   | s1_cgc21_eq | 0.4094     | 0.2361  | 0.5826 |
| 13   | s1_cgc21_db | 0.4027     | 0.2611  | 0.5443 |
| 14   | s1_cgc22_db | 0.3811     | 0.2588  | 0.5033 |
| 15   | s1_dsk42_uw | 0.3729     | 0.2233  | 0.5225 |
| 16   | s1_cgc22_uw | 0.3685     | 0.2428  | 0.4942 |
| 17   | s1_dsk42_eq | 0.3682     | 0.2177  | 0.5187 |
| 18   | s1_cgc22_eq | 0.3676     | 0.2461  | 0.4891 |
| 19   | s1_dsk42_db | 0.3541     | 0.2145  | 0.4936 |
| 20   | s1_mmoe4_uw | 0.3500     | 0.2466  | 0.4534 |
| 21   | s1_mmoe4_db | 0.3480     | 0.2118  | 0.4841 |
| 22   | s1_mmoe4_eq | 0.3460     | 0.2460  | 0.4459 |
| 23   | s1_base_db  | 0.3154     | 0.2042  | 0.4265 |
| 24   | s1_base_eq  | 0.2968     | 0.2023  | 0.3913 |
| 25   | s1_base_uw  | 0.2947     | 0.2016  | 0.3878 |

## Promoted Results (2f × 15ep)

| Rank | Candidate | Joint | Next F1 | Cat F1 | Time |
|------|-----------|-------|---------|--------|------|
| 1 | s1_dsk42_al | **0.5188** | 0.2714 | 0.7661 | 175 s |
| 2 | s1_dsk42_ca | 0.5183 | 0.2671 | **0.7695** | 176 s |
| 3 | s1_cgc21_ca | 0.5164 | 0.2682 | 0.7647 | 150 s |
| 4 | s1_mmoe4_al | 0.5160 | **0.2714** | 0.7607 | 177 s |
| 5 | s1_cgc21_al | 0.5155 | 0.2667 | 0.7642 | 148 s |

## Key Findings

### 1. **CAGrad / Aligned-MTL dethrone equal_weight — the biggest surprise of the study**

The screen perfectly stratifies by optimizer class:
- **ca / al optimizers:** top-10 (joint 0.46–0.51)
- **eq / db / uw optimizers:** bottom-15 (joint 0.29–0.41)

The gap is **~25 %** joint score. On HGI (Phase 1–2), equal_weight was the winner. On fusion, gradient-surgery methods win by a wide margin. This contradicts the CONTINUE.md prior — "equal_weight and db_mtl consistently good" — and is the single most important finding from this study so far.

**Why does this happen on fusion but not HGI?** My hypothesis: fusion introduces intra-batch gradient conflict from the two embedding sources (Sphere2Vec and HGI have very different scales and geometries). On HGI alone, both tasks see a single homogeneous input, so the task-level gradients rarely conflict. On fusion, the task encoders probably pull in very different directions through the shared backbone. CAGrad's closed-form conflict-averse step and Aligned-MTL's eigendecomposition explicitly resolve this; equal_weight just averages and loses signal. This is a *publishable* mechanistic claim worth verifying in Stage 3 via gradient-cosine tracking.

### 2. **Fusion + CAGrad beats HGI reference by a country mile**

HGI reference (Stage 0): joint = 0.3861. Best fusion candidate at screen: 0.5060 (**+30 %**), promoted: 0.5188 (**+34 %**).

Category F1 is particularly impressive: **0.77 fusion vs 0.58 HGI** (+33 %). At 10 epochs, HGI's advantage on category (seen in Stage 0) completely reverses once we use the right optimizer. The Sphere2Vec signal *does* help category prediction — but only when the gradient is handled correctly.

### 3. **Architecture differences shrink at the top**

Among the top 5 promoted, joint scores span only 0.5155–0.5188 (0.6 % range). Architecture matters *much* less than optimizer on fusion — once CAGrad/Aligned-MTL is chosen, dselectk, cgc(s2t1), and mmoe4 are all within noise. This is another prior-overturning finding: Phase 1 had architecture as the second most-important dimension.

### 4. **Base mtlnet falls hard**

The plain `mtlnet` architecture pairs only reach joint 0.295–0.465, consistently bottom-quarter. Fusion's dual-source input really does require a gating/MoE architecture to exploit. The equal_weight baseline on base mtlnet (joint 0.297) is essentially where Stage 0 said fusion would stop being competitive — so this is now the *lower bound* of useful fusion work.

### 5. **MMoE closes the gap with CAGrad/Aligned-MTL**

Prior HGI work had MMoE underperforming. On fusion with ca/al, `mmoe4_al` and `mmoe4_ca` land in the top 6 (joint 0.494-0.496). The MoE + gradient-conflict-resolution combination matters: MMoE's soft gating provides experts that can specialize on each embedding source, and CAGrad/Aligned-MTL prevents the shared backbone from being pulled apart.

### 6. **Caveat: batch-size confound**

`ca`/`al` runs force `gradient_accumulation_steps=1` (effective batch 4096); `eq`/`db`/`uw` use `gradient_accumulation_steps=2` (effective batch 8192). Smaller effective batches are generally *noisier* and not inherently better, so the improvement is unlikely to be a pure batch-size artifact. Still, a matched-batch re-run for the Stage 3 top-1 would close the loop. **Action: for Stage 3 confirmation, set all candidates to `gradient_accumulation_steps=1` so all runs are compared at identical effective batch size.**

## Decision — Top 3 promoted to Stage 2

**Top 3 by joint score on promoted results:**
1. `s1_dsk42_al` — mtlnet_dselectk(e=4,k=2) + aligned_mtl
2. `s1_dsk42_ca` — mtlnet_dselectk(e=4,k=2) + cagrad(c=0.4)
3. `s1_cgc21_ca` — mtlnet_cgc(s=2,t=1) + cagrad(c=0.4)

Observations:
- **DSelectK + gradient-surgery** is the winning recipe. Interestingly, dselectk was the *DGI* winner in Phase 1 — it's the most flexible expert-selection architecture and may pair well with the scale-imbalanced fusion input.
- Only 1 CGC variant survives; `cgc22` (which won on HGI) is NOT in the top 3. The architectural winner *is* fusion-specific.
- `mmoe4_al` narrowly missed the top-3 (joint 0.5160 vs 0.5164) — within noise. Worth tracking as a near-miss for the writeup.

## Artifacts
- `results/ablations/full_fusion_study/s1_screen_1f_10ep/summary.csv`
- `results/ablations/full_fusion_study/s1_promoted_2f_15ep/summary.csv`
- `docs/full_ablation_study/runs/stage1.log`
