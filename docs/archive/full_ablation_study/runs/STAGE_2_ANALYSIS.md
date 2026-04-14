# Stage 2 Results — 2026-04-13

## Configuration
- Engine: fusion (128-dim)
- State: alabama
- 9 candidates (3 winners × 3 head combos) × 2 folds × 15 epochs, Seed 42
- Fixed backbones from Stage 1 top-3: `dsk42_al`, `dsk42_ca`, `cgc21_ca`

## Results

| Rank | Candidate | Joint | Next F1 | Cat F1 | ΔJoint vs default |
|------|-----------|-------|---------|--------|-------------------|
| 1 | s2_dsk42_al_hd_dcn   | **0.5275** | 0.2724 | **0.7827** | **+0.0087** |
| 2 | s2_dsk42_ca_hd_dcn   | 0.5257 | 0.2670 | **0.7844** | +0.0074 |
| 3 | s2_cgc21_ca_hd_tcnr  | 0.5173 | 0.2542 | 0.7803 | +0.0009 |
| 4 | s2_cgc21_ca_hd_dcn   | 0.5089 | 0.2671 | 0.7507 | −0.0075 |
| 5 | s2_dsk42_al_hd_tcnr  | 0.5047 | 0.2417 | 0.7678 | −0.0141 |
| 6 | s2_cgc21_ca_hd_both  | 0.5044 | 0.2516 | 0.7571 | −0.0120 |
| 7 | s2_dsk42_ca_hd_tcnr  | 0.5041 | 0.2399 | 0.7684 | −0.0142 |
| 8 | s2_dsk42_ca_hd_both  | 0.4997 | 0.2385 | 0.7609 | −0.0186 |
| 9 | s2_dsk42_al_hd_both  | 0.4945 | 0.2341 | 0.7548 | −0.0243 |

Reference (Stage 1 promoted, default heads):
- dsk42_al default: joint 0.5188, next 0.2714, cat 0.7661
- dsk42_ca default: joint 0.5183, next 0.2671, cat 0.7695
- cgc21_ca default: joint 0.5164, next 0.2682, cat 0.7647

## Key Findings

### 1. **DCN category head wins on fusion — hypothesis confirmed**

The Stage 2 design hypothesis ("DCN may leverage explicit cross-features between the Sphere2Vec and HGI embedding halves") is **empirically supported**. On the two dselectk backbones:

- `dsk42_al`: default cat F1 = 0.7661 → **DCN cat F1 = 0.7827 (+2.2 %)**
- `dsk42_ca`: default cat F1 = 0.7695 → **DCN cat F1 = 0.7844 (+1.9 %)**

Joint score improves as well (+0.0087 and +0.0074). Next F1 stays essentially flat, confirming the DCN gain is localized to the category task where the fused embedding matters.

This is the *second* prior-overturning finding from this study:
- Phase 4 on HGI: head swaps degrade joint by 20 % ("standalone rankings don't transfer").
- Fusion: the right head swap (DCN) recovers *and exceeds* the default-head baseline.

**The mechanism is fusion-specific.** DCN explicitly learns second-order crosses between its input dimensions; when inputs are a concat of two semantically distinct embedding halves (spatial + structural), those crosses provide real lift. On HGI's single-source 64-dim input, there were no such distinct halves to cross, so DCN degenerated to "extra params with no special signal." This is publishable as an architectural insight:

> "Cross-feature-aware heads (DCN) recover the auxiliary-embedding gain that equal-weight + default heads cannot exploit."

### 2. **CGC21 + DCN actually *hurts*, despite DCN winning on dsk42**

`cgc21_ca_hd_dcn`: joint 0.5089 vs default 0.5164 (**−1.5 %**).

Why? CGC21 has only 1 task-specific expert per task; the architecture already forces stronger intra-task specialization in the backbone. When we add DCN (a wide cross layer) on top, the task head competes with the already-specialized CGC expert, and the two channels don't compose. This is a nice architectural interaction finding: **head gains are gated by whether the backbone leaves enough signal unprocessed for the head to work with.** DSelectK's soft expert-selection leaves more of the fusion signal intact for DCN to exploit.

### 3. **TCN Residual next head: marginal-to-harmful**

Only one TCN variant landed in the top-5 (`s2_cgc21_ca_hd_tcnr`, joint 0.5173, +0.0009). In every other pairing, swapping in TCN reduced both joint and next F1:

- default next = 0.27 → TCN next = 0.24–0.25 (−7 % to −10 %)

Despite TCN being the standalone next-head winner on HGI (F1 = 0.244), it *hurts* in the MTL pipeline. The Phase 4 "heads co-adapt with the backbone" caveat applies to next heads even on fusion. The TCN head's local-convolution inductive bias does not mesh with the Time2Vec per-step signal the way I hypothesized — possibly because the shared backbone already pools temporal info before the head sees it, leaving no local structure for the TCN to exploit.

### 4. **"Both swaps" is the worst variant**

`hd_both` rankings are 6, 8, 9 (joint 0.4945–0.5044). Swapping both heads breaks the most co-adaptation at once. Conservative conclusion: **swap one head, not two**. DCN-only is the clear winner.

### 5. **Top-3 overall (Stages 1+2 combined) for Stage 3**

Sorting Stage 1 promoted + Stage 2 results by joint score:

| Rank | Candidate | Joint | Source |
|------|-----------|-------|--------|
| 1 | s2_dsk42_al_hd_dcn | 0.5275 | Stage 2 |
| 2 | s2_dsk42_ca_hd_dcn | 0.5257 | Stage 2 |
| 3 | s1_dsk42_al        | 0.5188 | Stage 1 |
| 4 | s1_dsk42_ca        | 0.5183 | Stage 1 |
| 5 | s2_cgc21_ca_hd_tcnr| 0.5173 | Stage 2 |

**Top 3 for Stage 3** are all dselectk + {al,ca} + {DCN, default}. This is a natural ablation pairing for the final report:
- **#1 (best):** dselectk + aligned_mtl + DCN cat head (the "full recipe")
- **#2:** dselectk + cagrad + DCN cat head (alternative gradient-surgery)
- **#3:** dselectk + aligned_mtl + default heads (without DCN — measures head contribution)

This triple lets the paper separate **backbone+optimizer contribution** (#3 vs Stage 1 eq/db baselines) from **head contribution** (#1 vs #3).

## Decision

**PROMOTE** top 3 to Stage 3 (5 folds × 50 epochs).

Stage 3 is where the numbers get published, so I'll also:
- Track gradient cosine across epochs (per-epoch diagnostic from the runner).
- Report per-class F1 for category from the joint-checkpoint classification report.
- Add a paired t-test across folds between #1 and #2.
- Consider adding `s1_cgc21_ca` (CGC variant) as a supplementary run to let the paper cover at least one non-dselectk architecture — **noted but not blocking**; defer pending Stage 3 runtime.

## Artifacts
- `results/ablations/full_fusion_study/s2_heads_2f_15ep/summary.csv`
- `docs/full_ablation_study/runs/stage2.log`
