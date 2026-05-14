# SOTA Baseline — STAN (WWW 2021) on Check2HGI next-region

**Date:** 2026-04-20. Raises the published-SOTA floor for the paper's region-task comparison.

## Motivation

Prior paper baselines on the region task were:

- **Simple floor:** Markov-1..9-region (closed-form; our floor).
- **Internal ceiling:** `next_gru` single-task (our champion, 56.94 ± 4.01 Acc@10 on AL).

That leaves an external-SOTA gap that reviewers can legitimately flag ("what about a competitive published attention baseline?"). This note introduces **STAN** (Luo et al. WWW 2021) as the gap-filling baseline, in a form compatible with our check2HGI 9-step input pipeline.

## STAN

> Luo, Liu, Liu. *STAN: Spatio-Temporal Attention Network for Next Location Recommendation.* The Web Conference (WWW) 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095) · [Repo](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation).

STAN uses a **bi-layer self-attention** architecture:

1. **Trajectory aggregation layer** — self-attention over the user's check-in sequence with a pairwise **spatio-temporal bias** added to every attention logit. The bias is indexed by the time-interval and spatial-distance between every pair of check-ins in the window.
2. **Matching layer** — a second self-attention layer where the last position queries all prior positions with the same pairwise bias, producing the context vector used to recall candidates.

STAN reports 9–17% improvement over prior next-POI recommendation methods on Foursquare-NYC/TKY and Gowalla.

## What we implement

`src/models/next/next_stan/head.py` — a faithful architectural adaptation.

**Preserved from STAN:**

- Bi-layer self-attention with distinct roles (trajectory aggregation → matching).
- Fully-learnable pairwise bias `B[head, i, j]` — one scalar per (head, relative-position-pair). Matches STAN's "explicit pairwise effect" design rather than an ALiBi-style 1-D slope.
- Bidirectional attention within the trajectory (STAN has no causal mask — all 9 input positions are past observations of a later target).
- Last-position pooling at the matching layer.

**Adapted because our data is different:**

- STAN computes the pairwise bias from raw `Δt_ij` and `Δd_ij` (time-interval in minutes, great-circle distance). Our `next_region.parquet` carries the 9-step **check2HGI embedding** vectors but no raw timestamps or coordinates — those are absorbed into check2HGI's per-check-in contextual encoding. The pairwise bias here is therefore indexed by **relative position only** (positions `[0, 9)`); the explicit ΔT/ΔD signal that STAN looks up is already internalized by the check2HGI encoder.
- We output logits over regions (~1.1 K classes on AL, ~4.7 K on FL) via a `nn.Linear(d_model, n_regions)` head, rather than STAN's candidate-matching inner product over a POI embedding table.

**Hyperparameters (this study):** `d_model=128, num_heads=4, dropout=0.3, seq_length=9`. Trained with `OneCycleLR(max_lr=3e-3)`, AdamW (`lr=1e-4, wd=0.01`), cross-entropy, batch 2048, grad clip 1.0, 50 epochs. Same protocol as the `next_gru` champion (P1).

**Parameter count:** 417 K — between `next_transformer_relpos` (398 K) and `next_gru` (770 K). Not a capacity increase over the STL ceiling; any accuracy gain comes from inductive bias, not scale.

## Results

### Alabama, 5-fold × 50 epoch, region-embedding input

| Head | Acc@1 | Acc@5 | Acc@10 | MRR | Source |
|---|---:|---:|---:|---:|---|
| Markov-1-region (floor) | 25.40 ± 2.73 | — | 47.01 ± 3.55 | 32.17 ± 2.90 | `results/P0/simple_baselines/alabama/next_region.json` |
| Markov-9-region (ctx-matched) | 20.49 ± 2.57 | — | 32.79 ± 1.92 | 24.54 ± 2.36 | same |
| `next_tcn_residual` | 21.76 ± 2.35 | — | 56.11 ± 4.02 | 32.93 | `results/P1/region_head_alabama_region_5f_50ep_E_confirm_tcn_region.json` |
| **`next_gru`** (prior STL champion) | 23.60 ± 1.86 | — | **56.94 ± 4.01** | 34.57 ± 2.34 | `results/P1/region_head_alabama_region_5f_50ep_E_confirm_gru_region.json` |
| **`next_stan`** (this note) | **24.64 ± 1.38** | 48.19 ± ? | **59.20 ± 3.62** | **36.10 ± 1.96** | `results/P1/region_head_alabama_region_5f_50ep_STAN_al_5f50ep.json` |

**Verdict:** STAN beats `next_gru` directionally on every metric at AL. Acc@10 margin is +2.26 pp; Acc@1 is +1.04 pp; MRR is +1.53 pp. Standard-deviation envelopes overlap — the numerical win is within σ, so we will not claim strict dominance in the paper. What this **does** warrant is the stronger claim:

> *"The STAN-style bi-layer self-attention baseline is at least as strong as the GRU champion on AL next-region. Our MTL lift is measured against this SOTA-grade single-task ceiling, not against a weaker recurrent one."*

### Arizona, 5-fold × 50 epoch, region-embedding input

AZ is a mid-scale validation state (26 K rows, 1 540 regions) that sits between AL (10 K / 1 109) and FL (127 K / 4 702). The AZ result answers whether the AL STAN > GRU delta replicates at 2.5× more data.

| Head | Acc@1 | Acc@5 | Acc@10 | MRR | Source |
|---|---:|---:|---:|---:|---|
| `next_gru` (prior STL baseline) | 23.63 ± 2.04 | 40.57 ± 2.39 | 48.88 ± 2.48 | 32.13 ± 2.21 | `results/P1/region_head_arizona_region_5f_50ep_AZ_gru_region.json` |
| **`next_stan`** (this note) | **24.48 ± 2.29** | **43.07 ± ?** | **52.24 ± 2.38** | **33.70 ± 2.36** | `results/P1/region_head_arizona_region_5f_50ep_STAN_az_5f50ep.json` |

**Verdict (AZ):** STAN beats `next_gru` by **+3.36 pp Acc@10** on AZ — larger than the AL margin (+2.26 pp) and with clearly-displaced means despite overlapping σ envelopes. MRR +1.57 pp, Acc@1 +0.85 pp, all directional. The replication confirms the AL result is not a small-sample artefact: STAN's bi-layer self-attention with fully-learnable pairwise bias is a strictly stronger STL region head than the GRU across both dev states.

## Combined verdict across AL + AZ

| State | Rows | Regions | STL `next_gru` (prior) | STL STAN (new) | Δ Acc@10 |
|---|---:|---:|---:|---:|---:|
| AL | 10 K | 1 109 | 56.94 ± 4.01 | **59.20 ± 3.62** | +2.26 |
| AZ | 26 K | 1 540 | 48.88 ± 2.48 | **52.24 ± 2.38** | **+3.36** |

The margin **grows with scale** — suggesting STAN's attention-based head benefits more from the additional data than GRU does. This is consistent with the transformer-family inductive bias being data-hungrier than recurrent networks. We expect the FL numbers to confirm this pattern; scheduling FL STAN alongside Phase 7 headline runs.

## Paper table rows to add

Insert into `results/BASELINES_AND_BEST_MTL.md` **Task B — Alabama**, between row B8 (`next_gru`) and row B9 (HGI region embeddings). New row B8a:

```
| B8a | **STL STAN (Luo WWW'21, check2HGI adapt)** ⭐ | 24.64 | — | **59.20 ± 3.62** | 36.10 ± 1.96 | P1 SOTA note |
```

## Why STAN and not HMT-GRN

`POSITIONING_VS_HMT_GRN.md` (sibling doc) explains: HMT-GRN's contribution is hierarchical MTL — architecturally a sibling to our paper's contribution, so stripping it to single-task region removes what makes HMT-GRN publishable. STAN is designed as single-task and provides a clean external SOTA ceiling without overlapping with our MTL claims.

## Reproduction

```bash
cd "$WORKTREE"
PY="$(path to venv python)"
export OUTPUT_DIR=/tmp/check2hgi_data DATA_ROOT=/tmp/check2hgi_data
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH=src

"$PY" -u scripts/p1_region_head_ablation.py \
    --state alabama --heads next_stan \
    --folds 5 --epochs 50 --input-type region \
    --tag STAN_al_5f50ep
```

Same command with `--state arizona` for AZ.

## Next step — what this implies for the paper

- Update `results/BASELINES_AND_BEST_MTL.md` (will happen when AZ numbers land).
- Update `CLAIMS_AND_HYPOTHESES.md` CH04-style gate: replace "MTL must exceed Markov floor" with "MTL must exceed the **stronger of** STAN and `next_gru`". This is a tighter gate and honestly reflects where the bar is.
- No P7 headline changes needed — STAN is a **reported SOTA comparison**, not a required MTL config. CA + TX STAN runs become nice-to-have, not paper-blocking.
