# STAN

## Source
- **Paper:** Luo, Liu, Liu. *STAN: Spatio-Temporal Attention Network for Next Location Recommendation.* The Web Conference (WWW) 2021. [arXiv:2102.04095](https://arxiv.org/abs/2102.04095).
- **Reference impl:** [`yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation`](https://github.com/yingtaoluo/Spatial-Temporal-Attention-Network-for-POI-Recommendation).
- **Architecture (paper):** bi-layer self-attention. Layer 1 (*trajectory aggregation*) is bare single-head attention over a 9-step trajectory with a pairwise spatio-temporal bias added directly to QK^T logits; bias is `Sum_d(E_t[Δt] + E_d[Δd])` interpolated from learned 1-D interval-embedding tables (Eq. 4–5). Layer 2 (*matching*) ranks candidate POIs via attention from candidate-POI embeddings to trajectory states with a candidate-side Δd bias (Eq. 8–9). Multi-modal input embedding `e_loc + e_user + e_time(hour-of-week)` (§4.1.1).

## ⚠ Faithfulness audit (2026-06-26, two-agent: paper §4 + reference `layers.py`/`load.py`/`train.py`)

**The v4 `faithful` numbers (AL 34.46 / AZ 38.96, below the Markov-1 region floor) are UNTRUSTWORTHY ARTIFACTS — superseded, do NOT cite.** They are produced by a STAN that is simultaneously **(i)** mis-implemented in the matching layer (collapses to a proximity prior), **(ii)** ~9× data-starved, and **(iii)** under-trained. The "below floor" measures the broken code, not STAN. Reference behavior confirmed verbatim from `Attn`/`SelfAttn` (`layers.py`), `load.py`, `train.py`.

| # | Component | Verdict | Reference (actual) | Ours (file:line) | Fix |
|---|---|---|---|---|---|
| 1 | **Matching layer** | **MAJOR — collapses to proximity prior** | `Attn`: **multiplicative** Δd gate (`torch.mul`) on the content dot-product, then a learned **`Linear(max_len,1)`** position collapse, **NO softmax** | additive `content/√d + bias_match`, then `softmax` over positions + reuse scores as values (`model.py:197-206`) | multiplicative gate + `nn.Linear(seq_len,1)` collapse; drop softmax |
| 2 | **Embedding init / scale** | **MAJOR** | `nn.Embedding` default ≈ N(0,1); content on same scale as bias | std **0.02** (`model.py:228,239`) + `/√d` ⇒ distance bias ~25× content at init → content channel born dead | init std ≈ 1/√d (≈0.088) / N(0,1)-like |
| 3 | **`/√d` scaling** | MINOR–MED | reference omits `/√d` in `SelfAttn` | divides by √d (`model.py:140,197`) | remove (or compensate via #2) |
| 4 | **Sequence construction** | **CRITICAL — 9× starvation + bias** | prefix-expansion: every position t supervised from its causal prefix → **L−1 targets/user**, max_len=100 (`load.py`, `train.py` expanding mask) | non-overlapping **stride-9**, one target/window → ~ceil(n/9)/user, targets only at {9,18,…} (`etl.py:54,179-220`) | prefix-expansion, CONTEXT_LEN≥50 (paper 100) |
| 5 | **Convergence** | **CRITICAL — under-trained** | constant LR (StepLR γ=1), 100 ep, early-stop on plateau, fp32 | 50 ep **OneCycleLR**→LR≈0 by ep50 ⇒ "best at 49/50" never plateaued (`train.py:122-124`) | constant/cosine-floor LR, 100–200 ep, early-stop patience ~15-20, best-ep interior |
| 6 | **Precision / seed** | DEVIATION | fp32; (board) seed 0 | **bf16** autocast + TF32 (`train.py:127-128,144,160,182-184`); default seed **42** | fp32 (`use_amp=False`, tf32 off); `--seed 0` |
| — | input emb, hour-of-week token, Δt/Δd, pad mask, region sjoin, CV split | **FAITHFUL** | — | — | keep |
| — | next-**region** target, CrossEntropy, single-head, dropped user-emb | DEVIATION (task-driven, defensible) | next-POI, BPR, multi-head, e_user | documented board adaptations | keep + note |

**Feasibility correction:** prefix-expansion is ~**8–9× rows** but **CA/TX are NOT infeasible** — AL/AZ trivial, FL minutes/fold, CA/TX ~3–4M instances fit GPU; the only bottleneck is the O(N_val·C) val-logit cache (`train.py:165,170`), which must be streamed. (The "large-state infeasible" lore was the unrelated MTL N×C path.) Scope per handoff stays **AL/AZ/FL** headline; CA/TX optional.

**Verdict:** fixes #1+#2 (matching layer + init) and #4+#5 (prefix-expansion + convergence) are the decisive four; #3/#6 are compliance. With them, faithful STAN should **plausibly clear** the Markov-1 region floor (modestly — coarse regions erode STAN's edge). **If audited-faithful + converged STAN STILL lands below the floor, THAT is the honest reportable result** (STAN is built for fine next-POI, not coarse regions).

## ✅ Converged faithful-STAN results (2026-06-26, seed 0, user-disjoint 5-fold)

Audited-faithful, converged, optimized (A+C+D+compile; see [`FAITHFUL_STAN_FINDINGS`](../../studies/closing_data/FAITHFUL_STAN_FINDINGS.md)).
**These supersede v4 — strike v4 from all artifacts.** STAN clears the region floor but lands **below our MTL reg** at every state (it ranks the coarse-region task it wasn't built for):

| State | regions | Acc@10 (5f) | our MTL reg | verdict | precision |
|---|---:|---:|---:|---|---|
| AL | 1109 | **60.72 ±5.20** | 69.81 | we beat | fp32+compile |
| AZ | 1547 | **49.86 ±11.52** | 59.34 | we beat | fp32+compile |
| Istanbul | 520 (mahalle) | **61.86 ±0.61** | 74.28 | we beat | bf16+compile |
| FL | 4703 | running (`v6_opt`) | 77.28 | (expect beat/near) | bf16+compile |
| CA / TX | 8501 / 6553 | pending (~1.5–2 h/state on A40, opt) | 65.66 / 67.02 | pending | bf16+compile |

## Why this is a baseline (not our model)
External published-method reference for the next-region task. We use it to:

1. Establish a *substrate-bound STAN* ceiling (`stl_check2hgi`, `stl_hgi`) — the same architecture consuming our pre-trained substrates, isolating the substrate's contribution.
2. Quantify the *substrate gap* via a faithful-from-scratch reproduction (`faithful`) — STAN with raw POI tokens + Δt + Δd, no in-house pre-training, target adapted to next-region.

The substrate-bound vs faithful gap quantifies how much our pre-trained Check2HGI / HGI buy on top of what STAN's published architecture can learn from raw inputs at our data scale.

## What's faithful, what's adapted

### Faithful to paper (`faithful` variant)
- Δt in **minutes**, Δd in **km via haversine** — matches reference `load.py:30-50`.
- Multi-modal input `e_loc + e_time` (hour-of-week 1..168) — matches paper §4.1.1 and reference `MultiEmbed`.
- Bare single-head self-attention with no FFN, no LayerNorm, no residual — matches reference `layers.SelfAttn`.
- Scalar pairwise bias added directly to QK^T logits (paper Eq. 5; we use 1-D scalar tables `[K]` since `Sum_d(E[K,D])` is functionally equivalent and avoids a `[B, n, R, D]` intermediate).
- Bi-layer architecture: trajectory-aggregation self-attention + matching layer ranking candidates with candidate-side Δd bias.
- Learned POI embedding from scratch (`nn.Embedding(n_pois+1, d_model)`), no pre-training.
- 50 epochs, max_lr 3e-3 — matches paper §5.4.

### Adapted because our task / data differ
- **Output is `n_regions` not `n_pois`.** Our table reports next-region; STAN's published task is next-POI. The matching layer ranks region candidates instead of POI candidates; region embedding `nn.Embedding(n_regions, d_model)` learned from scratch; candidate-side Δd bias derived from TIGER-tract centroids. Necessary task adaptation.
- **User embedding dropped.** STAN learns a per-user embedding (paper §4.1.1) and is evaluated under per-user temporal split (warm-user). We evaluate under cold-user `StratifiedGroupKFold` for table-comparability with the rest of our baselines, which makes user embeddings useless (random at val for held-out users). Dropped to avoid pretending we have signal we don't.
- **CrossEntropy loss instead of negative-sampled BPR.** Closed-set classification over ~1.5K regions makes the negative-sampling apparatus unnecessary.
- **Window=9, non-overlapping stride.** Paper uses `max_len=100` with prefix-expansion training. We match our in-house pipeline so cross-method comparisons are apples-to-apples.
- **AdamW + OneCycleLR(max_lr=3e-3) instead of vanilla Adam(lr=3e-3).** Our standard recipe for 5-fold CV; lr peak matches paper.
- **Substrate-bound variants (`stl_check2hgi`, `stl_hgi`).** Use our in-house `next_stan` head (`src/models/next/next_stan/`) which is STAN's bi-layer attention but with a *relative-position-only* pairwise bias instead of ΔT/ΔD interval embeddings — because our pre-trained substrates already absorb spatio-temporal context per check-in. These variants are **not** literature-faithful by themselves; they're the substrate-as-input version of STAN.

## Variants we run

| Variant | Inputs | Bias | Output | Where |
|---|---|---|---|---|
| `faithful` | raw POI tokens + lat/lon/datetime | ΔT minutes + Δd km, scalar interp | linear → `n_regions` via matching layer | `research/baselines/stan/` |
| `stl_check2hgi` | 9-step Check2HGI region embeddings | relative-position-only | linear → `n_regions` (last-position readout) | `scripts/p1_region_head_ablation.py --heads next_stan --region-emb-source check2hgi` |
| `stl_hgi` | 9-step HGI region embeddings | relative-position-only | linear → `n_regions` | `scripts/p1_region_head_ablation.py --heads next_stan --region-emb-source hgi` |

## Reproduction commands (current — A40, audited-faithful + optimized)

The faithful path is **self-contained** (`research/baselines/stan/`): ETL once per state, then train.
Recipe = audit fixes #1–#6 (prefix-expansion, multiplicative matching, constant-LR + early-stop, fp32
seed 0) + the validated optimization stack **A(bf16, big states)+C+D+compile** (quality bit-identical;
see [`FAITHFUL_STAN_FINDINGS §3`](../../studies/closing_data/FAITHFUL_STAN_FINDINGS.md)). Full run guide:
[`research/baselines/stan/README_FAITHFUL_STAN.md`](../../../research/baselines/stan/README_FAITHFUL_STAN.md).

```bash
# 1) ETL (prefix-expansion windows; CONTEXT_LEN=50). Run ETL ALONE — never alongside a train
#    (geopandas + prefix-expansion is RAM-heavy; co-running with training OOMs).
python -m research.baselines.stan.etl --state alabama

# 2) Train — small states (AL/AZ): fp32 + compile (bf16 buys launch-bound small states nothing)
python -u -m research.baselines.stan.train --state alabama --folds 5 \
    --epochs 200 --patience 20 --seed 0 --compile --tag v6_opt

# 2') Train — big states (FL/CA/TX): add --amp bf16 (halves the [B,n,R] memory traffic)
python -u -m research.baselines.stan.train --state florida --folds 5 \
    --epochs 200 --patience 20 --seed 0 --amp bf16 --compile --tag v6_opt
#    Istanbul (Massive-STEPS) needs its data pipeline first — see FAITHFUL_STAN_FINDINGS §4.
#    --only-fold k runs a single split (per-fold JSON) for cross-MACHINE fold fan-out (e.g. H100).
```

`--compile-mode {default,max-autotune,reduce-overhead}` exists but **default is the one to use**:
max-autotune segfaults, reduce-overhead gives no speedup. Larger batch (`--batch-size`) is a **reject**
(no wall-time gain + small quality drop). All per `FAITHFUL_STAN_FINDINGS §3.1`.

The substrate-bound `stl_check2hgi` / `stl_hgi` ablations are unchanged (`scripts/p1_region_head_ablation.py
--heads next_stan --region-emb-source {check2hgi,hgi}`); they are an OPTIONAL labelled ablation, NOT the
headline STAN cell.

## Source JSONs

**Headline `faithful` (converged, seed 0, supersedes v4):**

| State | JSON |
|---|---|
| AL | `docs/results/baselines/faithful_stan_alabama_5f_200ep_v5_compiled.json` |
| AZ | `docs/results/baselines/faithful_stan_arizona_5f_200ep_v5_compiled.json` |
| Istanbul | `docs/results/baselines/faithful_stan_istanbul_5f_200ep_v5_bf16c.json` |
| FL | `docs/results/baselines/faithful_stan_florida_5f_200ep_v6_opt.json` (running) |
| CA / TX | pending |

> ⚠ **The v4 JSONs (`*_5f_50ep_FAITHFUL_STAN_*_v4.json`) are the SUPERSEDED collapse artifacts — do NOT cite.**

**Substrate-bound ablations (`stl_check2hgi`, `stl_hgi`):** `docs/results/P1/region_head_{state}_region_5f_50ep_STAN_*.json`.

## Cross-references

- **Findings + data + optimization knowledge:** [`docs/studies/closing_data/FAITHFUL_STAN_FINDINGS.md`](../../studies/closing_data/FAITHFUL_STAN_FINDINGS.md) (the source of truth).
- **Run guide:** [`research/baselines/stan/README_FAITHFUL_STAN.md`](../../../research/baselines/stan/README_FAITHFUL_STAN.md).
- Region-baseline plan: [`articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/STAN_REFOOTING_HANDOFF.md).
