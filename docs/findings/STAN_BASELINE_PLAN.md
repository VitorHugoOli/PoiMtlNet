# STAN baseline experiments — plan

**Date:** 2026-04-25. **Scope:** AL + AZ only (FL / CA / TX deferred per user). **Owner:** check2HGI study.

## 0 · Why this plan

The existing "STL STAN" rows in our paper (`SOTA_STAN_BASELINE.md`) are **STAN-architecture runs on our check2HGI substrate**, not literature-baseline reproductions. Two independent reasons:

1. **Substrate.** Input is the 9-step check2HGI region-embedding sequence; STAN's published pipeline ingests raw POI tokens + timestamps + coordinates.
2. **Pairwise bias.** STAN's signature contribution is a `B[head, i, j]` bias indexed by interpolated `Δt_ij` (minutes) and `Δd_ij` (km). Our `next_stan` head **strips that** and uses relative-position-only bias (`SOTA_STAN_BASELINE.md:38`, `head.py:18-32`) on the grounds that check2HGI already absorbs ΔT/ΔD signal.

So the published-method ceiling is missing from the paper. Two new experiment groups close it.

---

## 1 · What exists today (coverage map)

| Cell | Substrate | Head config | AL | AZ | Source JSON |
|---|---|---|:-:|:-:|---|
| STL STAN on Check2HGI | check2HGI region emb | `next_stan` Gaussian d=128 | ✅ | ✅ | `results/P1/region_head_{state}_region_5f_50ep_STAN_*.json` |
| STL STAN ALiBi init | check2HGI region emb | `next_stan` ALiBi d=128 | ✅ | — | `results/P1/..._STAN_alibi_al_5f50ep.json` |
| MTL+STAN d=128 | check2HGI region emb | `next_stan` Gaussian | ✅ | ✅ | `results/P8_sota/mtl_crossattn_pcgrad_{state}_stan_5f50ep.json` |
| MTL+STAN d=256 | check2HGI region emb | `next_stan` Gaussian | ✅ | ✅ | `results/P8_sota/..._stan_d256_*.json` |
| MTL+STAN d=256 ALiBi | check2HGI region emb | `next_stan` ALiBi | ✅ | ✅ | `results/P8_sota/..._stan_d256_alibi_*.json` |
| **Faithful STAN** | **raw POI tokens + Δt + Δd** | **`next_stan_faithful`** (new) | ❌ | ❌ | — |
| **STL STAN on HGI** | **HGI region emb** | `next_stan` Gaussian d=128 | ❌ | ❌ | — |
| MTL+STAN on HGI (optional) | HGI region emb | `next_stan` Gaussian | ❌ | ❌ | — |

Two new STL groups close the gap. MTL-on-HGI is a stretch goal contingent on STL-on-HGI lifting (or sinking) the substrate.

---

## 2 · Experiment A — Faithful STAN baseline

**Goal:** publish-comparable external SOTA ceiling. STAN's bi-layer attention with its native `Δt`/`Δd` pairwise bias and a learnable POI-token embedding, on Gowalla state-level data, predicting **next region**.

### 2.1 Why next-region (not next-POI)

STAN's published task is next-POI on ~10K candidates. Our paper's task is next-region. To be **comparable to our table**, the baseline must predict the same target. Adapting STAN's classifier head to project to `n_regions` keeps STAN's input pipeline + bi-layer attention intact while making the metric apples-to-apples. This is a **task-adaptation**, not an architecture swap; it's the honest framing for the paper.

(A pure STAN reproduction predicting next-POI is out of scope — STAN's own published numbers are on different splits and aren't comparable to anything in our table.)

### 2.2 What to build

**A1. Input pipeline** (~150 LOC, `src/data/inputs/next_region_faithful.py`)

- Source: `data/checkins/<State>.parquet` (already has `userid`, `placeid`, `datetime`, `latitude`, `longitude`, plus the region map via the existing `data.inputs.next_region` helper).
- Per user: sort by datetime, slide 9-step windows + 1 target, drop windows where target lacks a region label.
- Per window emit: `placeid_0..8` (int), `lat_0..8` (float), `lon_0..8` (float), `t_minutes_0..8` (int — minutes since first checkin in window), `target_region_idx` (int), `userid` (for fold grouping).
- StratifiedGroupKFold(5) on `userid`, seed 42, identical to current pipeline.

**A2. Head** (~250 LOC, `src/models/next/next_stan_faithful/head.py`)

- Learnable POI embedding table `nn.Embedding(n_pois, d_model)`. Dimensions: AL ~5 K POIs, AZ ~9 K. Memory negligible.
- Pairwise bias: STAN's interval-embedding tables with linear interpolation. Two learned 1-D embedding tables `ET[K_t, d_model]` and `ED[K_d, d_model]` (K_t, K_d ≈ 64). For each (i, j) pair, compute `Δt_ij` (clipped to [0, T_max]) and `Δd_ij` (haversine, clipped to [0, D_max]); look up + linearly interpolate; project to a per-head bias scalar.
- Bi-layer self-attention as in current `next_stan` (preserved structurally).
- Final classifier: `nn.Linear(d_model, n_regions)` projecting from last position's hidden state.

**A3. Training script** (~50 LOC, `scripts/faithful_stan_baseline.py`)

Single-task only. Wraps the existing fold/CE/OneCycleLR loop with the new dataset + head. Mirrors `scripts/p1_region_head_ablation.py` API: `--state {alabama,arizona} --folds 5 --epochs 50 --tag FAITHFUL_STAN_<state>_5f50ep`.

### 2.3 Hyperparameters

Same as current `next_stan`: `d_model=128, num_heads=4, dropout=0.3, seq_length=9, OneCycleLR(max_lr=3e-3), AdamW(lr=1e-4, wd=0.01), batch=2048, 50 ep`. Interval-embedding sizes `K_t=K_d=64`. Clip `T_max=10080` (1 week, minutes), `D_max=200` (km — sanity ceiling on intra-state distances). Total params ≈ 1.5 M (POI table dominates).

### 2.4 Expected outputs

- `results/P1/region_head_alabama_region_5f_50ep_FAITHFUL_STAN.json`
- `results/P1/region_head_arizona_region_5f_50ep_FAITHFUL_STAN.json`
- New research note: `research/FAITHFUL_STAN_FINDINGS.md` reporting Acc@10 / Acc@1 / MRR vs:
  - STL STAN on Check2HGI (substrate-equivalent published-architecture ceiling)
  - STL `next_gru` on Check2HGI (recurrent ceiling)
  - Markov-1-region (classical floor)

### 2.5 What we'll learn

| Outcome | Interpretation |
|---|---|
| Faithful STAN ≥ STL STAN on Check2HGI | The published architecture with native ΔT/ΔD inputs reaches our ceiling — paper claim "we beat literature SOTA" is honest |
| Faithful STAN < STL STAN on Check2HGI | Check2HGI substrate is doing real work — this is the paper's substrate-contribution claim made quantitative |
| Faithful STAN < Markov-1-region | STAN's learned attention isn't extracting signal beyond the 1-gram prior at our state-level scale (would be a finding about scale, not about STAN) |

### 2.6 Costs

- Implementation: ~1–2 days (new pipeline + head + script + sanity test).
- Compute: ~30 min × 2 states on M4 Pro (5f × 50 ep, batch 2048, 9-window).
- Risk: medium — first time running a from-scratch POI-embedding learner in this study; debugging the pairwise interval-embedding interpolation is the likely time sink.

---

## 3 · Experiment B — STAN on HGI substrate

**Goal:** substrate ablation for the region task. Mirrors CH16 (Check2HGI > HGI on cat F1) but for the region head. Tests "does check2HGI's per-checkin contextual encoding actually help the region task vs HGI's POI-level encoding mapped to region embeddings?"

### 3.1 What's already there

- `output/hgi/alabama/region_embeddings.parquet` exists. So does `arizona`. The substrate is generated.
- `data/inputs/next_region.py:171-177` currently hard-codes `EmbeddingEngine.CHECK2HGI` and raises NotImplementedError otherwise.
- `IoPaths.get_next_region(state, engine)` already routes by engine; the path is reserved.

### 3.2 What to build

**B1. Extend `build_next_region_frame(state, engine)` to accept HGI** (~40 LOC, `src/data/inputs/next_region.py`)

- For HGI, the 9-window source is the same `sequences_next.parquet` from Check2HGI preprocessing (we keep the user/temporal grouping consistent across substrates).
- For each check-in `placeid_i` in the window, look up `region_id_i` from the existing placeid→region map, then look up the HGI region embedding from `output/hgi/<state>/region_embeddings.parquet[region_id_i]`.
- Emit the same parquet schema as the check2HGI version: `[emb_0..emb_d, last_region_idx, target_region, userid]`. Embedding dimension may differ (HGI is 256-d per spec; verify).
- Drop the engine guard in `load_next_region_data`; route via `IoPaths.get_next_region(state, engine)`.

**B2. CLI plumbing** (~10 LOC)

`scripts/p1_region_head_ablation.py` already has `--engine` plumbing through `EmbeddingEngine`. Confirm it propagates into the FoldCreator and into `last_region_idx` derivation. Likely no code change needed.

**B3. Generate inputs**

```bash
python scripts/regenerate_next_region.py --state alabama --engine hgi
python scripts/regenerate_next_region.py --state arizona --engine hgi
```

(extend the existing script to take `--engine`, or duplicate.)

### 3.3 Runs

```bash
# STL STAN on HGI substrate, AL + AZ
python scripts/p1_region_head_ablation.py \
    --state alabama --engine hgi --heads next_stan \
    --folds 5 --epochs 50 --input-type region \
    --tag STAN_HGI_al_5f50ep

python scripts/p1_region_head_ablation.py \
    --state arizona --engine hgi --heads next_stan \
    --folds 5 --epochs 50 --input-type region \
    --tag STAN_HGI_az_5f50ep
```

Optional second pass: `next_gru` on HGI (so the substrate ablation isn't head-confounded — same head, different substrate, head-by-head). ~25 min extra per state.

### 3.4 Expected outputs

- `results/P1/region_head_alabama_region_5f_50ep_STAN_HGI.json`
- `results/P1/region_head_arizona_region_5f_50ep_STAN_HGI.json`
- (Optional) `..._GRU_HGI.json` for matched-head substrate comparison.
- New research note: `research/STAN_HGI_FINDINGS.md` extending CH16 to the region task.

### 3.5 What we'll learn

| Outcome | Interpretation |
|---|---|
| STAN on Check2HGI > STAN on HGI (both states, σ-clean) | CH16 generalises from cat to region — check2HGI's per-checkin contextual encoding is substrate-superior on both heads |
| STAN on HGI ≈ STAN on Check2HGI | Substrate doesn't matter for the region head (which would weaken CH16's generalisation) |
| STAN on HGI > STAN on Check2HGI | Surprising; would require investigation (likely a regime where HGI's stronger POI-level signal beats check2HGI's contextual aggregation for short-range region prediction) |

### 3.6 Costs

- Implementation: ~2–4 hours (small extension to existing pipeline; the bulk is extending `build_next_region_frame` and verifying the embedding-dim stitch).
- Compute: ~25 min × 2 states × 1 head; +25 min × 2 states if `next_gru` matched-head pass added.
- Risk: low — reuses 95 % of existing infrastructure.

---

## 4 · Sequence of execution

Cost asymmetry is large. Recommend:

1. **Experiment B first** (HGI substrate). Cheap, reuses infra, answers the substrate-generalisation question for CH16.
2. **Experiment A second** (Faithful STAN). New pipeline + head; biggest implementation cost; gives the external-baseline ceiling for the paper's table.

If A turns out impractical (e.g., POI embedding fails to converge from scratch on 10K-row AL), the fallback is **honesty in writing**: explicitly label the existing `STL STAN` rows as "STAN architecture on check2HGI substrate" and not as a literature reproduction. The current paper text already concedes part of this in CH10; the wording would be tightened.

---

## 5 · Open questions to confirm before launch

1. **Faithful-STAN POI vocab.** Do we restrict the POI vocab to placeids appearing in the state's check-ins (~5K AL, ~9K AZ), or include all Gowalla POIs the user may visit? Recommend: state-restricted, matches current substrate scope.
2. **Faithful-STAN target.** Confirm we predict next-region (consistent with our table) and not next-POI (consistent with STAN's published task). Plan above assumes next-region.
3. **HGI region-embedding dim.** Verify dimension (likely 256 vs check2HGI's 64) and ensure the STL STAN head's `d_model=128` projection accommodates either (it should — STAN's input projection is `nn.Linear(emb_dim, d_model)`).
4. **Pairwise interval table size.** STAN's published `K_t, K_d` are ~50 each. Use 64 as a slightly larger default; verify no regression vs paper's published numbers.
5. **Optional MTL+STAN on HGI.** Stretch goal — only add if STL-on-HGI changes the picture. Skip otherwise.

Resolve 1–4 before opening the implementation PR; question 5 deferred to after Experiment B results land.

---

## 6 · Tracker entries to add (FOLLOWUPS_TRACKER.md)

- **F36** (P1) — STAN on HGI substrate, AL + AZ STL 5f × 50 ep. Owner: TBD. Cost: ~4 h dev + 1 h compute.
- **F37** (P2) — Faithful STAN baseline (raw ΔT/ΔD), AL + AZ STL 5f × 50 ep. Owner: TBD. Cost: ~2 days dev + 1 h compute.
- **F38** (P3, conditional on F36/F37 outcomes) — MTL+STAN on HGI substrate, AL + AZ. Cost: ~30 min compute.

FL / CA / TX equivalents (F36-FL, F37-FL, etc.) deferred per user.
