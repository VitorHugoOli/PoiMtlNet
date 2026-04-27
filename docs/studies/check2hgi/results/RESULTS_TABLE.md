# Results Table — Check2HGI MTL Study

**Last updated:** 2026-04-26 (F48-H3-alt per-head LR + F40 + F48-H2 + F31 post-F27 B3 + F21c STL GETNext-hard rows landed).

**Champion candidate (2026-04-26):** **F48-H3-alt** = B3 architecture + per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3`, `--scheduler constant`). Closes the F21c STL gap (CH18 Tier B → A): AL exceeds STL by +6.25 pp; AZ closes 75%; FL validates at 5-fold scale (cat preserved, reg +6.7 pp over predecessor B3). See [`../NORTH_STAR.md`](../NORTH_STAR.md), [`../MTL_ARCHITECTURE_JOURNEY.md`](../MTL_ARCHITECTURE_JOURNEY.md) for derivation, [`../research/F48_H3_PER_HEAD_LR_FINDINGS.md`](../research/F48_H3_PER_HEAD_LR_FINDINGS.md) for detail.

**Predecessor (2026-04-24, kept as comparand):** **B3** = `mtlnet_crossattn + static_weight(cat=0.75) + next_gru (cat) + next_getnext_hard (reg)`, OneCycleLR max=0.003, 50ep.

**Common protocol** (unless noted per row): user-disjoint `StratifiedGroupKFold(groups=userid)`, 5 folds × 50 epochs, seed 42, AdamW (lr=1e-4 → OneCycleLR max_lr=0.003 for predecessor B3 / per-head constant for H3-alt), batch 2048 (1024 for FL H3-alt to avoid MPS OOM), `gradient_accumulation_steps=1`. Reg metrics are `*_indist` (restricted to regions seen in training set of the fold) where applicable.

**Legend:**
- ✅ best-in-table for that metric · ⭐ champion config · 🔴 pending · 🟡 known limitation
- `(n=1)` = single-fold; all others are 5-fold mean ± std unless noted
- `(SUPERSEDED)` = value replaced by a later run
- **Cat F1** = macro-F1 on 7 categories. **Reg Acc@10** = `top10_acc_indist`. **Reg Acc@5** = `top5_acc_indist`. **Reg MRR** = `mrr_indist`. **Reg F1** = macro-F1 on ~1K–4.7K regions.

---

## 1 · Ablation states (AL + AZ + FL-1f) — mechanism + validation

These three blocks are the study's ablation evidence (`PAPER_STRUCTURE.md §2.1`). They are not the paper's headline numbers but justify the method's behavior.

### 1.1 Alabama (AL, 10 K rows, 1,109 regions, 5f × 50ep unless noted)

**Baselines — next-category**

| Method | cat F1 | cat Acc@1 | Source |
|---|---:|---:|---|
| Random (1/7) | ≈14.3 | ≈14.3 | theoretical |
| Majority | — | 34.20 | P0 |
| Markov-1-POI | ≈31.7 | ≈32 | P0 |
| POI-RGNN (state-level range reference) | 31.8–34.5 | — | `docs/baselines/BASELINE.md` (Capanema et al.) |

**Baselines — next-region**

| Method | Acc@1 | **Acc@10** | Acc@5 | MRR | F1 | Source |
|---|---:|---:|---:|---:|---:|---|
| Random | 0.09 | 0.90 | — | — | — | theoretical |
| Majority | 1.97 | 1.97 | — | — | — | P0 |
| Top-K popular | 1.97 | 14.67 | — | — | — | P0 |
| **Markov-1-region** (simple floor) | 25.40 ± 2.73 | **47.01 ± 3.55** | — | 32.17 ± 2.90 | — | P0 |
| Markov-5-region w/ backoff | 20.80 ± 2.65 | 33.42 ± 2.16 | — | 24.99 ± 2.53 | — | P0 |
| Markov-9-region (ctx-matched) | 20.49 ± 2.57 | 32.79 ± 1.92 | — | 24.54 ± 2.36 | — | P0 |

**STL — matched-class & literature-aligned**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat (`next_mtl`) | **38.58 ± 1.23** | — | — | — | — | — | P1_5b refair |
| STL HGI cat (substrate ablation, CH16) | 20.29 ± 1.34 | — | — | — | — | — | P1_5b refair |
| STL GRU reg | — | 23.60 ± 1.86 | 56.94 ± 4.01 | — | 34.57 ± 2.34 | — | P1 |
| STL TCN-residual reg | — | 21.76 ± 2.35 | 56.11 ± 4.02 | — | 32.93 | — | P1 |
| **STL STAN reg** (Luo WWW'21 adapt) | — | 24.64 ± 1.38 | **59.20 ± 3.62** | — | 36.10 ± 1.96 | 24.64 ± 1.38 | P1 SOTA |
| STL HGI reg (substrate ablation) | — | ? | 57.02 ± 2.92 | — | 33.14 ± 1.87 | — | P1.5 (tied with Check2HGI reg) |
| **STL GETNext-hard reg (matched-head)** — **F21c** ⭐ | — | 24.07 ± 1.94 | **68.37 ± 2.66** | 53.62 ± 3.02 | 41.17 ± 2.28 | 11.91 ± 0.86 | **F21c 2026-04-24 — STL ceiling for H3-alt comparison** |

**MTL — all variants**

| # | Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B-M1 | dselectk + pcgrad + GRU | 36.08 ± 1.96 | 13.31 | 48.88 ± 6.26 | — | 24.43 | — | P2 prior champion |
| B-M2 | ~~dselectk + MTLoRA r=8 + GRU~~ | — | — | ~~50.72 ± 4.36~~ | — | — | — | SUPERSEDED (MTL_PARAM_PARTITION_BUG) |
| B-M2b | dselectk + MTLoRA r=8 + GRU (post-fix) | 36.53 ± 1.24 | 17.48 ± 1.35 | 53.71 ± 3.80 | 40.54 ± 3.17 | 29.60 ± 2.01 | 8.31 ± 1.02 | `results/P5_bugfix/a7_mtlora_r8_al_5f50ep_postfix_pcgrad.json` |
| B-M3 | cross-attn + static λ=0.5 + GRU | ? | 11.34 | 50.26 ± 4.34 | — | 25.35 | — | P2 ablation step 2 |
| B-M4 | cross-attn + pcgrad + GRU | 38.58 ± 0.98 | 10.06 | 45.09 ± 5.37 | — | 20.94 | — | P2 ablation step 6 |
| B-M5 | cross-attn + pcgrad + STAN d=128 | 39.07 ± 1.18 | 12.48 ± 1.44 | 50.27 ± 4.47 | — | 24.16 ± 2.25 | — | P8 MTL-STAN |
| B-M6 | cross-attn + pcgrad + STAN d=256 | 38.11 ± 1.11 | 13.86 ± 3.43 | 51.60 ± 10.09 | — | 25.69 ± 5.34 | — | P8 high-σ |
| B-M6a | + ALiBi on STAN d=256 | ? | 14.09 ± 3.71 | 51.64 ± 8.92 | — | 25.69 ± 5.40 | — | P8 ALiBi null |
| B-M6b | **cross-attn + pcgrad + GETNext-SOFT d=256** | 38.56 ± 1.45 | 15.72 ± 2.74 | 56.49 ± 4.25 | 43.40 ± 4.60 | 28.93 ± 3.20 | 8.66 ± 1.20 | P8 soft champion (prior north-star) |
| B-M6c | + static (cat=0.50) | 8.65 ± 0.56 (F1_macro) | 15.60 ± 2.06 | 56.45 ± 4.34 | 43.08 ± 4.42 | 28.88 ± 2.55 | 8.52 ± 0.64 | attribution test: pcgrad ≈ static on soft |
| B-M6d | + ALiBi on soft | 9.05 ± 1.02 | 16.37 ± 2.13 | 57.27 ± 4.17 | 43.87 ± 3.84 | 29.52 ± 2.63 | 9.05 ± 1.02 | B7 AL — optional stabiliser |
| B-M6e | cross-attn + pcgrad + GETNext-HARD d=256 | 38.50 ± 1.56 | 15.03 ± 3.04 | 57.96 ± 5.09 | 44.22 ± 5.58 | 28.93 ± 3.88 | 9.47 ± 0.71 | B5 AL hard — +1.47 over soft (tied within σ) |
| B-M_B3 (pre-F27) | cross-attn + static (cat=0.75) + GETNext-HARD d=256 + NextHeadMTL cat | 39.28 ± 0.80 | ? | 56.33 ± 8.16 | 42.81 ± 7.89 | 28.55 ± 5.33 | 9.43 ± 0.71 | B3 validation AL (F18, 2026-04-23). Superseded by post-F27 row below. |
| **B3 (post-F27)** | cross-attn + static(0.75) + **next_gru cat** + GETNext-HARD reg (OneCycleLR 50ep) | **42.71 ± 1.37** | ? | 59.60 ± 4.09 | ? | 28.55 ± 5.33 | ? | **B3 post-cat-head-swap (F31, 2026-04-24)** — predecessor champion. +3.43 cat F1 over pre-F27 row. |
| **MTL-H3-alt** | ⭐ B3 + **per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, constant)** | **42.22 ± 1.00** | 34.93 ± 3.26 | ✅ **74.62 ± 3.11** | 61.65 ± 4.06 | 47.49 ± 3.29 | ? | **F48-H3-alt 2026-04-26** — **MTL EXCEEDS STL F21c ceiling by +6.25 pp on reg** while preserving cat F1 within σ of B3. CH18 Tier B → A. |
| — | F40 scheduled-static (cat_weight 0.75 → 0.25 ramp, OneCycleLR) | 42.63 ± 1.26 | 16.61 ± 2.20 | 60.81 ± 3.10 | 47.33 ± 3.58 | 30.85 ± 2.64 | ? | **negative control (2026-04-26)** — cat preserved, reg only +1.21 pp over B3 (Pareto fails). Refutes loss-side mechanism. |
| — | F48-H1 single LR const 1e-3, 150ep | 40.99 ± 1.80 | ? | 61.43 ± 9.60 | ? | ? | ? | gentle const refuted — reg-best ep collapses [4..10], σ inflated. |
| — | F48-H2 warmup_constant (50ep ramp + 100ep plateau @ 3e-3, single LR) | 41.35 ± 0.78 | 17.35 ± 1.58 | 57.84 ± 4.48 | 44.32 ± 3.26 | 30.17 ± 1.69 | ? | **negative control (2026-04-26)** — cat survives warmup but reg DROPS below B3 (-1.76 pp). Cat-vs-reg compete for shared cross-attn at plateau LR. |
| — | F45 single LR const 3e-3, 150ep | **10.44 ± 0.04 💀** | ? | **74.20 ± 2.95** | ? | ? | ? | breakthrough — proves reg arch CAN exceed STL but cat collapses. Mechanism source for H3-alt. |
| — | F48-H3 per-head (sh=3e-3, instead of 1e-3) | 11.53 ± 1.63 💀 | 34.93 ± 3.26 | 74.24 ± 2.58 | 61.64 ± 4.01 | 47.41 ± 3.33 | ? | reproduces F45 — confirms shared cross-attn at 3e-3 destabilises cat path. Refutes "throttle cat encoder alone" hypothesis. |
| — | MTLoRA rank sweep r=16 / r=32 (post-fix) | — | 15.83 / 17.01 | 51.62 / 53.28 | — | 27.78 / 29.24 | — | rank-insensitive — `P5_bugfix` |
| — | AdaShare (post-fix) | — | 10.66 ± 3.76 | 44.51 ± 6.87 | — | 21.62 ± 4.68 | — | below Markov — not competitive |

### 1.2 Arizona (AZ, 26 K rows, 1,547 regions, 5f × 50ep unless noted)

**Baselines**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR |
|---|---:|---:|---:|---:|---:|
| Majority | — | 7.43 | 7.43 ± 0.70 | — | — |
| Top-K popular | — | 7.43 | 20.82 ± 1.28 | — | — |
| **Markov-1-region** | — | 23.98 ± 1.13 | **42.96 ± 2.05** | — | — |
| Markov-9-region (ctx-matched) | — | — | 33.38 ± 1.33 | — | — |

**STL**

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat (`next_mtl`) | **42.08 ± 0.89** | — | — | — | — | — | P1_5b |
| STL HGI cat (CH16 extension) | 🔴 F3 pending | — | — | — | — | — | — |
| STL GRU reg | — | 23.63 ± 2.04 | 48.88 ± 2.48 | — | 32.13 ± 2.21 | — | P1 |
| **STL STAN reg** | — | 24.48 ± 2.29 | 52.24 ± 2.38 | 40.41 ± ? | 33.70 ± 2.36 | 24.48 ± 2.29 | P1 SOTA |
| **STL GETNext-hard reg (matched-head)** — **F21c** ⭐ | — | 25.13 ± 2.07 | **66.74 ± 2.11** | 52.18 ± 2.20 | 41.15 ± 2.13 | 12.28 ± 0.91 | **F21c 2026-04-24 — STL ceiling for H3-alt comparison** |

**MTL**

| # | Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | reg F1 | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| B-M7 | cross-attn + pcgrad + GRU | 43.13 ± 0.55 | 13.20 ± 1.99 | 41.07 ± 3.46 | — | 22.49 ± 2.49 | 13.20 ± 1.99 | P2 az1 |
| B-M8 | + STAN d=128 | 42.64 ± 0.26 | 9.79 ± 1.98 | 37.47 ± 4.01 | — | 18.53 ± 2.54 | — | MTL-STAN d=128 bottleneck |
| B-M9 | + STAN d=256 | 42.74 ± 0.45 | 11.53 ± 2.11 | 41.04 ± 4.55 | — | 20.93 ± 2.86 | — | P8 hp-tuned |
| B-M9a | + STAN d=256 + ALiBi | 42.74 ± 0.45 | 11.24 ± 1.41 | 41.04 ± 3.26 | — | 20.79 ± 2.03 | — | null on mean, −28% σ |
| B-M9b | **soft GETNext d=256** | 42.82 ± 0.96 | 12.63 ± 1.79 | 46.66 ± 3.62 | 35.70 ± 3.38 | 23.81 ± 2.30 | 6.93 ± 0.68 | P8 AZ soft |
| B-M9c | + static (cat=0.50) | 7.30 ± 0.46 (F1_macro) | 12.79 ± 1.98 | 47.32 ± 3.02 | 36.16 ± 3.17 | 24.16 ± 2.27 | 7.13 ± 0.54 | attribution: pcgrad ≈ static on soft |
| B-M9d | cross-attn + pcgrad + GETNext-HARD | 42.22 ± 0.53 | 14.55 ± 2.53 | 53.25 ± 3.44 | 40.06 ± 3.36 | 26.89 ± 2.62 | 8.95 ± 0.52 | B5 AZ hard — **+6.59 pp Acc@10 over soft**, all 4 region metrics p=0.0312 (F1 Wilcoxon) |
| B-M_B3 (pre-F27) | cross-attn + static(0.75) + GETNext-HARD + NextHeadMTL cat | 43.62 ± 0.74 | 14.27 ± 2.53 | 52.76 ± 3.92 | 39.68 ± 3.61 | 26.40 ± 2.45 | 9.17 | B3 validation AZ (F19, 2026-04-23). Wilcoxon p=0.0312 on cat F1 vs STL. Superseded by post-F27 row. |
| **B3 (post-F27)** | cross-attn + static(0.75) + **next_gru cat** + GETNext-HARD reg (OneCycleLR 50ep) | **45.81 ± 1.30** | 49.30 ± 0.67 | 53.82 ± 3.11 | 40.54 ± 3.40 | 27.66 ± 2.41 | ? | **F31 post-cat-head-swap (2026-04-24)** — predecessor champion. +2.19 cat F1 over pre-F27 row. |
| **MTL-H3-alt** | ⭐ B3 + **per-head LR (cat=1e-3, reg=3e-3, shared=1e-3, constant)** | **45.11 ± 0.32** | 32.42 ± 2.88 | **63.45 ± 2.49** | 51.96 ± 3.41 | 42.15 ± 3.02 | ? | **F48-H3-alt 2026-04-26** — closes 75% of B3-vs-STL gap (53.82 → 66.74) while preserving cat F1. CH18 Tier B → A. |
| — | F40 scheduled-static (cat_weight 0.75 → 0.25 ramp, OneCycleLR) | 44.98 ± 1.05 | 15.49 ± 1.82 | 54.39 ± 3.15 | 41.70 ± 3.24 | 28.04 ± 2.17 | ? | **negative control (2026-04-26)** — cat preserved, reg only +0.57 pp over B3 (Pareto fails). |
| — | F48-H1 single LR const 1e-3, 150ep | 45.34 ± 0.84 | ? | 50.68 ± 6.89 | ? | ? | ? | gentle const refuted — reg-best ep collapses [7..10], σ inflated. |
| — | F48-H2 warmup_constant (50ep ramp + 100ep plateau @ 3e-3, single LR) | 44.45 ± 0.54 | 14.63 ± 1.89 | 48.91 ± 5.12 | 37.83 ± 4.29 | 25.81 ± 2.65 | ? | **negative control (2026-04-26)** — cat preserved but reg DROPS by 4.91 pp vs B3. Cat-vs-reg compete for shared cross-attn. |
| — | F45 single LR const 3e-3, 150ep | **12.23 ± 0.16 💀** | ? | **63.34 ± 2.46** | ? | ? | ? | breakthrough — proves reg arch gains; cat collapses. |
| — | F48-H3 per-head (sh=3e-3, instead of 1e-3) | 19.61 ± 13.34 💀 | ? | 62.04 ± 1.90 | ? | ? | ? | reproduces F45 — confirms shared cross-attn at 3e-3 destabilises cat path. |
| — | MTLoRA r=8 pcgrad (post-fix) | — | 11.31 ± 2.90 | 39.51 ± 3.83 | — | 20.95 ± 2.96 | — | `results/P5_bugfix/az2_mtlora_r8_fairlr_5f50ep_postfix.json` |

### 1.3 Florida (FL, 127 K rows, 4,702 regions)

#### FL 5-fold

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Markov-1-region** (baseline) | — | 46.36 ± 0.89 | 65.05 ± 0.93 | — | 52.37 ± 0.90 | P0 |
| Markov-5-region | — | 42.95 ± 0.69 | 54.91 ± 0.74 | — | 47.07 ± 0.71 | P0 |
| Markov-9-region | — | 42.56 ± 0.68 | 54.10 ± 0.72 | — | 46.53 ± 0.71 | P0 |
| Majority | — | 22.25 | 22.25 | — | 22.25 | P0 |
| Top-K popular | — | 22.25 | 33.82 | — | 25.65 | P0 |
| STL GRU reg | — | 44.49 ± 0.51 | 68.33 ± 0.58 | — | 52.74 ± 0.45 | P1 |
| STL STAN reg | 🔴 F6 pending | | | | | |
| STL GETNext-hard reg (matched-head F37) | 🔴 F37 4050-assigned | | | | | pending — STL ceiling for FL H3-alt comparison |
| **MTL-H3-alt** ⭐ | **67.92 ± 0.72** | 50.27 ± 0.55 | ✅ **71.96 ± 0.68** | 63.62 ± 0.80 | 56.96 ± 0.55 | **F48-H3-alt FL 2026-04-26** — first 5f H3-alt FL run. σ excepcionalmente baixa (N=127k). cat +2.20 pp over F32 B3 1f (65.72); reg +6.70 pp over F32 B3 1f (65.26). Used `--batch-size 1024` to avoid MPS OOM at fold 2 (bs=2048 silent kill). |

#### FL 1-fold (ablation evidence — k=2 CV first fold)

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | Notes |
|---|---:|---:|---:|---:|---:|---|
| STL Check2HGI cat | **63.17** | — | — | — | — | P1_5b FL 1f |
| dselectk + pcgrad + GRU MTL | 64.78 | 15.43 | 57.05 | — | 27.49 | FL P2-validate |
| cross-attn + pcgrad + GRU MTL | 66.46 | — | 57.60 | — | — | FL P2 |
| cross-attn + pcgrad + STAN d=256 MTL | 66.16 | 12.09 | 57.71 | — | 24.51 | P8 sanity |
| **cross-attn + pcgrad + GETNext-SOFT d=256 MTL** (B-M13) | 66.01 | 12.74 | 60.62 | 36.01 | 25.55 | prior north-star FL 1f |
| cross-attn + pcgrad + GETNext-HARD d=256 MTL | 55.43 | 13.70 | 58.88 | 49.54 | 28.01 | **hard-under-pcgrad cat starvation** — diagnosed in `research/B5_FL_SCALING.md` |
| **B3 (F2 Phase B3)** — static cat=0.75 + hard ⭐ | 66.23 | 13.46 | 65.82 | 39.88 | 27.94 | **Pareto-dominates soft**: +0.22 cat, +5.20 Acc@10, +3.87 Acc@5, +2.39 MRR |
| **B3 (F17 fold 1)** — independent replicate ⭐ | **67.06** | 16.36 | **66.55** | 53.60 | 31.29 | diagnostic-task-best epoch · +1.05 cat, +5.93 Acc@10 vs B-M13 |

---

## 2 · Headline states (FL + CA + TX 5-fold) — paper primary table

**Status as of 2026-04-26:** FL 5-fold landed for MTL-H3-alt (this session). CA + TX data pipelines not yet built (F22-F25).

| State | cat POI-RGNN | cat STL | cat **MTL-H3-alt** | reg Markov-1 | reg STL STAN | reg STL GETNext-hard | reg STL GRU | reg **MTL-H3-alt** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FL** | 34.49 | 63.17 (n=1) | **67.92 ± 0.72** ✅ | 65.05 ± 0.93 | 🔴 F6 | 🔴 F37 4050 | 68.33 ± 0.58 | **71.96 ± 0.68** ✅ |
| **CA** | 31.78 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F21+F24 | 🔴 F24 | 🔴 F24 |
| **TX** | 33.03 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F21+F25 | 🔴 F25 | 🔴 F25 |

FL row now populated. Compared to predecessor B3 (1f): cat 65.72 → H3-alt 67.92 (+2.20 pp), reg 65.26 → 71.96 (+6.70 pp). MTL-H3-alt strictly beats Markov-1-region (+6.91 pp) and STL GRU (+3.63 pp) on reg without losing cat F1.

---

## 3 · Cross-state "best-of-each" summary (5-fold data available)

| State | cat F1 best | Method | reg Acc@10 best | Method | Joint (H3-alt) |
|---|---:|---|---:|---|---|
| AL | **42.71 ± 1.37** (B3 post-F27) | next_gru cat | **74.62 ± 3.11** (MTL-H3-alt) ✅ | per-head LR (1e-3/3e-3/1e-3) | **cat 42.22 / reg 74.62** ⭐ |
| AZ | **45.81 ± 1.30** (B3 post-F27) | next_gru cat | 66.74 ± 2.11 (STL F21c) | STL GETNext-hard matched | **cat 45.11 / reg 63.45** (75% gap closed) |
| FL | **67.92 ± 0.72** (MTL-H3-alt) ✅ | per-head LR | **71.96 ± 0.68** (MTL-H3-alt) ✅ | per-head LR | **cat 67.92 / reg 71.96** ⭐ |

**Reading under the H3-alt champion constraint ("single MTL model for both tasks, joint cat+reg paper claim"):**

- **H3-alt jointly wins on FL** (best cat AND best reg, both ⭐) — first time MTL is the per-state champion on both heads simultaneously at the headline state.
- **H3-alt exceeds STL F21c ceiling on AL** (+6.25 pp reg Acc@10), preserving cat F1 within σ of B3. **Strict MTL-over-matched-STL win on reg at AL.**
- **H3-alt closes 75% of the F21c gap on AZ** (53.82 → 63.45 vs 66.74 ceiling). MTL no longer trails by 12-14 pp; residual gap is 3.3 pp within ~1.5σ.
- **CH18 reframed Tier B → A.** The "MTL trails STL by 12-14 pp on reg" finding was a single LR-schedule confound; per-head LR resolves it. See `MTL_ARCHITECTURE_JOURNEY.md`.
- **Three negative controls bracket H3-alt as unique** in this design space — F40 (loss-side), F48-H1 (gentle const), F48-H2 (warmup+plateau). See `research/F40_*` and `research/F48_H2_*`.

---

## 4 · Archived / superseded rows

Kept in source of truth for audit but not referenced in the paper:

| Row | What | Reason |
|---|---|---|
| Pre-fix B-M2 MTLoRA r=8 AL (50.72 ± 4.36) | pre-partition-bug value | SUPERSEDED by B-M2b (53.71 ± 3.80) |
| All AdaShare MTL rows (pre-fix + post-fix) | AdaShare trails MTLoRA by ~9 pp post-fix | Drop from paper; not a competitive MTL sharing mechanism on this task |
| Pre-B5 "MTL-dselectk is the champion" framing | Pre-B5 incorrect narrative | Replaced by B3 (post-F2) |
| FL hard + pcgrad (55.43 / 58.88) | FL-scale gradient starvation | Known failure mode; kept in table as the mechanism anchor for `research/B5_FL_SCALING.md` |

---

## 5 · Index of source JSONs

| Group | Location |
|---|---|
| **F48-H3-alt (AL + AZ 5f, batch=2048)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260425_18*/summary/full_summary.json` |
| **F48-H3-alt FL 5f, batch=1024** | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs1024_ep50_20260426_0045/summary/full_summary.json` |
| **F40 scheduled-static (AL + AZ 5f)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep50_20260426_08*/summary/full_summary.json` |
| **F48-H2 warmup_constant (AL + AZ 5f, 150ep)** | `results/check2hgi/{alabama,arizona}/mtlnet_lr1.0e-04_bs2048_ep150_20260426_09*/summary/full_summary.json` |
| **F45 / F48-H1 / F48-H3 control runs** | per-state under `results/check2hgi/<state>/mtlnet_*_20260425_*/summary/full_summary.json` (see `research/F44_F48_LR_REGIME_FINDINGS.md` for matrix) |
| F31 B3 post-F27 (AL + AZ 5f) | superseded older B3_validation by post-F27 — see `results/F27_validation/al_5f50ep_b3_cathead_gru.json` and AZ counterpart |
| F21c STL GETNext-hard (AL + AZ 5f) | `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json` |
| B3 validation (pre-F27, AL + AZ 5f) | `results/B3_validation/{al,az}_5f50ep_b3.json` |
| B5 hard-index (AL + AZ 5f + FL 1f) | `results/B5/{al_5f50ep,az_5f50ep,fl_1f50ep}_next_getnext_hard.json` |
| F2 diagnostic (4 × FL 1f) | `results/F2_fl_diagnostic/fl_1f50ep_hard_{pcgrad_ckpt,static_cat0.25,static_cat0.50,static_cat0.75}.json` |
| F17 partial (FL fold 1 only) | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260423_0630/folds/fold1_info.json` |
| MTLoRA post-fix suite | `results/P5_bugfix/*.json` |
| P8 SOTA MTL-STAN / TGSTAN / STA-Hyper / GETNext | `results/P8_sota/*.json` |
| STL cat (P1.5b) | `results/P1_5b/next_category_{alabama,arizona,florida}_*_5f_50ep_fair.json` |
| STL reg heads (P1) | `results/P1/region_head_*_region_5f_50ep_*.json` |
| Simple baselines (Markov k=1..9, Majority, Top-K, Random) | `results/P0/simple_baselines/{alabama,florida}/*.json`; AZ pending |
| Historical comparisons (pre-B3 framing) | `BASELINES_AND_BEST_MTL.md` (kept for audit; annotated 2026-04-23) |
