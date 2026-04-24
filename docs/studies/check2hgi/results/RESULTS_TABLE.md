# Results Table — Check2HGI MTL Study

**Last updated:** 2026-04-23 (post doc-cleanup, B3 validation through F19).

**Champion:** `mtlnet_crossattn + static_weight(category_weight=0.75) + next_getnext_hard d=256, 8 heads` — see [`../NORTH_STAR.md`](../NORTH_STAR.md). Scope, baselines set, and STL-matching policy defined in [`../PAPER_STRUCTURE.md`](../PAPER_STRUCTURE.md).

**Common protocol** (unless noted per row): user-disjoint `StratifiedGroupKFold(groups=userid)`, 5 folds × 50 epochs, seed 42, AdamW (lr=1e-4 → OneCycleLR max_lr=0.003), batch 2048, `gradient_accumulation_steps=1`. Reg metrics are `*_indist` (restricted to regions seen in training set of the fold) where applicable.

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
| STL GETNext-hard reg (matched-head) — **F21** | — | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | pending |

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
| **B-M_B3** | ⭐ **cross-attn + static (cat=0.75) + GETNext-HARD d=256** | **39.28 ± 0.80** | ? | 56.33 ± 8.16 | 42.81 ± 7.89 | 28.55 ± 5.33 | 9.43 ± 0.71 | **B3 validation AL (F18, 2026-04-23)** |
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
| **STL STAN reg** ⭐ | — | 24.48 ± 2.29 | **52.24 ± 2.38** | 40.41 ± ? | 33.70 ± 2.36 | 24.48 ± 2.29 | P1 SOTA (also source of per-fold data for F1/F19 Wilcoxon) |
| STL GETNext-hard reg — **F21** | — | 🔴 | 🔴 | 🔴 | 🔴 | 🔴 | pending |

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
| **B-M_B3** | ⭐ **cross-attn + static (cat=0.75) + GETNext-HARD** | **43.62 ± 0.74** | 14.27 ± 2.53 | 52.76 ± 3.92 | 39.68 ± 3.61 | 26.40 ± 2.45 | **9.17** | **B3 validation AZ (F19, 2026-04-23)** · vs B-M9d: +1.40 cat, −0.49 Acc@10 (tied) · vs B-M9b soft: +0.80 cat, +6.10 reg Acc@10 · vs STL STAN: **+1.54 cat (p=0.0312 cat F1), +0.52 Acc@10 (tied), +3.75 reg F1 (p=0.0312)** — see `research/B3_AZ_WILCOXON_VS_STL.md` |
| — | MTLoRA r=8 pcgrad (post-fix) | — | 11.31 ± 2.90 | 39.51 ± 3.83 | — | 20.95 ± 2.96 | — | `results/P5_bugfix/az2_mtlora_r8_fairlr_5f50ep_postfix.json` |

### 1.3 Florida (FL, 127 K rows, 4,702 regions)

#### FL 5-fold (headline target — **pending clean run**)

Pending:
- **F4** MTL-B3 5f clean re-run (prior attempts F17 killed mid-CV; F20 per-fold persistence now enabled).
- **F6** STL STAN 5f.
- **F9** STL HGI cat 5f.

Current 5-fold entries available:

| Method | cat F1 | reg Acc@1 | reg Acc@10 | reg Acc@5 | reg MRR | Notes |
|---|---:|---:|---:|---:|---:|---|
| **Markov-1-region** (baseline) | — | 46.36 ± 0.89 | **65.05 ± 0.93** | — | 52.37 ± 0.90 | P0 |
| Markov-5-region | — | 42.95 ± 0.69 | 54.91 ± 0.74 | — | 47.07 ± 0.71 | P0 |
| Markov-9-region | — | 42.56 ± 0.68 | 54.10 ± 0.72 | — | 46.53 ± 0.71 | P0 |
| Majority | — | 22.25 | 22.25 | — | 22.25 | P0 |
| Top-K popular | — | 22.25 | 33.82 | — | 25.65 | P0 |
| **STL GRU reg** ⭐ | — | 44.49 ± 0.51 | **68.33 ± 0.58** | — | 52.74 ± 0.45 | P1 |
| STL STAN reg | 🔴 F6 pending | | | | | |

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

**Status as of 2026-04-23:** CA and TX data pipelines not yet built. FL 5-fold pending clean re-run.

| State | cat Markov/Majority | cat POI-RGNN | cat STL | cat **MTL-B3** | reg Markov-1 | reg STL STAN | reg STL GETNext-hard | reg STL GRU | reg **MTL-B3** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FL** | 37.2 / 22.25 | 34.49 | 63.17 (n=1) | 66.23/67.06 (n=1 ×2) 🔴 5f | **65.05** | 🔴 F6 | 🔴 F21 | 68.33 | 58.88/65.82/66.55 (n=1 ×3) 🔴 5f |
| **CA** | 🔴 pipeline F22 | 31.78 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F24 | 🔴 F21+F24 | 🔴 F24 | 🔴 F24 |
| **TX** | 🔴 pipeline F23 | 33.03 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F25 | 🔴 F21+F25 | 🔴 F25 | 🔴 F25 |

This table populates as `FOLLOWUPS_TRACKER.md §1` items complete.

---

## 3 · Cross-state "best-of-each" summary (ablation-available states)

For the states where we have 5-fold data, this is the best we can publish right now:

| State | cat F1 best | Method | reg Acc@10 best | Method | Joint (B3 config) |
|---|---:|---|---:|---|---|
| AL | **39.28 ± 0.80** (B3) | B3 (cat=0.75) + GETNext-hard | **59.20 ± 3.62** (STL) | STL STAN | B3 cat 39.28 / Acc@10 56.33 |
| AZ | **43.62 ± 0.74** (B3) | B3 | **53.25 ± 3.44** (MTL) | B-M9d hard+pcgrad MTL · STL STAN 52.24 | B3 cat 43.62 / Acc@10 52.76 |
| FL | **66.01 / 66.23 / 67.06** (n=1 × 3) | B-M13 soft, B3 F2, B3 F17 | — 5f pending | STL GRU 68.33 is current best · Markov saturates at 65.05 | B3 cat 66.23 / Acc@10 65.82 (n=1) |

**Reading under the champion constraint ("single model for both tasks, beats baselines, improves STL where possible"):**

- **B3 is the single joint champion.** It beats every baseline on cat F1 across all states (AL +7, AZ +1.5, FL +28+ vs POI-RGNN), and beats Markov-1-region on AL + AZ (+12 / +10 pp Acc@10).
- **FL reg Acc@10** is the known limitation (`PAPER_STRUCTURE.md §6` — approach (a)). Markov-1-region saturates; every neural method is within ±3 pp of it. B3's n=1 Acc@10 = 65.82 matches Markov at n=1; n=5 still pending.
- **MTL-over-STL on cat** is strict at AZ (p=0.0312 F1 Wilcoxon); tied within σ at AL; positive n=1 at FL.
- **MTL-over-STL on reg** needs F21 (STL GETNext-hard matched-head) to be honestly assessed. Currently B3 is tied-within-σ with STL STAN on AL + AZ; FL pending.

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
| B3 validation (AL + AZ 5f) | `results/B3_validation/{al,az}_5f50ep_b3.json` |
| B5 hard-index (AL + AZ 5f + FL 1f) | `results/B5/{al_5f50ep,az_5f50ep,fl_1f50ep}_next_getnext_hard.json` |
| F2 diagnostic (4 × FL 1f) | `results/F2_fl_diagnostic/fl_1f50ep_hard_{pcgrad_ckpt,static_cat0.25,static_cat0.50,static_cat0.75}.json` |
| F17 partial (FL fold 1 only) | `results/check2hgi/florida/mtlnet_lr1.0e-04_bs2048_ep50_20260423_0630/folds/fold1_info.json` |
| MTLoRA post-fix suite | `results/P5_bugfix/*.json` |
| P8 SOTA MTL-STAN / TGSTAN / STA-Hyper / GETNext | `results/P8_sota/*.json` |
| STL cat (P1.5b) | `results/P1_5b/next_category_{alabama,arizona,florida}_*_5f_50ep_fair.json` |
| STL reg heads (P1) | `results/P1/region_head_*_region_5f_50ep_*.json` |
| Simple baselines (Markov k=1..9, Majority, Top-K, Random) | `results/P0/simple_baselines/{alabama,florida}/*.json`; AZ pending |
| Historical comparisons (pre-B3 framing) | `BASELINES_AND_BEST_MTL.md` (kept for audit; annotated 2026-04-23) |
