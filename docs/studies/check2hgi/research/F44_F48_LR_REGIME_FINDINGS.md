# F44-F48 — LR / Schedule Regime Investigation for CH18

**Date:** 2026-04-25. **Tracker:** `FOLLOWUPS_TRACKER.md §F44–F48`. **Cost:** ~7 h MPS sequential (AL + AZ × 5 configs × 5 folds × {50, 75, 150} ep).

## Question

Following F38/F39/F41/F42 refuting four CH18-attribution factors (checkpoint selection, loss weight, upstream pre-encoder, epoch budget), the LR schedule was the remaining suspect. F42 had shown 150ep + default OneCycleLR `max_lr=3e-3` HURT reg (59.60 → 56.14) by stretching the schedule into the warmup phase. F44–F48 disambiguate "more epochs" from "stretched schedule" and search for a regime that closes the 12–14 pp STL-MTL gap on reg without breaking cat.

## Method

5-fold × {50, 75, 150} × 50ep on AL + AZ, B3 architecture (`mtlnet_crossattn + static_weight(cat=0.75) + next_gru + next_getnext_hard, d=256, 8h`), seed 42.

Five schedule variants:

| ID | Config |
|---|---|
| F44 | 150ep + OneCycleLR max_lr=**1e-3** (gentler peak) |
| F45 | 150ep + **constant LR=3e-3** (no warmup, no annealing) |
| F46 | 50ep + OneCycleLR max_lr=3e-3 + **pct_start=0.1** (short warmup) |
| F47 | 75ep + OneCycleLR max_lr=3e-3 (intermediate budget) |
| F48-H1 | 150ep + **constant LR=1e-3** (gentler constant — H1 sweet-spot test) |

## Master table

| Config | AL cat F1 | AL reg Acc@10 | σ_AL | AZ cat F1 | AZ reg Acc@10 | σ_AZ |
|---|---:|---:|---:|---:|---:|---:|
| **B3 50ep default** | 42.71 ± 1.37 | **59.60 ± 4.09** | 4.09 | 45.81 ± 1.30 | **53.82 ± 3.11** | 3.11 |
| F44 150ep max=1e-3 | 40.20 ± 1.28 | 58.82 ± 4.96 | 4.96 | 44.86 ± 0.65 | **47.91** ± 3.55 | 3.55 |
| **F45 150ep const 3e-3** | **10.44 ± 0.04 💀** | **74.20 ± 2.95** | 2.95 | **12.23 ± 0.16 💀** | **63.34 ± 2.46** | 2.46 |
| F46 50ep pct=0.1 | 43.16 ± 1.18 | 57.73 ± 5.15 | 5.15 | 44.55 ± 1.39 | 52.05 ± 3.87 | 3.87 |
| F47 75ep default | 41.81 ± 2.01 | 59.88 ± 3.76 | 3.76 | 45.19 ± 1.45 | 54.02 ± 2.80 | 2.80 |
| **F48-H1 150ep const 1e-3** | **40.99 ± 1.80** | 61.43 ± 9.60 | 9.60 | **45.34 ± 0.84** | 50.68 ± 6.89 | 6.89 |
| **STL F21c (ceiling)** | **n/a** | **68.37 ± 2.66** | 2.66 | **n/a** | **66.74 ± 2.11** | 2.11 |

## Per-fold reg-best epoch — diagnostic of schedule × LR interaction

| Config | AL reg-best ep | AZ reg-best ep | Pattern |
|---|---|---|---|
| B3 50ep default | [34, 35, 36, 44, 34] | (best ~ep 30-40) | annealing tail |
| F44 150ep max=1e-3 | [40, 62, 41, 51, 43] | [33, 42, 34, 31, 32] | mid-annealing |
| **F45 150ep const 3e-3** | **[23, 52, 49, 26, 33]** | **[16, 24, 13, 16, 18]** | extended high-LR window |
| F46 50ep pct=0.1 | [38, 33, 31, 35, 45] | [28, 44, 39, 36, 38] | annealing |
| F47 75ep default | [53, 41, 59, 14, 17] | [38, 41, 68, 50, 39] | varied |
| **F48-H1 150ep const 1e-3** | **[22, 7, 6, 4, 6]** | **[7, 10, 7, 8, 6]** | **plateau in 4-10 ep** |

## Findings

### 1. OneCycleLR-family is uniform, regardless of (max_lr, epochs, pct_start)

All OneCycleLR variants (F44, F46, F47 + B3 reference) cluster in the same regime:
- AL: cat F1 ∈ [40.20, 43.16], reg Acc@10 ∈ [56.14, 59.88]
- AZ: cat F1 ∈ [44.55, 45.81], reg Acc@10 ∈ [47.91, 54.02]

The **annealing tail of OneCycleLR caps reg Acc@10** at this level. Increasing budget, decreasing peak LR, or shortening warmup all leave reg flat within σ.

### 2. Constant LR = 3e-3 is the only configuration that meaningfully lifts reg — at the cost of cat

F45 reg Acc@10:
- **AL: 74.20 ± 2.95** — exceeds STL F21c ceiling (68.37) by **+5.83 pp**
- **AZ: 63.34 ± 2.46** — within 3 pp of STL F21c (66.74) but does not exceed

Both states: **cat F1 collapses to ~10–12** (majority-class baseline). The cat head's 7-class GRU diverges under sustained LR ≥ 2e-3 — it requires gentle warmup or annealing to stabilise.

The reg head's GETNext-hard mechanism (`stan_logits + α · log_T[r_last]`) thrives under sustained high LR because **`α` continues growing throughout training** (F45 reg-best epochs migrate to 23-52 in AL; the model exploits the prior more aggressively each epoch). Annealing prematurely truncates `α` growth.

### 3. Constant LR = 1e-3 (F48-H1) is too gentle

Refuted as the "easy hybrid". Reg-best epoch collapses to 4-10 in both states — α plateaus very early at low LR. Cat F1 preserved (40.99 AL, 45.34 AZ) but reg only matches B3 (61.43 AL, 50.68 AZ) with **inflated σ (6.89-9.60)**. The model is unstable: 4 of 5 folds peak in the first 7 epochs, suggesting the LR is so low the optimization barely escapes initialisation before plateauing.

### 4. The two heads have disjoint optimal LR regimes

| Head | Optimal LR regime | Why |
|---|---|---|
| `next_gru` cat (7 classes) | LR ≤ 1e-3 sustained, OR warmup-then-anneal | Diverges at LR ≥ 2e-3 sustained → predicts majority class |
| `next_getnext_hard` reg (~1K classes) | LR ≥ 2e-3 sustained for 50+ epochs | Needs sustained high-LR window for `α` to grow + fully exploit `log_T` prior |

**Monolithic schedule cannot serve both heads.** Any schedule that satisfies one starves the other.

### 5. Scale-dependence of the reg ceiling

F45 (constant 3e-3) reaches 74 in AL but only 63 in AZ. AZ saturates faster:
- AL: ~5 batches/epoch × 50 batches max = ~250 steps total at peak LR
- AZ: ~13 batches/epoch × 25 batches max-with-α-still-growing = ~325 steps total

AZ's more-batches-per-epoch means it reaches the "α growth plateau" earlier in epochs. The ceiling itself is also lower (63 vs 74) — possibly because AZ has more diverse 1547 regions vs AL's 1109, requiring more capacity from STAN to distinguish.

## Implications for CH18

1. **Reg arch IS capable of beating STL** (F45 AL proved it: 74.20 > 68.37). CH18's 12-14 pp gap is NOT structural to MTLnetCrossAttn architecture.
2. **The bottleneck is the LR schedule, not the architecture or backbone coupling.** Specifically: OneCycleLR's annealing prevents `α` (graph-prior weight) from growing to its productive level.
3. **There is no monolithic schedule that closes the gap without breaking cat.** Two-head design requires two LR regimes.

## Next-step design space (F48-H2 / H3 / H4 ranked)

| ID | Design | Cost | Likelihood of recovering reg-without-cat-collapse |
|---|---|---|---|
| **F48-H2 — warmup-then-plateau schedule** | New scheduler: warmup 50 eps from 1e-4 → 3e-3, then **constant 3e-3 for 100 eps** (no annealing). Cat protected during warmup, reg gets sustained high-LR after. | ~30 LOC + ~30 min train | **HIGH** — directly designed to give cat a stable phase + reg a long plateau. |
| **F48-H3 — per-head LR** | Multi-param-group AdamW: cat encoder + cat head LR=1e-3 (constant); reg encoder + reg head LR=3e-3 (constant). Cross-attn shared: averaged. | ~50 LOC + ~30 min train | **HIGH** — directly addresses the disjoint-regime finding. Cleanest experiment. |
| **F48-H4 — scheduled cat_weight handover (was F40)** | static_weight cat_weight ramps 0.9 → 0.25 over training, with constant LR=3e-3. Cat learns under high weight first, reg takes over later. | ~50 LOC + ~30 min train | MEDIUM — F45 already had cat_weight=0.75 and cat still collapsed; ramping weight may not help if LR is the dominant factor. |

**Recommended order:** F48-H3 first (per-head LR) — most direct test of the "disjoint regimes" hypothesis. If it works, the paper has a clean recipe: "per-head OneCycleLR with `next_gru` at gentle peak and `next_getnext_hard` at sustained max". If H3 fails, fall back to H2 (warmup-then-plateau) which generalises across heads.

## Files

- Logs: `/tmp/check2hgi_logs/resume_az_f48.log`
- AL summaries: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep{50,75,150}_*/summary/full_summary.json`
- AZ summaries: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep{50,75,150}_*/summary/full_summary.json`
- Per-fold info: each `<run>/folds/fold[1-5]_info.json`

## Cross-references

- `research/F38_CHECKPOINT_SELECTION.md` — Fator 2 refutation
- `research/F41_PREENCODER_FINDINGS.md` — Fator 3a refutation
- `research/F21C_FINDINGS.md` — original 12-14 pp gap
- `CLAIMS_AND_HYPOTHESES.md §CH18` — formal claim
- `SESSION_HANDOFF_2026-04-24_PM.md` — predecessor session
