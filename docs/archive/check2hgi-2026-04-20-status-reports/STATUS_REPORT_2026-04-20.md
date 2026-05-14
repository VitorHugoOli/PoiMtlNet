# Check2HGI Study — Status Report (2026-04-20)

Self-contained snapshot of where we are. Complements `MASTER_PLAN.md` (the plan) and `BASELINES_AND_BEST_MTL.md` (the headline-table source of truth).

## TL;DR

- **Substrate claim (CH16) locked**: Check2HGI STL cat F1 38.58 vs HGI 20.29 on AL — **+18.30 pp**, non-overlapping σ envelopes. Paper's strongest claim.
- **Best MTL architecture**: `mtlnet_crossattn + pcgrad`. On AL matches STL cat (38.47 vs 38.58), on AZ exceeds STL (43.13 vs 42.08, +1.05), on FL exceeds STL (66.46 vs 63.17, +3.29 — **paper headline**).
- **Scale-curve discovered (2026-04-19)**: cat benefit increases monotonically with data scale; reg penalty widens monotonically with class-count. `SCALE_CURVE.md` documents the three-state trend.
- **Fair LR fix (2026-04-19)**: max_lr=0.003 reshaped the whole narrative — dilution *reverses sign* at fair LR (cat loss now *helps* reg by +1.26 pp, was hurting by −2.65 at old LR).
- **Nash-MTL tested on AZ (2026-04-19)**: statistically tied with PCGrad. Pareto-equilibrium framing does not beat gradient projection at this scale. Result in `research/NASH_MTL_AZ_FINDINGS.md`.
- **Currently running**: 4-experiment chain (AZ3 dselectk, H-R1 hd=384, H-R4 cat_weight=0.3, FL λ=0.0). PID 49661, ~4 h ETA total. Log: `/tmp/check2hgi_logs/chain_4exp.log`.

## Phase status table

| Phase | Status | Evidence | Notes |
|---|---|---|---|
| **P0** Preparation + simple baselines | ✅ **complete** | `results/P0/` Markov-region floor, POI-level floor, audits | CH01–CH04, CH14, CH15 gates passed |
| **P1** Region-head ablation (AL, single-task) | ✅ **complete** | `results/P1/` — GRU 56.94 ± 4.01 A@10 (champion), TCN 56.11 ± 4.02 (tied within σ) | CH06 region-head champion locked to GRU |
| **P1.5** HGI vs Check2HGI region-task (AL) | ✅ **complete** | `P1.5` comparison — tied, as expected (region embedding not CH2HGI-specific) | Confirms Check2HGI's benefit is *category-specific* |
| **P1.5b** HGI vs Check2HGI category-task (AL) | ✅ **complete, substrate claim locked** | `P1_5b/` — CH16 +18.30 pp | Paper's primary substrate claim |
| **P2** MTL arch × optim ablation | ✅ **complete (unfair + fair LR)** | 10 JSONs in `results/P2/`. Fair-LR reruns (`rerun_R1..R5`) on AL; `FINAL_ABLATION_SUMMARY.md` | Cross-attention champion at fair LR |
| **P3** Arizona scale validation | 🟡 **2/3 done**, AZ3 running now | `az1_crossattn` + `az2_mtlora_r8` done; **AZ3 dselectk running** (~40 min) | Closes Phase 3 table |
| **P4** Florida replication (scale-curve test) | 🟡 **1-fold champion confirmed**; 5-fold + λ=0.0 pending | `fl_crossattn_fairlr_1f50ep.json` (66.46 cat, 57.60 reg), `validate_fl_dselectk_pcgrad_gru_1f_50ep.json` | 1-fold std not computed → headline rests on n=1 |
| **P5** Ablations (cross-attn micro-design, dilution decomposition) | 🟡 **partial** | Architectural-overhead (λ=0.0) confirmed at fair LR on AL. FL λ=0.0 pending (in current chain) | Two legs done, FL leg running |
| **P6** Check2HGI encoder enrichment | ⛔ **not started** (literature survey only) | `research/SOTA_MTL_BACKBONE_DILUTION.md` | Deferred; current findings obsolete the original motivation |

## Key research findings (chronological)

### 1. Check2HGI is the right substrate for next-category (CH16, 2026-04-16)
STL cat F1: **38.58 ± 1.23** (Check2HGI) vs **20.29 ± 1.34** (HGI), AL 5f×50ep. Non-overlapping σ. Beats published POI-RGNN (31.8–34.5%) by 4–7 pp.

### 2. P2 champion at *unfair* LR (2026-04-18): dselectk + pcgrad
AL: cat 36.08, reg 48.88. Below STL on cat (−2.5 pp) → "capacity ceiling" narrative.

### 3. Fair LR rewrites the story (2026-04-19)
User observed: STL cat uses max_lr=0.01, STL reg GRU uses 0.003, MTL was using 0.001. Fair rerun at max_lr=0.003 on AL:

| Config | cat F1 (old LR → new) | reg A@10 (old LR → new) |
|---|---|---|
| mtlnet baseline | 35.8 → 37.31 | 42.5 → 50.03 |
| **cross-attn + pcgrad** ⭐ | 35.6 → **38.47** | 41.3 → **52.41** |
| MTLoRA r=8 | 36.0 → 36.95 | 44.5 → 53.13 |
| λ=0.5 equal-weight | 34.5 → 37.00 | 38.2 → 51.86 |

**Cross-attn went from tied-for-last to champion on both heads.** Δm improved +7.68 pp (largest of any config).

### 4. Dilution sign reverses at fair LR (2026-04-19)
At max_lr=0.001: cat loss hurt reg by −2.65 pp. At max_lr=0.003: cat loss *helps* reg by +1.26 pp. Multi-task training is *net positive transfer* at fair LR.

### 5. Scale-curve discovered (2026-04-19)
Three-state curve (AL 10K → AZ 26K → FL 127K rows):
- **cat Δ (MTL − STL)**: −0.11 → +1.05 → **+3.29** (monotone up with data)
- **reg Δ (MTL − STL)**: −4.53 → −7.81 → −10.73 (monotone down with class count)

Mechanistic claim: **MTL's help-vs-hurt depends on the ratio of output-space cardinality to training-data scale**. Cat (7 classes) benefits more as data grows; reg (1K→5K classes) needs proportional shared-backbone capacity, which stays fixed. Documented in `results/SCALE_CURVE.md`.

### 6. Nash-MTL tested on AZ (2026-04-20 night)
User hypothesis: Nash's Pareto-equilibrium framing should help a bidirectional task pair. **Result: tied with PCGrad within σ.** Δ cat +0.22 pp, Δ reg −0.93 pp — both within per-fold variance. Interpretation: cross-attn's content-based routing already suppresses gradient conflict, so PCGrad's projections have little to act on. Details in `research/NASH_MTL_AZ_FINDINGS.md`.

## Paper framing (revised)

| Claim ID | Statement | Status |
|---|---|---|
| **CH16** | Check2HGI beats HGI on next-category | ✅ locked (+18.30 pp, non-overlapping σ) |
| **CH17** | Check2HGI beats POI-RGNN and prior HGI work | ✅ locked (+4–7 pp vs published) |
| **CH-M1** | MTL is task-asymmetric: cat benefits, reg dilutes | ⚠️ rewritten after fair-LR fix: at fair LR, cat benefits AND reg positively transfers (dilution sign flip) |
| **CH-M3** | Architectural overhead ≈ 5 pp (constant across LR) | ✅ confirmed AL (51.87 vs 56.94); FL leg running |
| **CH-M4** | Cross-attention uniquely closes weak-head gap | ✅ strengthened: matches/exceeds STL on cat across all three states |
| **CH-M5** (new) | HP calibration dominates architecture choice | ✅ established: +4–8 pp Δm just from max_lr 0.001 → 0.003 |
| **CH-M6** (new) | Scale-curve: cat benefit grows with data, reg penalty grows with cardinality | ✅ demonstrated at AL/AZ/FL |
| **CH-MX** | Nash-MTL improves over PCGrad on Pareto-bidirectional | ❌ rejected at AZ scale |

**Headline paragraph (draft):** Cross-attention MTL on Check2HGI achieves **+3.29 pp next-category F1 over single-task on Florida** (66.46 vs 63.17, seed 42, 1f×50ep — 5-fold replication pending). The same architecture regresses next-region by 10.73 pp, and the regression decomposes into ~5 pp of LR-invariant architectural overhead (confirmed at AL) plus ~5.7 pp of cardinality-scaling dilution (FL has 4.2× more region classes than AL).

## Infrastructure state

| Issue | Status | Workaround / Fix |
|---|---|---|
| **I1** `--no-checkpoints` ignored on MTL check2hgi path | ✅ **fixed** (commit `10889ba`) | Proper `_NO_CHECKPOINTS` short-circuit in `_run_mtl_check2hgi` |
| **I2** Thunderbolt SSD intermittent flakes | ⚠️ workaround in use | `OUTPUT_DIR=/tmp/check2hgi_data` on all long runs |
| **I3** max_lr=0.01 run corrupted by Mac going idle | ⚠️ quarantined | `ablation_07_maxlr_0.01_QUARANTINED.md`, `--no-checkpoints` now saves no data |
| **I4** (new) Cross-attn + GRU hd=512 MPS OOM mid-fold 5 | ⚠️ identified | Retry at hd=384 in current chain |

## Currently running (2026-04-20 ~02:14 onwards)

Chain of 4 sequential experiments, each wrapped in 3-attempt retry:

| # | Experiment | Config | ETA | Result path |
|---|---|---|---|---|
| 1 | **AZ3 dselectk baseline** | `mtlnet_dselectk + pcgrad` AZ 5f×50ep | ~40 min (running now) | `P2/az3_dselectk_fairlr_5f50ep.json` |
| 2 | H-R1 retry at hd=384 | cross-attn + pcgrad + GRU hd=384 AZ 5f×50ep | ~45 min | `P2/hr1_crossattn_gru_hd384_az_5f50ep.json` |
| 3 | H-R4 cat_weight=0.3 | cross-attn + static_weight 0.3/0.7 AZ 5f×50ep | ~45 min | `P2/hr4_crossattn_static_cat0.3_az_5f50ep.json` |
| 4 | **FL λ=0.0 isolation** | dselectk + static_weight 0.0 FL 1f×50ep | ~90 min | `P2/fl_lambda0_dselectk_fairlr_1f50ep.json` |

Log: `/tmp/check2hgi_logs/chain_4exp.log`. Total ETA ~3.5 h.

## Pending (not launched)

1. **FL cross-attn 5-fold** (~8 h) — tightens the +3.29 pp headline number.
2. **Hybrid architecture** (cross-attn cat + dselectk reg, ~2-3 h) — speculative, run after H-R1/H-R4 results.
3. **Multi-seed n=15 on FL** (P3 headline, ~15 h) — required if paper needs tight CIs.
4. **CA + TX replication** (~12–18 h total) — three-state paper table, deferred to end.
5. **P6 Check2HGI encoder enrichment** — deferred indefinitely; current findings already sufficient for paper.

## Result file index

| Scope | Path |
|---|---|
| Headline table | `docs/studies/check2hgi/results/BASELINES_AND_BEST_MTL.md` |
| Fair-LR ablation | `docs/studies/check2hgi/results/P2/FINAL_ABLATION_SUMMARY.md` |
| Scale curve | `docs/studies/check2hgi/results/SCALE_CURVE.md` |
| Nash findings | `docs/studies/check2hgi/research/NASH_MTL_AZ_FINDINGS.md` |
| Region hparam plan | `docs/studies/check2hgi/research/REGION_HPARAM_PLAN.md` |
| Known infra issues | `docs/studies/check2hgi/research/KNOWN_INFRA_ISSUES.md` |
| State machine | `docs/studies/check2hgi/state.json` (stale: says P0 running; reality is P5 mostly done) |

## Recent commits (worktree-check2hgi-mtl)

- `bb153d5` docs(study): Nash-MTL on AZ — statistically tied with PCGrad
- `10889ba` fix(train): honor --no-checkpoints on _run_mtl_check2hgi path
- `ae6d320` docs(study): record known infra issues (SSD + --no-checkpoints bug)
- `97f7fdb` fix: fix the gradnorm
- `2059ea9` feat(train): add --no-checkpoints flag to skip model weight saves
