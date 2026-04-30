# F50 — Study History (chronological narrative)

**Purpose:** the timeline of the F50 series for future agents who need to understand *why* decisions were made, not just the final state. The decisive events were not hypothesis-driven — many were stumbled into during execution. This doc preserves the sequence so the post-hoc rationale in `F50_T4_FINAL_SYNTHESIS.md` makes sense.

**TL;DR:** F50 started as an ablation/improvement study to close the FL STL→MTL gap. Mid-stream we discovered (a) the original 8.83 pp gap was inflated by a graph-prior leak (~13–17 pp drop after the C4 fix), and (b) a per-batch alternating-SGD intervention (P4) closes a real +3.3 pp under leak-free conditions. The mechanism turned out to be temporal training dynamics, not architectural. Three independent experiments (P1, F52 P5, F53 cw sweep) confirmed the cross-attention mixing is structurally dead at FL.

For full canonical numbers: `F50_T4_FINAL_SYNTHESIS.md` and `F50_RESULTS_TABLE.md`.
For the original tiered plan that bootstrapped this study: `archive/F50/F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md`.

---

## Phase 0 — Setup (pre-2026-04-28)

- Predecessor `B3` champion at AL/AZ closed CH18 substrate-specific MTL.
- F37 STL ceiling at FL = 82.44 ± 0.38 reg @ ≥ep10 (later restated to **71.12 ± 0.59 clean** after C4 fix).
- Phase-1 grid (CH16/CH18) all green at AL/AZ; FL pending.
- Existing concerns: C12 hyperparameter mismatch STL vs MTL, C15 MTL-on-reg coupling, C14 cat-head FL scale-dependence.

## Phase 1 — Tiered plan executed (2026-04-28)

`F50_PROPOSAL_AUDIT_AND_TIERED_PLAN.md` (archived) split the work into Tier 0 (zero-compute audits), Tier 1 (architecture + loss probes), Tier 2 (cross-stitch / PLE / FAMO / Aligned-MTL), Tier 3 (deferred).

**Tier 0 closed 2026-04-28:**
- F50 T0 — joint Δm + paired Wilcoxon AL/AZ/FL: AL +8.70 % p=0.0312; AZ +3.19 %; FL **−1.63 %, p_two_sided=0.0625, 5/5 negative** at n=5 ceiling. Backs CH22 scale-conditional.
- F50 T1.1 — F33 cat-head Path A confirmed (universal `next_gru`, F1 = 68.21 ± 0.42, all 5 folds above pre-F27 envelope). Closes C14.
- F37 STL F21c ceilings landed (FL cat 66.98 / FL reg 82.44 inflated, restated later).
- F49 λ=0 architectural decomposition: AL +6.48 / AZ −6.02 / FL −16.16 pp.

**Tier 1 closed 2026-04-29 morning:**
- T1.2 HSM head — STL +0.21 pp vs flat (architecture preserved); MTL −3.01 pp (rejected).
- T1.3 FAMO — failed acceptance.
- T1.4 Aligned-MTL — failed acceptance.
- → No architectural alternative closes the FL gap.

## Phase 2 — Optimizer / training-dynamics screening (2026-04-29 afternoon)

`F50_T3_HYPERPARAM_BRAINSTORM.md` (archived) proposed B1–B10 + tier-A scheduler/cw combos. Screening landed:
- A1 OneCycle alone: −9 pp Δreg.
- A2 Cosine alone: collapse (σ=8.99).
- A6 cw=0.25 + OneCycle: −9.22 pp.
- D6 reg_head_lr=3e-2 fold-1: spike to top10=77.93 at ep 0 (the α growth IS mechanistically achievable in MTL, but joint training destabilises it).
- D8 cw=0 (cat-loss-removed limit case): reg=74.06, ep-5 plateau → **refuted cat-dominance hypothesis**.
- D1 STL α=0 frozen: encoder-only ceiling = 72.61 (no leak).

**Posthoc selector breakthrough (F61):** instrumenting `min_best_epoch=5` revealed ALL prior MTL "reg-best" epochs were trapped at ep 0–2 by the GETNext α init bias. Re-aggregated 17 prior runs under `min_best_epoch ≥ 5` → reg-best shifted to ep 4–6 across the board. **`F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md` §5.5** reframed F50 closure: the bottleneck is *temporal*, not architectural — STL reaches reg-best at ep 16–20 where α has grown; MTL reg-best is structurally pinned at ep 4–5.

## Phase 3 — P4 alternating-SGD discovery (2026-04-29 ~17:00)

P4 (`--alternating-optimizer-step`) was added as a brainstorm long-shot. Even batches update cat from L_cat only; odd batches update reg from L_reg only. Result on FL leaky:
- P4 alone: +3.83 pp Δreg, p=0.0312, 5/5 positive ✅
- P4 + Cosine (initial champion): +4.63 pp Δreg, +0.55 pp Δcat — **paper-grade Pareto-positive on BOTH tasks for the first time in the study**.

**Tier-A grid (A1–A6) confirmed P4+Cosine as the leak-time champion at 76.07 ± 0.62 reg / 68.51 ± 0.88 cat.**

## Phase 4 — C4 leakage discovery (2026-04-29 ~19:00)

`F50_T3_AUDIT_FINDINGS.md` flagged 8 concerns (C1–C8). C4 was the load-bearing one:

> The GETNext head reads `α · log_T[last_region]`. The transition matrix `region_transition_log.pt` was built from the FULL check-in dataset (train + val). At α=0.1 init the leak is small (~0.5–2 pp). But α grows during training to ~1.8 (18× amplification). The trained model's val ranking is dominated by the leaked prior.

**Audit empirical verification (independent advisor):**
- Direct measurement on FL fold 1: 70.2 % of val transitions present in full-data log_T.
- Prior-only val top10 gap from log_T = 10.19 pp (just from leak, no learned signal).
- α grows 0.1 → 1.8 over 50 epochs; trained-model gap = **15.7 pp absolute drop**.
- All 4 alternative explanations (code bug, indexing bug, filter bug, top10 calc bug) rejected with high confidence.

**Fix:** `--per-fold-transition-dir` builds `log_T` from train-rows only per fold. Landed in commit `60107eb`.

**Re-runs at FL clean:**
- P4+Cosine champion: 76.07 → **63.23 ± 0.64** (−12.84 pp drop).
- H3-alt baseline: 71.44 → **60.12 ± 1.14** (−11.32 pp).
- ALL `next_getnext_hard*` numbers in the study inflated by ~13–17 pp.

**Critical observation — uniform leak:** the drop is approximately uniform across recipes. Relative Δs are preserved → 17 of 18 F50 ablations don't need re-running. Validity matrix in `F50_T4_PRIOR_RUNS_VALIDITY.md`. Validated twice (B9 vs H3-alt = +3.34 pp; PLE vs H3-alt = +0.26 vs leaky +0.25, matches to 0.01 pp).

**Broader audit** (independent agent): C4 propagates to all 5 log_T-loading heads (`next_getnext_hard{,_hsm}`, `next_getnext`, `next_tgstan`, `next_stahyper`). Class weights, samplers, baselines clean ✅. Embeddings have structural full-data training but no learnable amplifier (low severity). Full report: `F50_T4_BROADER_LEAKAGE_AUDIT.md`.

## Phase 5 — Champion re-grounding (2026-04-29 ~19:40 to ~22:00)

Re-runs under leak-free conditions consolidated the headline:

| recipe | reg @≥ep5 | cat F1 @≥ep5 | Δreg vs H3-alt clean | paired Wilcoxon |
|---|---:|---:|---:|---|
| STL F37 ceiling | **71.12 ± 0.59** | n/a | — | — |
| **B9 (P4 + Cosine + α-no-WD)** ⭐ | **63.47 ± 0.75** | **68.59 ± 0.79** | **+3.34** | **p=0.0312, 5/5 on BOTH tasks** |
| P4-alone (constant scheduler) | 63.41 ± 0.77 | 67.82 | +3.28 | p=0.0312, 5/5 |
| P0-A (P4 + Cosine, no α-no-WD) | 63.23 ± 0.64 | 68.51 | +3.11 | p=0.0312, 5/5 |
| F62 two-phase (cw=0→0.75 step) | 60.25 ± 1.26 | n/a | +0.13 | n.s. ❌ REJECTED |
| PLE-lite (clean full 5×50) | 60.38 ± 0.79 | 64.13 ± 1.04 ⚠ | +0.26 | cat **−4.22 pp**, 0/5 ⚠ Pareto-WORSE |
| H3-alt (clean baseline) | 60.12 ± 1.14 | 68.34 | 0 | anchor |

Three findings simplified the paper story:
1. **B9 ≈ P4-alone within 0.06 pp** → the recipe is just `--alternating-optimizer-step`. Cosine + α-no-WD give marginal lift.
2. **F62 two-phase REJECTED** — coarse-grained scheduling does NOT replicate P4's per-batch granularity. Mechanism finding: P4's per-step alternation is essential.
3. **PLE Pareto-WORSE** under leak-free (NEW finding) — expert routing hurts cat without helping reg under clean conditions. Strictly dominated by P4-alone, B9, and P0-A.

**Cross-state portability** (clean from start, AL/AZ/GA): reg @≥ep10 is 49.44 / 40.61 / 46.57 — significantly lower than FL's 60.36. **Recipe doesn't transfer cleanly to small states.** Paper claim becomes "FL-strong; cross-state directional but not paper-grade."

## Phase 6 — Mechanism receipts + audit hygiene (2026-04-29 22:00 → 2026-04-30 01:50)

- **D5 encoder weight-trajectory diagnostic:** Frobenius norm + drift logging on `next_encoder` and `category_encoder`. Reg-side encoder saturates 26–32 epochs before cat encoder under both H3-alt and B9. Under B9, reg saturation aligns to reg-best epoch (both ep 6) — encoder physically stops updating in the same window as the val plateau. Mechanism receipt #2 for the temporal narrative. `F50_D5_ENCODER_TRAJECTORY.md`. Plot: `figs/f50_d5_encoder_trajectory.png`.
- **F63 α-trajectory plot:** smoking-gun figure (`figs/f63_alpha_trajectory.png`) showing α grows late (ep 30+) — confirms the head's per-region prior amplifies AFTER the encoder is done.
- **C7 audit fix:** `aggregation_basis ∈ {joint_best, per_task_f1_best, per_metric_best}` stamped on every aggregate in `full_summary.json`. Closes audit C-series.
- **C05 P3 null-result fallback** decision tree pre-written in case CA/TX MTL fails.

## Phase 7 — Final follow-up suite (2026-04-30 ~01:00)

Three architectural follow-ups + one cw sweep + one CA attempt:

- **B2/F64 warmup-decay reg_head LR** — REJECTED. Δreg = −5.19 pp vs B9 (paired Wilcoxon p=0.0625, 0/5+, 5/5− across all metrics). Pareto-dominated.
- **F52 P5 identity-crossattn** — TIED with B9. Δreg = +0.30 pp (p=0.81, 3/5+). **Cross-attn MIXING is dead at FL**; the productive component is the per-task FFN+LN structure.
- **F65 min_size_truncate joint loader** — TIED with B9. Δreg = +0.00 pp (identical numbers @≥ep5). Joint-loader cycling is NOT the cause of reg saturation.
- **F53 cat_weight sweep** (H3-alt + P1, cw ∈ {0.25, 0.50, 0.75}) — H3-alt ≈ P1 across all cw values; cross-attn does NOT unlock at low cw.
- **CA P3 unblock attempt** — GPU-OOM on 4090 (24 GB) at the per-epoch train-side logit catting line. CA's 8501 regions × ~280K rows × fp32 = 9 GB on top of 17 GB model state. Deferred to A100 per user direction.

**Three-way confirmation that cross-attention mixing is structurally dead at FL:**
- P1 (remove block) ≈ H3-alt
- F52 P5 (zero mixing output, keep FFN+LN) ≈ B9
- F53 (cw sweep) — H3-alt ≈ P1 at every cw

→ The MTL backbone reduces to two parallel per-task FFN+LN stacks. Paper can drop the "cross-attn mixing is productive" framing entirely.

## Phase 8 — Consolidation (2026-04-30 ~02:00)

F50 declared functionally complete:
- Mechanism story locked (temporal dynamics + reg encoder saturation + α growth gating).
- Champion locked (B9 / P4-alone / P0-A within 0.25 pp; recipe = P4 alone).
- Architectural decomposition locked (mixing dead via 3 independent receipts).
- Headline numbers locked (B9 = 63.47 ± 0.75 reg / 68.59 ± 0.79 cat; +3.34 pp Δ vs H3-alt p=0.0312 5/5 on both tasks).
- All C-series audit hygiene closed.

**Outstanding (deferred, not missing):**
- CA + TX P3 5f×50ep — A100-targeted (`scripts/run_p3_ca_unblock_attempt.sh` ready).
- Multi-seed variance — only if reviewer asks.

## Phase 9 — F51 deep exploration (2026-04-30 ~02:30 → 16:33)

Triggered by `F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT.md` (committed `130f2ee`). F51 ran two tiers:

### Tier 1: multi-seed validation (~3 h on 4090)

5 seeds × 2 arms × 5f×50ep at FL with seed-correct per-fold log_T. **Result: paper claim DECISIVELY ROBUST** (Δreg = +3.48 ± 0.12 pp across seeds, pooled p=2.98×10⁻⁸ on 25 fold-pairs, 25/25 positive folds; cat also paper-grade once pooled at p=1.33×10⁻⁵).

**Bug found and fixed mid-sweep (2026-04-30 05:25):** the original C4 fix wrote per-fold log_T as `region_transition_log_fold{N}.pt` with no seed in the filename — the script's CLI default was `--seed 42` and was never overridden. The trainer loaded this file unconditionally regardless of its own `--seed N`. At any seed != 42, ~80% of val users live in seed=42's fold-N TRAIN set → ~80% of val transitions leaked back into the prior. Empirical magnitude: B9 absolute reg inflated from clean ~63 to leaky ~72.5 at seeds {0, 1, 7, 100}. Caught by an env-B 1f×10ep verification smoke that didn't reproduce the seed=42 5f×50ep reference, then a 5f×50ep seed=42 rerun reproduced the handover ref bit-exactly — exposing the fold-split mismatch. Fix landed in `scripts/compute_region_transition.py` (filename now includes seed), `src/training/runners/mtl_cv.py` (reads seed-tagged file, hard-fails on legacy/missing), `scripts/run_f51_multiseed_fl.sh` (idempotent per-seed builds). Paired Δs survive (uniform-leak), absolute v1 numbers are wrong; v2 numbers are clean.

**Full doc:** `F51_MULTI_SEED_FINDINGS.md`.

### Tier 2: capacity-knob sweep (~4 h on 4090)

21 capacity smokes (5f×30ep, B9 base, FL) across 7 architecture dimensions: encoder_layer_size, num_encoder_layers, encoder_dropout, shared_layer_size, num_crossattn_blocks, num_crossattn_heads, crossattn_ffn_dim.

**Result: B9 is locally optimal in 5/7 dimensions; no paper-grade promotion candidate.** Single Pareto-trade (`num_crossattn_blocks=3`: Δreg +0.75 pp / Δcat -2.62 pp) refines F52's "mixing is dead at FL" claim to depth-conditional (alive at depth=3, breaks cat at depth=4). Three width-knobs catastrophically break cat without affecting reg (`shared_layer_size=384`: fold 2 collapse; `shared_layer_size=512`: 5/5 folds fail; `num_crossattn_blocks=4`: multi-fold; `crossattn_ffn_dim=1024`: multi-fold) — adds a NEW "cat width-stability cliff" mechanism: P4 alternating-SGD + higher per-head reg LR (3e-3) shields reg; cat at LR=1e-3 has no shield.

**Full doc:** `F51_TIER2_CAPACITY_FINDINGS.md`.

### Net F51 outcome
- Paper claim **strengthened** from single-seed +3.34 pp to multi-seed +3.48 ± 0.12 pp pooled p<10⁻⁷.
- Architecture-via-capacity-scaling track **closed** (no lift available).
- Two NEW mechanism receipts: (a) cat width-stability cliff, (b) cross-attn mixing depth-conditional.
- One critical bug (per-seed log_T leak) caught and fixed; trainer now hard-fails on legacy/missing files.

---

## Cross-references

- Current synthesis: `F50_T4_FINAL_SYNTHESIS.md`
- Headline numbers compiled: `F50_RESULTS_TABLE.md`
- Live tracker (rich log entries source): `F50_T4_PRIORITIZATION.md`
- **F51 multi-seed validation:** `F51_MULTI_SEED_FINDINGS.md`
- **F51 Tier 2 capacity sweep:** `F51_TIER2_CAPACITY_FINDINGS.md`
- C4 leak diagnosis (load-bearing receipt): `F50_T4_C4_LEAK_DIAGNOSIS.md`
- Broader leakage audit: `F50_T4_BROADER_LEAKAGE_AUDIT.md`
- Validity matrix (which prior runs survive C4): `F50_T4_PRIOR_RUNS_VALIDITY.md`
- Mechanism narrative source: `F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md`
- Latest F50 follow-ups: `F50_B2_F52_F65_F53_FINDINGS.md`
- D5 encoder receipt: `F50_D5_ENCODER_TRAJECTORY.md`
- Phase-3 fallback plan: `C05_P3_NULL_RESULT_FALLBACK.md`
- Archived sub-experiment docs: `archive/F50/` (T1.1, T1.5, T2/T3 sub-results, brainstorms, original plan)
