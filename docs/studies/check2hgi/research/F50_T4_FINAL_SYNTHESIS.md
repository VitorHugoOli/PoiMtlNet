# F50 T4 — Final Synthesis (2026-04-30 — multi-seed strengthened)

**Status:** Paper story locked. F50 study complete except CA/TX A100-deferred. **F51 multi-seed validation (2026-04-30) strengthens the headline from single-seed +3.34 pp p=0.0312 to multi-seed +3.48 ± 0.12 pp pooled p < 10⁻⁷.**

**This doc is the canonical entry point.** Read this first; everything else is supporting.

Quick navigation:
- **Chronological history of how we got here:** `F50_HISTORY.md`
- **All headline numbers in one place:** `F50_RESULTS_TABLE.md`
- **F51 multi-seed validation:** `F51_MULTI_SEED_FINDINGS.md` ← **strengthens paper claim**
- **F51 Tier 2 capacity sweep:** `F51_TIER2_CAPACITY_FINDINGS.md` ← **closes architecture-via-capacity-scaling track**
- **C4 leak root cause (load-bearing receipt):** `F50_T4_C4_LEAK_DIAGNOSIS.md`
- **Broader leakage audit:** `F50_T4_BROADER_LEAKAGE_AUDIT.md`
- **Validity of prior F50 ablations under C4:** `F50_T4_PRIOR_RUNS_VALIDITY.md`
- **Mechanism narrative (temporal dynamics):** `F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md`
- **Mechanism receipts (encoder + α trajectories):** `F50_D5_ENCODER_TRAJECTORY.md` + `figs/f63_alpha_trajectory.png`
- **Latest follow-ups (B2/F52/F65/F53):** `F50_B2_F52_F65_F53_FINDINGS.md`
- **CA/TX P3 fallback plan:** `C05_P3_NULL_RESULT_FALLBACK.md`
- **Live tracker (rich update log):** `F50_T4_PRIORITIZATION.md`
- **Archived sub-experiment docs (T1.1, T1.5, T2/T3 sub-results):** `archive/F50/`

---

## 1 · Headline (paper-grade, leak-free, FL)

### 1.1 Multi-seed champion (F51, 2026-04-30) — **STRENGTHENED PAPER CLAIM**

B9 (P4 + Cosine + α-no-WD) vs H3-alt anchor across seeds {42, 0, 1, 7, 100}, FL 5f×50ep with **seed-correct per-fold log_T** (`region_transition_log_seed{S}_fold{N}.pt`):

| seed | B9 reg | H3-alt reg | Δreg | p_reg | n+/n | Δcat | p_cat | n+/n |
|---:|---:|---:|---:|:---:|:---:|---:|:---:|:---:|
| 42 | 63.47 ± 0.75 | 60.12 ± 1.15 | +3.34 | 0.0312 | 5/5 | +0.24 | 0.156 | 3/5 |
| 0 | 63.24 ± 0.89 | 59.58 ± 0.95 | +3.65 | 0.0312 | 5/5 | +0.61 | 0.062 | 4/5 |
| 1 | 63.41 ± 1.16 | 60.02 ± 1.03 | +3.39 | 0.0312 | 5/5 | +0.68 | 0.0312 | 5/5 |
| 7 | 63.21 ± 0.50 | 59.72 ± 0.54 | +3.49 | 0.0312 | 5/5 | +0.35 | 0.062 | 4/5 |
| 100 | 63.38 ± 0.93 | 59.87 ± 1.17 | +3.51 | 0.0312 | 5/5 | +0.22 | 0.156 | 3/5 |
| **mean ± σ across seeds** | **63.34 ± 0.11** | 59.86 ± 0.22 | **+3.48 ± 0.12** | — | — | **+0.42 ± 0.21** | — | — |

**Pooled paired Wilcoxon (5 seeds × 5 folds = 25 fold-pairs):**
- **Δreg = +3.48 pp, p = 2.98 × 10⁻⁸, 25/25 positive folds** ✅
- **Δcat = +0.42 pp, p = 1.33 × 10⁻⁵, 19/25 positive folds** ✅

**Headline claim (revised):** Across MTL POI prediction on FL with the check2HGI substrate, **alternating-optimizer-step (P4) yields a paper-grade reg lift of +3.48 ± 0.12 pp without sacrificing cat (5/5 seeds at p=0.0312 each; pooled paired Wilcoxon p=2.98×10⁻⁸ on 25 fold-pairs)**. Cat reaches paper-grade significance once seeds pool (+0.42 pp, p=1.33×10⁻⁵). The recipe is essentially deterministic in the partition-difficulty axis (B9 abs reg σ_across_seeds = 0.11 pp). STL→MTL gap is ~7.7 pp; closed by ~3.5 pp by the recipe.

### 1.2 Single-seed table (kept for cross-references) — seed=42 only

| recipe | reg top10 (≥ep5) | cat F1 (≥ep5) | Δreg vs H3-alt | paired Wilcoxon | verdict |
|---|---:|---:|---:|---|---|
| **STL F37 ceiling** (clean) | **71.12 ± 0.59** | n/a | — | — | upper bound |
| **B9 (P4+Cosine + α-no-WD)** ⭐ | **63.47 ± 0.75** | **68.59 ± 0.79** | **+3.34** | p=0.0312, 5/5 ✅ | **CHAMPION** (Pareto-dominant; multi-seed validated) |
| P4-alone (constant) | 63.41 ± 0.77 | 67.82 | +3.28 | p=0.0312, 5/5 ✅ | minimal-paper-grade |
| P0-A (P4+Cosine, no α-no-WD) | 63.23 ± 0.64 | 68.51 | +3.11 | p=0.0312, 5/5 ✅ | predecessor |
| F62 two-phase (cw=0→0.75) | 60.25 ± 1.26 | n/a | +0.13 | n.s. | **REJECTED** |
| PLE-lite (clean full 5×50) | 60.38 ± 0.79 | 64.13 ± 1.04 ⚠ | +0.26 | n.s. (cat **−4.22**) | Pareto-WORSE |
| H3-alt (clean baseline) | 60.12 ± 1.14 | 68.34 | 0 | — | anchor |

The seed=42 single-seed table above is unchanged from 2026-04-29; F51 multi-seed (§1.1) supersedes it as the paper-grade headline. P4-alone, P0-A, F62, PLE-lite, and H3-alt have NOT been multi-seed-validated; if a reviewer asks for cross-seed on those, F51's runner pattern (seed-correct per-fold log_T per seed) applies directly.

---

## 2 · The C4 leakage finding (paper § supplementary)

`region_transition_log.pt` was built from the **full** check-in dataset (train + val). The GETNext head reads it as `α · log_T` where α is a learnable scalar. Across training, **α grows from 0.1 → 1.8 (18×)**, so the prior dominates ranking near convergence. Every `next_getnext_hard*` number quoted before per-fold log_T was inflated by ~13–17 pp.

| measurement | value |
|---|---|
| Val transitions present in full-data log_T | 70.2% |
| Prior-only val top10 gap from log_T (fold 1) | 10.19 pp |
| α at init / convergence | 0.1 / 1.8 |
| Trained-model leak amplification | 13–17 pp absolute drop |
| Fix | `--per-fold-transition-dir` builds log_T from train-only rows |

**Uniform-leak hypothesis** (validated TWICE): the leak is approximately uniform across recipes that share the same head + log_T source. → Relative Δs are preserved. Validated by:
1. **B9 vs H3-alt clean = +3.34 pp** (matches expected direction, all 5 folds positive)
2. **PLE vs H3-alt clean Δreg = +0.26 pp** matches leaky +0.25 pp to within 0.01 pp ✅

→ The 17 other F50 ablations skipped re-running can be trusted to preserve their orderings (`F50_T4_PRIOR_RUNS_VALIDITY.md`).

**What survives:**
- All paired Δs comparing recipes with `next_getnext_hard` head + full log_T.
- F50 D1 (STL α=0) — 72.61 encoder-only ceiling is REAL (α=0 ⇒ no log_T contribution).
- HGI-vs-check2HGI substrate study (no learnable graph-prior amplifier).
- All baselines (HAVANA, POI-RGNN) — independent encoders, no shared leak.

**What gets restated:**
- "STL ceiling = 82.44" → 71.12 (clean F37 measured) or stronger.
- "Champion = 76.07" → 63.47 (clean B9).
- "FL has 8.83 pp STL-MTL gap" → ~7.7 pp clean.
- "10 architectural alternatives ≈ 74" → ~60 clean.

---

## 3 · New paper-grade findings (from this round)

### 3.1 PLE Pareto-WORSE under leak-free (NEW)

PLE-lite was originally tagged "PLE +0.25 pp ≈ H3-alt" under leaky log_T. Clean full 5f×50ep:
- Δreg = +0.26 pp (matches leaky to 0.01 pp — uniform-leak validated)
- **Δcat = −4.22 pp**, 0/5 positive folds (NEW under leak-free)

→ PLE's expert routing **hurts cat without helping reg** under clean conditions. PLE is **Pareto-WORSE than H3-alt** — strictly dominated by P4-alone, B9, and even P0-A. Not a usable architectural alternative.

### 3.2 P4-alone matches B9 within 0.06 pp

Under leak-free conditions:
- B9 = 63.47 reg, P4 = 63.41 reg, P0-A = 63.23 reg
- All three within ~0.25 pp of each other.

→ **The intervention is P4 (alternating optimizer step). Cosine and α-no-WD give marginal lift.** Paper recipe simplifies to: take the H3-alt baseline + add `--alternating-optimizer-step`.

The B9 claim of "Pareto-dominant on BOTH tasks" still holds (cat +0.08 pp), but the absolute numbers are incremental. The simplification is the paper-friendly story.

### 3.3 F62 two-phase REJECTED

`scheduled_static` cw=0→0.75 step at ep=10: reg = 60.25 ± 1.26, cat n.s. drop.

→ Two-phase coarse-grained scheduling does NOT replicate P4's per-batch alternating granularity. **Rejected as champion candidate.** P4's per-step granularity is essential to the mechanism.

### 3.4 D5 — reg encoder saturates 26–32 epochs before cat encoder (NEW)

Per-epoch Frobenius drift logging on FL fold 1 (H3-alt baseline + B9 champion, leak-free):

| run | reg sat ep | cat sat ep | gap | reg-best ep | encoder sat aligns? |
|---|---:|---:|---:|---:|---|
| H3-alt | 24 | 50 (never) | 26 | 3 | reg encoder wastes ~21 ep after val plateau |
| B9 | **6** | 38 | 32 | **6** | **encoder saturation = reg-best epoch (tight)** |

→ **Mechanistic receipt #2** for the temporal-dynamics narrative: the reg encoder physically stops updating in the same window where reg val plateaus. P4 doesn't fix the saturation timing — it instead lets head/α keep growing AFTER the encoder is done (see F63 α-trajectory). Full doc: `F50_D5_ENCODER_TRAJECTORY.md`. Plot: `figs/f50_d5_encoder_trajectory.png`.

### 3.5 Cross-state portability (clean from start)

| state | regions | reg @≥ep10 (per-fold log_T) | gap to FL |
|---|---:|---:|---:|
| FL | 4703 | 60.36 | 0 |
| GA | 2283 | 46.57 | −13.79 |
| AL | 1109 | 49.44 | −10.92 |
| AZ | 1547 | 40.61 | −19.75 |

Recipe doesn't transfer cleanly to small states. Best-epoch distributions differ (AL fold-best epochs {45,38,44,31,35} suggest instability). **Paper claim becomes "FL-strong; cross-state directional but not paper-grade."**

### 3.6 F51 Tier 2 — capacity sweep closes the architecture-via-scaling track (2026-04-30, NEW)

21 capacity smokes (5f×30ep, B9 base, FL) across 7 capacity dimensions:

| dimension | levels | result |
|---|---|---|
| `encoder_layer_size` | {128, 256(B9), 384, 512} | tied (Δreg ∈ [-0.41, +0.0] pp) |
| `num_encoder_layers` | {1, 2(B9), 3, 4} | tied at {1,3}, **regression at 4** (-0.60 reg) |
| `encoder_dropout` | {0.05, 0.1(B9), 0.2, 0.3} | tied on reg; cat degrades monotonically with dropout |
| `shared_layer_size` | {128, 256(B9), 384, 512} | 128 tied; **384/512 catastrophically break cat** without affecting reg |
| `num_crossattn_blocks` | {1, 2(B9), 3, 4} | 1 tied; **3 = Pareto-trade** (+0.75 reg / -2.62 cat); **4 collapses cat** |
| `num_crossattn_heads` | {2, 4(B9), 8, 16} | tied across all (Δreg ∈ [-0.28, -0.13]) |
| `crossattn_ffn_dim` | {128, 256(B9), 512, 1024} | {128, 512} tied; **1024 regresses + collapses cat** |

**Three NEW findings:**

1. **B9 is locally optimal in 5/7 capacity dimensions** — no architectural lift available via capacity scaling. Closes the architecture-axis exploration started in F50 T1.5.

2. **F52's "mixing is dead at FL" is depth-conditional.** F52 P5 (identity-mixing at 2 blocks) tied B9; this Tier 2 result shows 3 blocks with mixing on yields Δreg = +0.75 pp (but cat -2.62 pp). Cross-attn mixing has a real, small contribution that B9's 2-block default deliberately suppresses for cat stability. Refined paper claim: "cross-attn mixing has a sharp Pareto cliff at B9's 2-block depth."

3. **NEW mechanism — cat width-stability cliff.** Three width-knobs (`shared_layer_size=384` and 512, `num_crossattn_blocks=4`, `crossattn_ffn_dim=1024`) catastrophically break cat training without affecting reg. P4 alternating-SGD + higher per-head reg LR (3e-3) shields reg; cat at LR=1e-3 has no shield and falls off the optimum when the shared backbone widens. Adds a third Pareto-worse direction (alongside PLE expert routing and F62 two-phase) anchored on a different mechanism (capacity stability vs. expert routing vs. temporal scheduling).

**No paper-grade promotion** from Tier 2. The single PROMOTE candidate (`num_crossattn_blocks=3`) is Pareto-trade (cat -2.62 pp), same disposition as P4+OneCycle.

Full doc: `F51_TIER2_CAPACITY_FINDINGS.md`. Structured JSON: `F51_tier2_results.json`. Sweep runner: `scripts/run_f51_tier2_capacity_smoke.sh`. Analyzer: `scripts/analysis/f51_tier2_analysis.py`.

### 3.7 F51 — per-seed log_T leak found and fixed mid-sweep (2026-04-30)

The original C4 fix (`scripts/compute_region_transition.py --per-fold`) wrote per-fold log_T as `region_transition_log_fold{N}.pt` with NO seed in the filename — the script's CLI default was `--seed 42` and was never overridden. The trainer (`src/training/runners/mtl_cv.py`) loaded this file unconditionally regardless of its own `--seed N` argument. At any seed != 42, ~80% of val users live in seed=42's fold-N TRAIN set → ~80% of val transitions leaked back into the prior. Empirical magnitude: B9 absolute reg inflated from clean ~63 to leaky ~72.5 at seeds {0, 1, 7, 100}.

**Fix landed (3 files):**
- `scripts/compute_region_transition.py` writes `region_transition_log_seed{S}_fold{N}.pt`.
- `src/training/runners/mtl_cv.py` reads the seed-tagged file using the trainer's own `--seed` and **hard-fails** with explanatory `FileNotFoundError` if the seed-tagged file is missing OR if a legacy unseeded file is present (preventing silent reuse).
- `scripts/run_f51_multiseed_fl.sh` builds per-seed log_T idempotently before each seed's runs.

**Paired-Δ findings survive** (uniform-leak property: both arms read the same wrong prior on the same val set, so paired difference cancels most of the leak — clean and leaky Δs match within 0.10 pp at every seed). But absolute numbers from the v1 multi-seed sweep were wrong; v2 (clean) numbers are in `F51_MULTI_SEED_FINDINGS.md`.

---

## 4 · Re-run priority for paper

All TIER 0 paper-blocking runs are **DONE**. Tier 1/2 confirmations done too.

| done | run | numbers |
|---|---|---|
| ✅ | H3-alt clean FL | 60.12 ± 1.14 |
| ✅ | P0-A (P4+Cosine clean FL) | 63.23 ± 0.64 |
| ✅ | B9 clean FL | 63.47 ± 0.75 |
| ✅ | P4-alone clean FL | 63.41 ± 0.77 |
| ✅ | P4+OneCycle clean FL | (Pareto-trade preserved) |
| ✅ | STL F37 clean FL | 71.12 ± 0.59 |
| ✅ | F62 two-phase clean FL | 60.25 ± 1.26 (REJECTED) |
| ✅ | Cross-state AL/AZ/GA (clean from start) | see §3.4 |
| ✅ | TGSTAN clean smoke | confirms uniform leak |
| ✅ | PLE-lite clean full | 60.38 ± 0.79 (Pareto-worse) |
| ✅ | **F51 multi-seed (5 seeds)** | **Δreg = +3.48 ± 0.12 pp; pooled p=2.98×10⁻⁸** |
| ✅ | **F51 Tier 2 capacity sweep** | B9 locally optimal; no paper-grade lift |

→ **No more paper-blocking runs needed.** All headline numbers measured. F51 strengthens the headline; Tier 2 closes the architecture-via-scaling track.

---

## 5 · Paper-grade survival summary

| claim | leak-corrected status |
|---|---|
| "STL→MTL gap closed by paper-grade Δ" | ✅ +3.34 pp B9 vs H3-alt seed=42; **strengthened by F51 to +3.48 ± 0.12 pp across 5 seeds, pooled p=2.98×10⁻⁸** |
| "MTL reg-best is structurally pinned at ep 4–5" | ✅ epoch trajectory preserved (F63 confirms; F51 Tier 2 confirms across 21 capacity perturbations) |
| "10 architectural alternatives all give reg ≈ baseline" | ✅ relative observation; absolutes ~13 pp lower |
| "Architecture has no capacity-scaling lift" | ✅ NEW (F51 Tier 2): B9 locally optimal in 5/7 dimensions |
| "D8 cw=0 → reg-best ep 5 across all folds" | ✅ trajectory preserved |
| "P4 alternating-SGD wins by paired Wilcoxon p=0.0312" | ✅ both arms leaky → uniform leak preserves Δ; F51 multi-seed: 5/5 seeds at p=0.0312 |
| "B9 alpha-no-WD is Pareto-dominant +0.24/+0.08" | ✅ measured leak-free; cat reaches paper-grade once seeds pool (F51: +0.42 pp, p=1.33×10⁻⁵) |
| "Cross-attn mixing is structurally dead at FL" | ⚠ **refined by F51 Tier 2 to depth-conditional**: dead at depth=2 (B9), small contribution at depth=3 (Pareto-trade), breaks cat at depth=4 |
| "Cat width-stability cliff" | ✅ NEW (F51 Tier 2): wider shared backbone breaks cat without affecting reg (4 cases: shared 384/512, blocks=4, ffn=1024) |
| "P4+Cosine champion = 76.07 reg" | ❌ → 63.47 (B9 seed=42) → 63.34 ± 0.11 (5-seed mean) |
| "STL ceiling = 82.44 reg" | ❌ → 71.12 (F37 clean seed=42) |
| F49 architectural decomposition (AL/AZ/FL gaps) | ✅ relatively (uniform leak); absolutes inflated |

**11/13 paper claims survive (post-F51), 2 refined, 2 absolute headlines restated.** The mechanism narrative — temporal training dynamics, P4 per-step alternation, FiLM/cross-attn architecture — is preserved and strengthened by F51.

---

## 6 · Remaining work (re-prioritized)

After consolidation, four pending tasks remain. Re-prioritized given the paper story is locked:

| priority | task | effort | rationale |
|---|---|---|---|
| **P0** (do now) | **#70 C7 — stamp aggregation_basis** | ~30 min | Hygiene; finishes the audit C-series cleanly. Low risk. Tracker shows 30 min; the per_metric_best subdict already lands in fold JSON, just needs basis stamp on aggregates. |
| **P1** (do next) | **#39 D5 — reg encoder weight-trajectory diagnostic** | ~80 min | Paper-figure-only diagnostic. Useful for the temporal-dynamics narrative section. Generates a NEW figure showing reg encoder weight drift across epochs. |
| **P2** (defer) | **#66 P1-B B2 warmup-decay reg_head_lr** | ~1h dev + 19 min run | Paper story already locked at +3.3 pp; B2 was a "stack on top of B9" proposal. With B9 ≈ P4-alone within 0.06 pp, the lift ceiling is constrained. **Defer unless reviewer asks.** |
| **P3** (close) | **#49 F65 dataloader cycling** | — | D8 (cw=0) already refuted the cat-loss-dominance hypothesis. Mark as **deferred-pending-reviewer**. |

**Decision:** start with #70 (C7), then #39 (D5) for the paper figure. Park #66 and #49.

---

## 7 · One-paragraph summary

The C4 leakage (full-data graph prior leaking val transitions into a learnable α-amplified head) inflated every `next_getnext_hard*` number in the F50 study by 13–17 pp at convergence. The fix (per-fold log_T) drops the champion 76.07 → 63.47. **The paper headline survives and STRENGTHENS under F51 multi-seed validation (2026-04-30)**: B9 (P4 + Cosine + α-no-WD) Pareto-dominates H3-alt with **Δreg = +3.48 ± 0.12 pp across 5 seeds** (5/5 seeds at p=0.0312 each; pooled paired Wilcoxon **p=2.98×10⁻⁸ on 25 fold-pairs, 25/25 positive**). Cat reaches paper-grade significance once seeds pool (Δcat=+0.42 pp, p=1.33×10⁻⁵). F51 also caught and fixed a follow-on per-seed log_T leak (the original C4 fix wasn't seed-keyed in the filename, so trainers at non-default seeds silently leaked ~80% of val transitions back into the prior). Updated invariant: per-fold log_T filename is now `region_transition_log_seed{S}_fold{N}.pt`, trainer hard-fails if missing or if a legacy unseeded file is present. The paper recipe simplifies to: H3-alt baseline + `--alternating-optimizer-step`. PLE-lite is Pareto-WORSE under clean conditions (cat −4.22 pp) — added as a NEW paper finding. F62 two-phase REJECTED. Cross-state portability is FL-strong, AL/AZ/GA directional. All TIER 0 leak-free runs done; no more paper-blocking runs needed.
