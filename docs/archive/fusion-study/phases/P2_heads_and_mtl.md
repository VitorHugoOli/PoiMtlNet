# Phase 2 — Heads + MTL vs Single-task

**Goal:** (a) find the best task heads in the MTL setting with the new champion backbone, and (b) finally answer whether MTL provides any benefit over single-task training on fusion inputs.

**Duration:** ~8-10 h (AL + AZ).

**Embedded claims:**
- **C06 — MTL beats single-task (the most critical claim of the entire paper)**
- C07 — MTL benefit is asymmetric per task
- C08 — standalone head rankings don't transfer to MTL
- C09 — head co-adaptation with backbone (mechanism)
- C10 — DCN exploits Sphere2Vec × HGI cross-features

---

## Methodology (updated 2026-04-18 from P1 critical review)

**Mandatory metric discipline — every C06/C07/C08/C10 comparison in this phase
must report BOTH `joint@J` and `joint@T`.** See claim C32 and
`docs/studies/fusion/issues/P1_METHODOLOGY_FLAWS.md` F1.

- `joint@J` = harmonic mean at the **joint-peak** checkpoint (deployment view).
- `joint@T` = harmonic mean computed from **per-task-best** epochs
  (`diagnostic_task_best` in `full_summary.json`).

In P1 these metrics disagreed: at joint@J the "AL champion" was
`mmoe4 × gradnorm`; at joint@T `cgc22 × equal_weight` led. Rankings are
unstable across the two policies when task-peak epochs differ significantly
(category peaks at 17–45; next peaks at 10–22 on AL fusion). For C06 this is
load-bearing: a single-task-next model naturally picks its own next-peak
checkpoint, so comparing to MTL's joint-peak artificially disadvantages MTL
on the next task. **Use joint@T as the scientific comparison metric; joint@J
as the deployment view; report both.**

**Champion from P1 (multi-seed, 3 seeds):**
- **AL:** `mmoe4 × gradnorm` (joint@J 0.4080 ± 0.0008, joint@T 0.4232 ± 0.0022).
  Chosen for stability — not max mean. `cgc22 × equal_weight` ties at joint@T
  but has 10× higher seed variance.
- **AZ:** `cgc21 × uncertainty_weighting` or `cgc21 × dwa` (tied at joint@T
  ≈ 0.44). Pick uw for headline continuity.

**Multi-seed discipline:** any C06/P2 champion comparison should include
≥ 2 seeds. A single-seed C06 conclusion is not publishable.

---

## Preconditions

- P1 completed with a clear champion (call it `arch* × optim*`).
- All embedding integrity checks pass.

---

## Experimental design

### Head candidates

**Category heads (9, subject to registry availability):**

| ID | Description | Hypothesis |
|----|-------------|------------|
| `category_default` (MTL) | 3-path MLP ensemble | Baseline |
| `category_dcn` | Deep & Cross Network | **C10** — cross-features between fusion halves |
| `category_residual` | MLP with residuals | Alternative strong isolated baseline |
| `category_gated` | Gated MLP | Lightweight gating alternative |
| `category_single` | Single MLP | Capacity control |
| `category_deep` | 5-layer MLP | Capacity upper bound |
| `category_linear` | `nn.Linear` | Probe — measures what the backbone learned |
| `category_attention` | Attention pooling | Alternative inductive bias |
| `category_transformer` | Transformer encoder | Tests the paradox (was default isolated worst) |

**Next heads (10, subject to registry availability):**

| ID | Description | Hypothesis |
|----|-------------|------------|
| `next_default` (MTL) | 4-layer Transformer + causal | Baseline |
| `next_tcn_residual` | TCN with residual blocks | Was isolated winner |
| `next_temporal_cnn` | Shallow temporal CNN | Local-pattern hypothesis |
| `next_single` | MLP over concat | Capacity control |
| `next_conv_attn` | CNN + attention | Hybrid |
| `next_transformer_relpos` | Transformer + relative pos | Sequences are short (9) |
| `next_gru` | Bi-GRU | Classic recurrent |
| `next_lstm` | LSTM | Classic recurrent |
| `next_mlp_pool` | MLP over pooling | Diagnostic (no seq model) |
| `next_linear_probe` | Linear over pooled embeddings | Absolute lower bound |

**If the registry is missing some:** either add them first (cheap) or drop the missing ones and record the reduction in P2's SUMMARY.

### Phases of P2

#### P2a — MTL head sweep (grid)

Fix the champion backbone (`arch* × optim*` from P1). Sweep heads.

**Two-phase design:** instead of 9 × 10 = 90 runs, do one-at-a-time sweeps:

- **P2a-cat:** vary category head, fix next head = `next_default`. 9 runs.
- **P2a-next:** vary next head, fix category head = `category_default`. 10 runs.
- **P2a-combo:** top-3 cat × top-3 next = 9 runs. Tests interactions.

Total: 28 runs. Screen at 1f × 10ep (~30 min per state), top-5 promoted to 5f × 50ep (~110 min per state).

#### P2b — Single-task baseline (the critical control)

With the best MTL config (fusion + arch* + optim* + best heads), train each task alone:

- **P2b-cat-alone:** `--task category --engine fusion ...` with best category head.
- **P2b-next-alone:** `--task next --engine fusion ...` with best next head.

Both at 5f × 50ep on AL and AZ.

**This is the single most important pair of runs in the entire study.**

#### P2c — MTL vs single-task comparison (C06)

**Compute under BOTH checkpoint policies** (see Methodology section; this is
C32-mandatory):
- MTL `joint@J` = HM(MTL cat@joint-peak, MTL next@joint-peak).
- MTL `joint@T` = HM(MTL cat@cat-peak, MTL next@next-peak).
- Single-task-cat `cat F1` = cat@cat-peak (single-task naturally uses task-peak).
- Single-task-next `next F1` = next@next-peak.
- Single-task `joint@T` = HM(single-cat-F1, single-next-F1) — this is the
  *deployment-fair* single-task joint (both heads at their own peak).
- Single-task `joint@J` is ill-defined for single-task (no joint peak).
  Report it as the same as `joint@T` or flag as N/A.

**The load-bearing comparison is MTL `joint@T` vs single-task `joint@T`.**
joint@J would artificially favor single-task because MTL's joint-peak
checkpoint under-reports next F1 by ~0.012–0.017 (P1 evidence).

**Decision rules (under joint@T):**
- If MTL joint@T > single-task joint@T by > 2 p.p. (paired t-test across folds, p < 0.05): **confirm C06**.
- If within 2 p.p.: **refute C06**, the paper needs reframing to be about fusion (not MTL).
- If MTL < single-task: MTL is actively harmful on this configuration — a publishable but awkward finding; investigate before paper.

**Also report per-task deltas (C07):**
- Δcat = MTL-cat@best-epoch − single-cat-F1
- Δnext = MTL-next@best-epoch − single-next-F1
- Even if joint C06 refutes, one direction of Δ can still make a C07 story.

**Minimal C06 run (prescribed for this session, 2026-04-18):**
- MTL: re-use `P1_AL_confirm_mmoe4_gradnorm_seed42` (already archived).
  joint@T = 0.4220, cat = 0.8295, next = 0.2830.
- Single-task-cat: new run `P2_AL_single_cat_seed42` — `mtlnet` base encoder,
  default category head, `--task category`, fusion, 5f × 50ep.
- Single-task-next: new run `P2_AL_single_next_seed42` — `mtlnet` base encoder,
  default next head, `--task next`, fusion, 5f × 50ep.
- Multi-seed extension (seeds 123, 2024) strongly recommended before
  publishing; skipped in the minimal-run set to preserve budget.

#### P2d — Head co-adaptation probe (C08, C09, optional)

For the best MTL config, after training, freeze the backbone and swap the heads:
- Train MTL normally; save backbone.
- Attach each alternative head to the frozen backbone; fine-tune only the head; measure F1.
- Compare rankings to (a) standalone isolated head rankings and (b) the MTL-end-to-end rankings.

**Outcome:**
- If heads that perform well in MTL are specifically good on MTL's backbone (not isolation): co-adaptation confirmed.
- If ranking is the same: no co-adaptation; some other effect explains the paradox.

This is a supplementary analysis — 20-30 extra runs. Do only if budget permits.

#### P2e — DCN on fusion vs on HGI (C10)

With best arch*, run `category_dcn` vs `category_default` on fusion and on HGI separately (4 runs × 2 states = 8 runs at 5f × 50ep).

Hypothesis: DCN's benefit is specifically larger on fusion (cross-features between halves) than on HGI-only (no structured halves).

---

## Test IDs

- `P2a_AL_cat_<head>_seed42` — 9 runs
- `P2a_AL_next_<head>_seed42` — 10 runs
- `P2a_AL_combo_<cat>_<next>_seed42` — 9 runs
- `P2b_AL_single_cat_seed42`, `P2b_AL_single_next_seed42`
- `P2c` is analysis, not a run
- `P2d_AL_<head>_frozen_seed42`
- `P2e_AL_dcn_fusion`, `P2e_AL_default_fusion`, `P2e_AL_dcn_hgi`, `P2e_AL_default_hgi`

All replicated on AZ.

---

## Compute budget

| Step | AL runs | AZ runs | Time |
|------|---------|---------|------|
| P2a screen (cat + next) | 19 | 19 | ~40 min / state |
| P2a promote top-5 | 5 | 5 | ~110 min / state |
| P2a combo top-9 | 9 | 9 | ~200 min / state (5f×50ep; might cut to top-3 combos = 70 min) |
| P2b single-task (2) | 2 | 2 | ~45 min / state |
| P2d probe (optional) | 10-20 | — | ~3-6 h AL only |
| P2e DCN×embedding | 4 | 4 | ~90 min / state |
| **Total** | ~40-60 | ~40 | **~8-10 h across both states** |

---

## Analysis steps

After P2a + P2b:

### Step A: Best heads in MTL

Rank P2a confirmation runs by joint F1.
- Best category head (in MTL): `cat*`
- Best next head (in MTL): `next*`
- Best combination: `cat*' × next*'` (may differ from component winners if interactions matter)

### Step B: MTL vs single-task (C06)

Given best config `arch* × optim* × cat* × next*`:
- MTL: its 5f × 50ep run (from P2a).
- Single-task: P2b runs.

Paired t-test across 5 folds. Output: p-value, mean delta, 95% CI.

### Step C: Per-task delta (C07)

For each task independently:
- delta_cat = MTL_cat_F1 − single_cat_F1
- delta_next = MTL_next_F1 − single_next_F1

Report both. If delta_cat >> delta_next (or vice versa), that's C07.

### Step D: Standalone vs MTL head rankings (C08)

For this, we need standalone-head-training data on fusion. If we don't have it, run it as part of P2d or skip C08 refinement.

### Step E: DCN specificity (C10)

P2e: (DCN − default) on fusion vs (DCN − default) on HGI. If the former is significantly larger than the latter, C10 confirmed.

---

## Surprises to watch for

| Symptom | Interpretation |
|---------|----------------|
| Best MTL head ≠ default | Old head-co-adaptation claim (C08) challenged — default isn't uniquely co-adapted |
| Single-task-cat ≈ MTL-cat but single-task-next > MTL-next | MTL hurts the harder task (next) via negative transfer |
| `category_linear` (probe) reaches > 60% F1 | The backbone does most of the work; choice of cat head barely matters — supports "growing shared backbone" claim (C16, P4) |
| DCN wins on HGI too | Contradicts C10's specificity claim; DCN is just a better head generally |

---

## Outputs

- `docs/studies/results/P2/` with per-test dirs
- `docs/studies/results/P2/SUMMARY.md` summarizing:
  - Best heads (cat, next, combo)
  - MTL-vs-single-task table with p-values
  - DCN specificity result
- Updated claim statuses: C06, C07, C08, C10 (C09 if P2d was run)
- state.json: P2 marked complete

---

## Phase gate for P3

Proceed to P3 if:
1. P2c answered C06 (confirmed, refuted, or partial).
2. Best heads identified (or defaults retained if alternatives don't help).
3. AL is fully done. AZ can lag.

**If C06 is refuted:** stop, call a meeting, re-plan the paper around "fusion+optimizer" rather than "MTL done right".
