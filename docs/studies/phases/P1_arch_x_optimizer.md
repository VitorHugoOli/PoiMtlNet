# Phase 1 — Architecture × Optimizer grid

**Goal:** find the best (MTL architecture, optimizer) pair on fusion inputs. Simultaneously validate/refute claims C01-C05.

**Duration:** ~4-8 h (AL + AZ screening; FL deferred to P3).

**Embedded claims tested:**
- C02 — gradient-surgery improves over equal weighting on fusion (primary)
- C03 — equal weighting suffices on single-source (replicates Xin et al.) — partial, extended in P3
- C04 — architecture rankings depend on embedding — partial, extended in P3
- C05 — expert-gating > hard parameter sharing (FiLM)

---

## Experimental design

### Factors

**Architectures (5):**
| Short | Model | Config |
|-------|-------|--------|
| `base` | mtlnet | FiLM + shared layers (no gating) |
| `cgc22` | mtlnet_cgc | num_shared_experts=2, num_task_experts=2 |
| `cgc21` | mtlnet_cgc | num_shared_experts=2, num_task_experts=1 |
| `mmoe4` | mtlnet_mmoe | num_experts=4 |
| `dsk42` | mtlnet_dselectk | num_experts=4, num_selectors=2, temperature=0.5 |

**Optimizers (aim for 20; actual count depends on registry availability):**

| Category | IDs | Notes |
|----------|-----|-------|
| **Static** | `equal_weight`, `static_weight`, `uncertainty_weighting` | Baselines |
| **Loss-based dynamic** | `dwa`, `famo`, `bayesagg_mtl`, `excess_mtl`, `nash_mtl`, `stch` | Adapt weights from loss trajectory |
| **Gradient-based** | `pcgrad`, `gradnorm`, `cagrad`, `aligned_mtl`, `db_mtl`, `fairgrad`, `graddrop`, `mgda`, `moco`, `sdmgrad`, `imtl_h`, `imtl_l` | Operate on gradients directly |

Before starting P1, **audit the loss registry** (`src/losses/registry.py`) and commit the final list of optimizers to the state.json for this phase. Target 18-20.

### Engine

**Fusion only for the main grid.** DGI and HGI comparisons happen in P3.

**Rationale:** the 5×20 grid on fusion is where the main finding lives. P3 contextualizes by running the best subset on DGI and HGI too.

### States

- **Alabama (AL)** — primary.
- **Arizona (AZ)** — replication; we trust a finding more if it holds in both.

### Budget

| Stage | Runs per state | Config | Time per run | Total per state |
|-------|---------------|--------|--------------|-----------------|
| **P1a — screen** | 5 × 20 = 100 | 1 fold, 10 epochs | ~1 min | ~100 min |
| **P1b — promote top-10** | 10 | 2 folds, 15 epochs | ~3 min | ~30 min |
| **P1c — confirm top-5** | 5 | 5 folds, 50 epochs | ~22 min | ~110 min |
| **Total** | ~115 runs | | | **~4 h per state** |

AL + AZ ≈ 8 h sequential; can run AL while AZ runs on another machine.

### Seed

Seed 42 for the screen and promotion. Multi-seed (seeds 123, 2024) only for the top-3 in P1c confirmation.

### Effective batch size: normalization decision

**Decision: run all candidates at `gradient_accumulation_steps=1, batch=4096`** — matched across all optimizers. This fixes the confound that compromised the prior study. (Some loss variants — cagrad, aligned_mtl, pcgrad — require grad_accum=1 anyway; others run fine at matched settings.)

**Impact on prior results:** the Stage 1 "25% gap" in the old study was partly this confound. Expect a smaller gap this time — this is the scientific value of the re-run.

---

## Test IDs and claim assignment

Each test gets a canonical ID: `P1_<state>_<stage>_<arch>_<optim>_<seed>`.

Examples:
- `P1_AL_screen_dsk42_al_seed42` — DSelectK + aligned_mtl, screen on Alabama
- `P1_AZ_confirm_cgc22_eq_seed42` — CGC22 + equal_weight, confirmation on Arizona

### Claim → test mapping

Every screen run contributes to all four claims (C02, C03-partial, C04-partial, C05) because it's a single cell in the grid. The full 5×20 grid is a shared test for all embedded claims.

---

## Execution order

### Step 1 — P1a screen AL

Run 100 candidates sequentially on Alabama, each at 1f × 10ep.

**Existing infrastructure:** use `experiments/full_fusion_ablation.py` (or extend it) to iterate through the grid. Extend the candidate list to 20 optimizers (currently 5).

### Step 2 — Analyze P1a results

After all 100 runs complete:
- Produce a 5×20 heatmap of joint scores.
- Identify top-10 candidates for promotion.
- Check structure: are all gradient-surgery optimizers clustered at the top? Or is the pattern different from the old study?

**Decision gate:**
- If top-10 includes multiple optimizer classes (mix of ca/al/eq/db): C02 is being refuted — equal_weight competitive. Flag for review.
- If top-10 is dominated by one class: that class is the clear winner; proceed.

### Step 3 — P1a screen AZ

Same 100 candidates on Arizona.

### Step 4 — Cross-state comparison

For each architecture × optimizer cell, compute AL vs AZ correlation. If they disagree on the top-10 substantially, investigate.

### Step 5 — P1b promote top-10 (AL, then AZ)

Top-10 from AL's P1a → run at 2f × 15ep.

### Step 6 — P1c confirm top-5 (AL + AZ)

Top-5 from P1b → run at 5f × 50ep.

### Step 7 — Multi-seed top-3

Top-3 overall → seeds 123, 2024 at 5f × 50ep on AL (maybe AZ too if time permits).

---

## Embedded claim analyses

### C02 (gradient-surgery on fusion)

After P1c, compute:
- Best joint of any `ca` or `al` configuration
- Best joint of any `eq` / `uw` / `db_mtl` configuration
- Both must be at matched batch (4096).

**Outcome:**
- If ca/al best > eq best by > 2 p.p. (more than seed variance): **confirm C02**.
- If within 2 p.p.: **partially confirm** — gradient surgery helps a bit but doesn't dominate.
- If eq/uw best ≥ ca/al best: **refute C02**. Reframe paper.

### C05 (expert-gating > FiLM)

After P1c, compute:
- Mean joint for `base` configs across all optimizers
- Mean joint for each of `cgc22`, `cgc21`, `mmoe4`, `dsk42` across all optimizers

**Outcome:**
- Expect all expert-gating archs > base by ~0.05 joint (from prior data)
- If any expert-gating ≈ base: investigate

### C04 (embedding-dependent architecture rankings)

Partial — we only have fusion in P1. The full claim needs P3 data.

After P1, record the winning architecture on fusion. In P3, compare to winners on DGI and HGI.

---

## Surprises we should watch for

| Symptom | Possible interpretation |
|---------|------------------------|
| `uncertainty_weighting` enters top-5 | Xin et al. "adaptive doesn't help" may not replicate on fusion |
| `db_mtl` beats `cagrad` | DB-MTL's EMA mechanism may handle scale imbalance differently than surgery |
| `base` + ca/al beats expert-gating + eq | Optimizer effect > architecture effect on fusion |
| AL winner ≠ AZ winner | dataset-dependent; single-state conclusions would be fragile |
| No optimizer improves over Stage 1 prior numbers | Label bug cleanup may have regressed performance; investigate |

Any of these should be flagged `surprising` in state.json and require explicit handling before proceeding to P2.

---

## Outputs

- `docs/studies/results/P1/` populated with ~230 test directories (115 × 2 states)
- Summary heatmap (saved as `docs/studies/results/P1/heatmap_AL.png`, `_AZ.png`)
- Claim statuses in `CLAIMS_AND_HYPOTHESES.md` updated for C02, C03-partial, C04-partial, C05
- One-page markdown summary at `docs/studies/results/P1/SUMMARY.md` — the champion config and its justification
- state.json: phase status `completed`, current_phase advanced to P2

---

## Phase gate for P2

Proceed to P2 only if:
1. P1c has a single clear winner (top joint) or a tie of 2-3 within noise.
2. The winner has a sensible profile (category F1 > 60%, next F1 > 20%) — i.e., it's not degenerate.
3. At least AL is fully done; AZ confirmation is nice-to-have but can lag by 1 day.

If the winner involves `equal_weight` on fusion → big news, paper direction changes; pause and re-plan before P2.
