# Check2HGI Study — Master Plan

**Goal:** Validate that next-region as an auxiliary task improves next-category prediction on Check2HGI check-in-level embeddings, with a champion MTL configuration found through systematic ablation.

## Phase overview

| Phase | What | Claims | Duration | Requires |
|---|---|---|---|---|
| **P0** | ✅ Integrity + simple baselines + audits | CH04 floor | done | — |
| **P1** | Region-head validation + head ablation (single-task) | CH04, CH05 | ~2h | P0 |
| **P2** | TaskSet-parameterise all MTL archs + full arch × optim ablation | CH06 | ~1 day | P1 (head winner) |
| **P3** | MTL headline: champion config, multi-seed n=15 | **CH01**, CH02, CH07 | ~4h | P2 (champion) |
| **P4** | Dual-stream: region_embedding as parallel input | CH03, CH08 | ~3h | P3 |
| **P5** | Cross-attention (gated on P4 ≥ 2pp on FL) | CH09 | ~6h | P4 |

**Total:** ~16h sequential (excluding P5 if gated out). P2 is the heaviest phase.

## Execution order rationale

**P1 (heads) before P2 (MTL ablation):**
- The region head is **untested** — it's the next-category head repurposed for ~1K classes. We need to know it works before layering MTL.
- The head winner feeds into the MTL ablation as a **fixed choice**. Wrong head → wrong MTL conclusions.
- HMT-GRN uses a GRU for their region head (not a transformer). If GRU wins in P1, that changes which MTL backbone benefits most in P2.
- P1 is cheaper (~5 runs) than P2 (~15+ runs).

**P2 (MTL ablation) before P3 (headline):**
- The headline must use the champion (arch, optim) pair, not an arbitrary default.
- Running the headline first with NashMTL+base-MTLnet and then discovering CGC+equal_weight is better would waste the multi-seed n=15 budget.

## Phase details

### P1 — Region-head validation + head ablation

**Single-task next_region on Check2HGI, Alabama only.**

5 head variants, screen at 1-fold × 10-epoch, top-2 confirmed at 5-fold × 50-epoch:

| Head | Architecture | Why |
|---|---|---|
| `next_mtl` | 4-layer transformer + causal + attn pool | Default (next-category champion) |
| `next_gru` | Bi-GRU | HMT-GRN's region-head approach |
| `next_lstm` | LSTM | Classic recurrent |
| `next_tcn_residual` | TCN + residual | Was standalone next-category winner in prior fusion work |
| `next_temporal_cnn` | Shallow temporal CNN | Simple CNN baseline |

**Also run:** single-task next_category with `next_mtl` (reference for P3 pairing — already done: 38.67% F1 on AL).

**Gate:** at least one head achieves region Acc@10 ≥ 2× Markov floor (21.3% → ≥ 42.6%). If none do, investigate before P2.

### P2 — Full MTL architecture × optimiser ablation

**Pre-requisite (code work, ~half day):** Parameterise `MTLnetCGC`, `MTLnetMMoE`, `MTLnetDSelectK`, `MTLnetPLE` with `TaskSet` — same pattern as base `MTLnet`. All hardcode `category_x, next_x` in their `_mix()` internals and need the slot rename.

**Also:** research whether alternative MTL architectures better suit {next_category, next_region} (both tasks are sequential from the same input, unlike fusion's flat+sequential pair). Consider:
- Asymmetric sharing (region → category but not vice versa)
- Cross-stitch / sluice networks (learn which layers to share per task)
- Simpler approaches (FiLM may suffice since both tasks share input structure)

**Ablation grid:**

5 architectures × all available optimisers (NashMTL, equal_weight, CAGrad, Aligned-MTL, PCGrad, DWA, FAMO, etc.):

| Stage | Runs | Config | Time/run | Total |
|---|---|---|---|---|
| Screen | 5 × N_optim | 1-fold × 10-epoch, AL | ~1 min | ~1h |
| Promote top-10 | 10 | 2-fold × 15-epoch, AL | ~3 min | ~30 min |
| Confirm top-5 | 5 | 5-fold × 50-epoch, AL | ~22 min | ~2h |
| FL replication of top-2 | 2 | 5-fold × 50-epoch, FL | ~80 min | ~3h |

**Head config:** region head = P1 winner (slot B); category head = `next_mtl` (slot A, proven).

**Gate:** champion identified with sensible metrics (next-category F1 > 30%, next-region Acc@10 > 15% on AL).

### P3 — MTL headline (multi-seed n=15)

Champion arch + optim + heads from P2.

| Run | State | Seeds | Folds | Purpose |
|---|---|---|---|---|
| P3.1.AL | AL | {42, 123, 2024} | 5 each | CH01 + CH02 |
| P3.1.FL | FL | {42, 123, 2024} | 5 each | CH01 + CH02 replication |

Compare against P1 single-task references (same folds, same seeds).

FL runs with `--use-class-weights` to mitigate 22.5% majority next-region class.

### P4 — Dual-stream region input

Feed `[B, 9, 128]` (check-in ⊕ region embedding per timestep) instead of `[B, 9, 64]`.

Same champion config as P3. Multi-seed on AL + FL.

### P5 — Cross-attention (gated on P4)

Only runs if P4 shows ≥ 2pp next-category F1 lift on FL. New `MTLnetCrossAttn` architecture with bidirectional cross-attention between check-in and region streams.

## Datasets

| State | Check-ins | POIs | Regions | Sequence rows |
|---|---|---|---|---|
| Alabama (primary) | 113,846 | 11,848 | 1,109 | 12,709 |
| Florida (replication) | 1,407,034 | 76,544 | 4,703 | 159,175 |
| Arizona (triangulation) | ~120K | ~10K | 1,547 | 26,396 |

## Exit criteria

Branch merges when:
1. CH01 + CH02 resolved with evidence (P3).
2. CH06 resolved (P2 champion identified).
3. CH03 resolved (P4 dual-stream tested).
4. P1 head ablation documented.
5. All legacy tests green.
6. Paper findings section drafted.
