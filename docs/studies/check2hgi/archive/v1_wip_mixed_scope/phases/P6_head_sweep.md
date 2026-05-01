# Phase P6 — Head sweep + MTL vs single-task (ported from legacy P2)

**Gates:** P5 complete (champion `arch* × optim*`).
**Purpose:** Given the P5 champion backbone + optimiser, find the best per-head choice and answer the **critical question: does MTL actually help vs single-task** on check2HGI.

## Key difference from legacy P2

Legacy P2 had two *different kinds* of heads: `category_*` (flat input) and `next_*` (sequential input). On the check2HGI track **both heads are sequential** (both consume the 9-window of check-in embeddings), so both pull from the same head pool. The head sweep is 2D across the same head family instead of 9-cat × 10-next.

## Head candidates (both heads pick from this list)

| ID | Description | Compatible? | Notes |
|----|-------------|-------------|-------|
| `next_mtl` (default) | 4-layer Transformer + causal mask + attention pool | ✓ | Current default |
| `next_transformer_relpos` | Transformer + relative positional | ✓ | Seq is short (9), may help |
| `next_tcn_residual` | TCN with residual blocks | ✓ | Legacy single-task winner on next-category |
| `next_temporal_cnn` | Shallow temporal CNN | ✓ | Local-pattern hypothesis |
| `next_lstm` | LSTM | ✓ | Classic recurrent |
| `next_gru` | Bi-GRU | ✓ | Classic recurrent |
| `next_conv_attn` | CNN + attention hybrid | ✓ | |
| `next_hybrid` | Hybrid arch | ✓ | |
| `next_single` | MLP over sequence concat | ✓ | Capacity control |
| `next_transformer_optimized` | Optimised transformer | ✓ | If registered |

≈ 10 heads total. Confirm registry coverage in P6 kickoff.

## Phases

### P6a — head sweep (efficient two-axis design)

28 runs per state, matching legacy P2 budget:

1. **P6a-A:** fix `task_b` head at `next_mtl`, vary `task_a` head across all 10 → 10 runs.
2. **P6a-B:** fix `task_a` head at `next_mtl`, vary `task_b` head across all 10 → 10 runs.
3. **P6a-combo:** top-3 from A × top-3 from B → 9 runs (interaction test).

Each at 1f × 10ep screening, top-5 combinations promoted to 5f × 50ep.

### P6b — single-task baselines (the critical control)

After the MTL head champion is found, rerun each head **alone**:

- **P6b-cat-alone:** `--task next_category --engine check2hgi` with best task_a head. 5f × 50ep.
- **P6b-region-alone:** `--task next_region --engine check2hgi` with best task_b head. 5f × 50ep.

**Requires:** adding `TaskType.NEXT_REGION` + `_create_single_task_next_region_folds` (see `phases/P0_inputs.md`).

### P6c — MTL vs single-task (the paper's headline for CH02)

Compute on both states:

- `MTL_joint_acc1 = 0.5 * (MTL_next_cat_acc1 + MTL_next_region_acc1)`
- `single_joint_acc1 = 0.5 * (single_cat_acc1 + single_region_acc1)`

**Paired t-test** across folds, alpha=0.05.

- If MTL > single by > 2 p.p. (paired, p < 0.05): **confirm CH02**.
- If within 2 p.p.: **partial** — MTL doesn't hurt but doesn't help. Paper reframes to emphasise check2HGI embedding as the contribution.
- If MTL < single: **refute CH02** — MTL is actively harmful on this configuration. Investigate before paper (likely cause: class-imbalance starvation of next_category under NashMTL on FL).

### P6d — per-head co-adaptation probe (optional)

For the MTL champion, after training, freeze backbone, swap heads, fine-tune only the head. Compares:

- Standalone head ranking (from P6b)
- MTL end-to-end ranking (from P6a)
- Frozen-backbone-swap ranking (this phase)

If the three rankings diverge substantially: **confirms CH19** (new claim — heads in MTL are co-adapted to the backbone, not to the task).

## Claims touched

Matches legacy P2 but renumbered:

- **CH02** (existing headline) — MTL beats single-task on check2HGI joint_acc1 (or F1 if chosen).
- **CH03** (existing negative-transfer control).
- **CH17** (new) — head rankings in MTL are not a simple function of standalone rankings.
- **CH18** (new, check2HGI mirror of legacy C09) — per-head fine-tuning on a frozen MTL backbone matches MTL-end-to-end rankings more closely than standalone rankings.
- **CH19** (new, from P6d) — co-adaptation mechanism.

## Decision gate → P7

Only proceed to P7 if P6c confirms CH02. If MTL doesn't help on vanilla check2HGI, adding dual-stream region input (P7) won't fix that — the paper pivots to a different thesis.

## Outputs

- `docs/studies/check2hgi/results/P6/` — 28 screen + 10 confirmation + 10 single-task = ~50 runs × 2 states
- `results/P6/SUMMARY.md` — champion head pair + MTL-vs-single verdict
- `results/P6/pareto_heads.png` — per-head plot of F1_next_cat vs Acc@1_next_region

## Budget

~4h per state (similar to legacy P2). AL + AZ ≈ 8h.
