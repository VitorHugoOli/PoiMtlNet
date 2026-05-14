# F40 — Scheduled cat_weight handover (Experiment C)

**Date:** 2026-04-26. **Tracker:** `FOLLOWUPS_TRACKER.md §F40`. **Cost:** ~30 min MPS sequential.

## Question

After F45 showed sustained-LR drives reg lift (but kills cat) and F48-H3-alt confirmed the per-head LR recipe (cat=1e-3, reg=3e-3, shared=1e-3) preserves cat AND lifts reg, F40 tests an **orthogonal lever: loss-side scheduling**. Specifically: does ramping `cat_weight` from 0.75 → 0.25 over 50 epochs (with B3 LR schedule unchanged) yield Pareto-lift?

Hypothesis: cat converges early under high cat_weight (matching B3); gradient budget then shifts to reg in the second half, replicating the F45 mechanism without destabilizing cat.

## Method

New loss class `ScheduledStaticWeightLoss` in `src/losses/scheduled_static/loss.py` registered as `scheduled_static`:

```python
ScheduledStaticWeightLoss(n_tasks=2, device, cat_weight_start=0.75,
                          cat_weight_end=0.25, total_epochs=50, warmup_epochs=0)
```

Trainer hook in `src/training/runners/mtl_cv.py` calls `mtl_criterion.set_epoch(epoch_idx)` at the start of each epoch (no-op for losses without `set_epoch`). `total_epochs` auto-defaults from `config.epochs` when not pinned via `--mtl-loss-param`.

Schedule trace (50 epochs, no warmup):
| Epoch | cat_weight |
|---:|---:|
| 0 | 0.7500 |
| 10 | 0.6480 |
| 25 | 0.4949 |
| 40 | 0.3418 |
| 49 | 0.2500 |

Architecture: B3 (`mtlnet_crossattn + next_gru cat + next_getnext_hard reg`, d=256, 8h). LR: B3 default OneCycleLR `max_lr=3e-3`, 50 epochs, batch 2048, seed 42, 5 folds.

CLI: `--mtl-loss scheduled_static --mtl-loss-param cat_weight_start=0.75 --mtl-loss-param cat_weight_end=0.25`

## Results

| Config | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 |
|---|---:|---:|---:|---:|
| B3 50ep static_weight cat=0.75 | 42.71 ± 1.37 | 59.60 ± 4.09 | 45.81 ± 1.30 | 53.82 ± 3.11 |
| **F40 scheduled 0.75→0.25** | **42.63 ± 1.26** | **60.81 ± 3.10** | **44.98 ± 1.05** | **54.39 ± 3.15** |
| F48-H1 const 1e-3 | 40.99 ± 1.80 | 61.43 ± 9.60 | 45.34 ± 0.84 | 50.68 ± 6.89 |
| F45 const 3e-3 | 10.44 💀 | 74.20 ± 2.95 | 12.23 💀 | 63.34 ± 2.46 |
| **F48-H3-alt per-head (winner)** | 42.22 ± 1.00 | **74.62 ± 3.11** | 45.11 ± 0.32 | **63.45 ± 2.49** |
| STL F21c (ceiling) | n/a | 68.37 ± 2.66 | n/a | 66.74 ± 2.11 |

## Pareto-lift acceptance

Threshold: cat F1 ≥ B3 - 1 pp AND reg Acc@10 > B3 + 3 pp.

| State | Cat | Reg | Outcome |
|---|---|---|---|
| AL | 42.63 ≥ 41.71 ✓ | 60.81 > 62.60? **✗** (+1.21) | **Cat OK, reg fails** |
| AZ | 44.98 ≥ 44.81 ✓ | 54.39 > 56.82? **✗** (+0.57) | **Cat OK, reg fails** |

Both states fail the reg leg of Pareto-lift. F40 **refutes the strong form** of the scheduled-handover hypothesis: cat preservation works but the loss-side mechanism alone does not produce ≥3 pp reg lift.

## Mechanism — why F40 underdelivers

The reg-lift mechanism credited by F45 is **α (graph-prior weight in `next_getnext_hard.head`) growing under sustained-3e-3 LR**. Under OneCycleLR (B3 default), the LR peaks ~ep 15 at 3e-3 then anneals — α never gets the long stable window it needs to grow productively, regardless of how the loss balances the two heads.

Schedule cat_weight 0.75 → 0.25 effectively gives reg more gradient budget late in training, but the budget is multiplied by an annealing LR. So the ratio (effective reg-side step size / time) stays similar to B3:

- B3: cat 0.75 × LR_peak=3e-3 vs reg 0.25 × LR_peak=3e-3, then both anneal symmetrically
- F40: cat 0.75→0.25 × LR_anneal vs reg 0.25→0.75 × LR_anneal, still co-anneals

The advantage is marginal because **the limiting factor is sustained-high LR at α, not gradient balance between heads**. Per-head LR (H3-alt) decouples α's LR from cat's LR; F40 does not — both heads share the OneCycleLR.

## Negative result is publishable

This is the cleanest **counterfactual** for the H3-alt mechanism claim. The paper can now state:

> "Loss-side scheduling (linearly ramping cat_weight 0.75 → 0.25 across 50 epochs under OneCycleLR) preserves cat performance but yields only +1 pp reg lift on AL and +0.6 pp on AZ — well below the +3 pp Pareto threshold and 13 pp below the per-head LR recipe (H3-alt). This confirms that the reg-lift bottleneck is the LR schedule at the graph-prior weight, not the loss-balance between heads."

## Implications

1. **H3-alt remains the only recipe that closes the gap.** F40 doesn't compete; it complements as a refutation.

2. **Future loss-side experiments are low-priority.** The F40 evidence makes any pure loss-side intervention (DWA, GradNorm, scheduled NashMTL) unlikely to close the gap on its own, given the same OneCycleLR confound.

3. **Combined recipe candidate:** F40 + per-head LR (H3-alt's per-head LR but with scheduled_static cat_weight). Hypothesis: cat fully converges in early epochs under cat_weight=0.75, then F40's ramp + H3-alt's per-head LR shifts most of the late-stage capacity to reg. **Not run** — H3-alt already exceeds STL on AL and closes 75% on AZ; the marginal gain might not justify the complexity. Mark as future work.

## Code landed

| File | Change |
|---|---|
| `src/losses/scheduled_static/loss.py` | New `ScheduledStaticWeightLoss` class; ctor params `cat_weight_start/cat_weight_end/total_epochs/warmup_epochs`; `set_epoch(e)` hook. |
| `src/losses/scheduled_static/__init__.py`, `metadata.yaml` | Standard module wiring. |
| `src/losses/registry.py` | Register `scheduled_static` in `_canonical_entries`. |
| `src/training/runners/mtl_cv.py` | Auto-default `total_epochs` from `config.epochs`; epoch hook calls `mtl_criterion.set_epoch(epoch_idx)` if defined. Back-compat (no-op for non-scheduled losses). |
| `scripts/run_f40_scheduled_handover.sh` | AL+AZ launcher with the spec'd 0.75→0.25 ramp. |

## Files

- Logs: `/tmp/check2hgi_logs/f40_then_h2.log` (Stage 1)
- AL: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep50_20260426_0838/summary/full_summary.json`
- AZ: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep50_20260426_0848/summary/full_summary.json`

## Cross-references

- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — winning per-head LR recipe (the contrast)
- `research/F44_F48_LR_REGIME_FINDINGS.md` — original F45 finding that motivated both F40 and H3
- `FOLLOWUPS_TRACKER.md §F39` — earlier evidence (cat_weight static sweep) that loss-weight ∈ {0.25, 0.5, 0.75} doesn't shift reg
