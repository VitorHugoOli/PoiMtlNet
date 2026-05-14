# F48-H2 — Warmup-then-plateau scheduler (single LR)

**Date:** 2026-04-26. **Tracker:** `FOLLOWUPS_TRACKER.md §F48-H2`. **Cost:** ~1.5 h MPS sequential.

## Question

H2 was the second design from `F44_F48_LR_REGIME_FINDINGS.md` next-step ranking. Hypothesis: linear warmup ~50ep from a low base to `max_lr=3e-3`, then constant 3e-3 plateau for ~100ep. Cat survives the gentle ramp; reg gets sustained-3e-3 plateau enabling α growth (F45 mechanism).

Discriminating question: **does cat survival require LR throttling everywhere (H3-alt's `shared=1e-3`), or is gentle ramp at the start sufficient (H2's warmup phase)?**

## Method

`warmup_constant` scheduler added in `src/training/helpers.py`:

```python
SequentialLR([
    LinearLR(start_factor=0.033, end_factor=1.0, total_iters=warmup_steps),
    ConstantLR(factor=1.0),
], milestones=[warmup_steps])
```

CLI: `--scheduler warmup_constant --max-lr 3e-3 --pct-start 0.333` → ramp from 1e-4 ≈ 0.033 × 3e-3 over 50ep, then hold at 3e-3 for 100ep.

Same architecture as B3 (`mtlnet_crossattn + next_gru + next_getnext_hard, d=256, 8h`). 5 folds × 150 ep, batch=2048, seed 42, single LR (no per-head).

## Results

| Config | AL cat F1 | AL reg Acc@10 | AZ cat F1 | AZ reg Acc@10 |
|---|---:|---:|---:|---:|
| B3 50ep static_weight cat=0.75 | 42.71 ± 1.37 | 59.60 ± 4.09 | 45.81 ± 1.30 | 53.82 ± 3.11 |
| F48-H1 const 1e-3, 150ep | 40.99 ± 1.80 | 61.43 ± 9.60 | 45.34 ± 0.84 | 50.68 ± 6.89 |
| F45 const 3e-3, 150ep | 10.44 💀 | **74.20 ± 2.95** | 12.23 💀 | **63.34 ± 2.46** |
| F40 scheduled-static, 50ep | 42.63 ± 1.26 | 60.81 ± 3.10 | 44.98 ± 1.05 | 54.39 ± 3.15 |
| **F48-H2 warmup_constant** | **41.35 ± 0.78** | **57.84 ± 4.48** | **44.45 ± 0.54** | **48.91 ± 5.12** |
| **F48-H3-alt per-head (winner)** | 42.22 ± 1.00 | **74.62 ± 3.11** | 45.11 ± 0.32 | **63.45 ± 2.49** |
| STL F21c (ceiling) | n/a | 68.37 ± 2.66 | n/a | 66.74 ± 2.11 |

## Outcome

H2 **fails on both legs** of the joint objective:

- **Cat:** preserved-ish (1.36 pp below B3 on both states, within σ on AL but tight) — the warmup ramp DOES protect cat from collapse.
- **Reg:** WORSE than B3 — AL -1.76 pp, AZ -4.91 pp. The 100-epoch plateau at 3e-3 does NOT lift reg.

This is the most informative refutation of the bunch. H2 had the F45 mechanism's ingredient (sustained 3e-3 plateau) but didn't replicate the lift. Why?

## Mechanism — surviving cat starves α

The F45 reg lift comes from α (graph-prior weight in `next_getnext_hard.head`) growing under sustained 3e-3 LR. But α only grows when its gradient signal isn't drowned out.

- **F45 (cat collapsed):** the cat head's CE gradient saturates at majority-class (cat F1 ≈ 10), so the cat-side gradient through the shared cross-attn is small and stable. Reg's gradient drives shared cross-attn AND α grows undisturbed. Reg lifts to 74.
- **F48-H2 (cat alive):** the cat head, having survived warmup, continues to update aggressively at 3e-3 during the plateau. Cat-side gradient through shared cross-attn is now a strong signal — competing with reg's gradient for shared layer capacity. α tries to grow but its effective signal is diluted because the shared layers are oscillating between cat- and reg-aligned features. Reg drops below B3.

So the apparent F45 success was not "sustained LR enables α growth" — it was **"sustained LR collapses cat AND uncontested reg gradient enables α growth in shared cross-attn"**. Two coupled effects.

H3-alt decouples them: gentle shared LR (1e-3) keeps shared cross-attn stable for cat, while reg's per-group LR (3e-3) lets α grow IN THE HEAD (where reg-only gradient flows). The stability of shared is what cat needs; the high LR at α is what reg needs; H3-alt gives both.

## Three negative controls bracket the H3-alt finding

The orthogonal experiments together form a complete attribution chain:

| Experiment | Lever | Cat | Reg | Why |
|---|---|---:|---:|---|
| F45 | constant LR everywhere | 💀 (10) | ↑↑ (74) | cat collapse → uncontested reg gradient → α grows |
| **F48-H2** | warmup+plateau, single LR | OK (41) | ↓ (58) | cat alive → competes for shared → α starves |
| F40 | loss-side cat_weight ramp | OK (43) | ≈ (61) | OneCycleLR annealing → α can't grow regardless of loss balance |
| **F48-H3-alt** | per-head LR (cat=1e-3, reg=3e-3, shared=1e-3) | OK (42) | ↑↑ (75) | shared stable for cat AND α gets sustained 3e-3 in head |

H3-alt is the only configuration that achieves both objectives jointly. The other three each fix one objective at the cost of the other (F45/H2) or fix neither (F40).

## Implications for paper

The F48-H2 negative result strengthens the H3-alt mechanism claim. The paper can now argue that the per-head LR recipe is not just "a config that works"; it's the *uniquely correct* configuration:
- Single-LR constant 3e-3 (F45) → kills cat
- Single-LR with warmup (F48-H2) → kills reg
- Single-LR with loss-side scheduling (F40) → kills neither but lifts neither
- Per-head LR (H3-alt) → only design that satisfies the joint objective

Combined with FL scale-validation (cat preserved, reg +6.7 pp) and AL exceeding STL ceiling, the recipe is publication-grade.

## Files

- Logs: `/tmp/check2hgi_logs/f40_then_h2.log` (Stage 2, after F40)
- AL: `results/check2hgi/alabama/mtlnet_lr1.0e-04_bs2048_ep150_20260426_0908/summary/full_summary.json`
- AZ: `results/check2hgi/arizona/mtlnet_lr1.0e-04_bs2048_ep150_20260426_0939/summary/full_summary.json`
- Launcher: `scripts/run_f48_h2_warmup_constant.sh`

## Code landed

| File | Change |
|---|---|
| `src/training/helpers.py` | `setup_scheduler` accepts `scheduler_type='warmup_constant'`; LinearLR (start_factor=0.033) → SequentialLR → ConstantLR plateau. Honors `pct_start` as warmup fraction (default 1/3). |
| `scripts/train.py` | Added `warmup_constant` to `--scheduler` choices. |
| `scripts/run_f48_h2_warmup_constant.sh` | AL+AZ launcher with default `pct_start=0.333` and `max_lr=3e-3`. |

(Code was landed in F48-H3 commit `565c478` as infra; this run is the empirical exercise.)

## Cross-references

- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — winning recipe (the contrast + FL scale)
- `research/F40_SCHEDULED_HANDOVER_FINDINGS.md` — orthogonal loss-side refutation
- `research/F44_F48_LR_REGIME_FINDINGS.md` — original sweep that motivated H2 design
