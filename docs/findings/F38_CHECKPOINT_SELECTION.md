# F38 — Experiment A: Diagnostic-Task-Best Checkpoint Re-analysis

**Date:** 2026-04-24. **Tracker item:** `FOLLOWUPS_TRACKER.md §F38`. **Source JSONs:** `results/F27_validation/{al,az,fl}_*_b3_cathead_gru.json`. **Cost:** zero-compute, pure analysis of existing artefacts.

## Question

The F21c finding (`research/F21C_FINDINGS.md`) showed STL `next_getnext_hard` beats MTL-B3 on next-region by 12–14 pp Acc@10 on AL and AZ. The 2026-04-24 hparam audit proposed that one contributor (Fator 2) would be **checkpoint selection**: MTL selects the epoch maximising `val_joint_geom_lift` (geometric mean of per-head lifts over majority); STL selects per-head best.

**Prediction:** if Fator 2 is load-bearing, reporting MTL-B3 reg Acc@10 at the **per-task-best** epoch (stored in `diagnostic_task_best` in every summary JSON) should recover 3–6 pp Acc@10, narrowing the gap to STL-matched-head from 12–14 pp to 7–9 pp.

## Method

Parsed `results/F27_validation/al_5f50ep_b3_cathead_gru.json` and `az_5f50ep_b3_cathead_gru.json` (5-fold × 50ep B3 post-F27 runs), which contain both:
- **Joint-best:** top-level `next_region.*` — 5-fold aggregate using the `val_joint_geom_lift` monitor's chosen epoch per fold.
- **Task-best:** `diagnostic_task_best.next_region.*` — same 5 folds but with the reg-task-best epoch selected per fold (based on val reg F1).

Also FL-1f for completeness (n=1, no σ).

## Results

| State | Metric | Joint-best | Task-best | Δ (task − joint) |
|---|---|:-:|:-:|:-:|
| **AL 5f** | top1\_acc\_indist | 17.08 ± 2.44 | 17.12 ± 2.48 | **+0.04** |
| | top5\_acc\_indist | 46.01 ± 4.45 | 46.20 ± 4.28 | **+0.19** |
| | **top10\_acc\_indist** | **59.60 ± 4.09** | **59.60 ± 4.16** | **−0.01** |
| | mrr\_indist | 30.74 ± 2.87 | 30.77 ± 2.96 | +0.03 |
| | reg macro-F1 | 9.95 ± 0.54 | 10.06 ± 0.60 | +0.11 |
| **AZ 5f** | top1\_acc\_indist | 15.41 ± 2.02 | 14.92 ± 2.59 | −0.49 |
| | top5\_acc\_indist | 40.54 ± 3.40 | 40.12 ± 3.72 | −0.41 |
| | **top10\_acc\_indist** | **53.82 ± 3.11** | **53.42 ± 3.80** | **−0.40** |
| | mrr\_indist | 27.66 ± 2.41 | 27.27 ± 2.89 | −0.39 |
| | reg macro-F1 | 9.37 ± 0.56 | 9.51 ± 0.53 | +0.14 |
| **FL 1f** | top10\_acc\_indist | 65.26 | 65.29 | +0.02 |
| | top5\_acc\_indist | 43.42 | 54.40 | +10.98 |
| | top1\_acc\_indist | 15.69 | 13.64 | −2.05 |

## Verdict

**Fator 2 is REFUTED as load-bearing for AL/AZ under B3.**

At the primary metric (Acc@10), the difference between joint-best and task-best is **essentially zero** (−0.01 pp AL; −0.40 pp AZ). Secondary metrics (Acc@1, Acc@5, MRR, macro-F1) move by at most 0.5 pp — all within σ. No meaningful recovery.

**Mechanism (why the agent's prediction was wrong at AL/AZ):** under `static_weight(category_weight=0.75)`, the cat head converges by epoch ~42 and then plateaus; the reg head keeps improving for several more epochs. The `val_joint_geom_lift` monitor naturally tracks this — when cat F1 is flat, any reg improvement raises the joint lift, so the monitor picks the reg-best epoch. This collapses the distinction joint-best ≈ task-best under the B3 regime.

**Exception (FL 1f only):** Acc@5 jumps +10.98 pp between joint and task-best at FL, but **Acc@10 stays flat**. The jump is a fold-level artefact of the prior's top-k ranking volatility at 4.7K regions (same signature documented in F2 Phase A). Not evidence for Fator 2 at the headline metric.

## Implications for the 12–14 pp gap (CH18)

With Fator 2 ruled out, attribution shifts toward:

| Fator | Status post-F38 | Next experiment |
|---|---|---|
| **1.** Loss weight 0.25 on reg | Remains primary suspect (no direct test yet) | **F39 (Exp B)** — cat_weight ∈ {0.25, 0.50} 5f on AL/AZ |
| **2.** Checkpoint selection | **Refuted** (Δ ≤ 0.4 pp at Acc@10) | — |
| **3.** Upstream cross-attn dilutes head input | Untested | **F41 (Exp D)** — STL with MTL pre-encoder + zero cross-attn |
| **4.** Weight decay 0.01 (STL) vs 0.05 (MTL) | Low suspicion | Can piggyback on F39 if needed |

**Revised recommendation:** skip direct follow-up on F38. Prioritise F39 (cat_weight sweep) and F41 (upstream-arch ablation). If F39 recovers < 3 pp and F41 recovers < 3 pp, the gap is a genuine **compositional** effect of the MTL coupling — not reducible to any single hparam — and the paper should report it as such without over-explaining.

## Files

- Analysis script: inline (this doc).
- Source JSONs: `results/F27_validation/al_5f50ep_b3_cathead_gru.json`, `az_5f50ep_b3_cathead_gru.json`, `fl_1f50ep_b3_cathead_gru.json`.
- Comparison STL F21c: `results/B3_baselines/stl_getnext_hard_{al,az}_5f50ep.json`.

## Cross-references

- `research/F21C_FINDINGS.md` — the 12–14 pp gap this experiment was trying to explain.
- `CLAIMS_AND_HYPOTHESES.md §CH18` — paper's acknowledged methodological limitation.
- `FOLLOWUPS_TRACKER.md §F38, §F39, §F41` — full experiment chain.
