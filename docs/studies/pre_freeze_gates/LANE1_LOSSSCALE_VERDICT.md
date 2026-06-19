# Lane 1 — loss-scale normalization advisory: VERDICT = EXCLUDE (harmful at scale)

> A40, `study/pre-freeze-a40`, 2026-06-18. EXECUTION_PLAN §8 #5 / Decisions #5 (user-approved RUN).
> Same-code A/B vs champion G at AL+FL seed0, v14 substrate, geom_simple, KD off. Lever = `--loss-scale-norm`
> (the in-code `mtl_improvement` T4.0a: divide each task CE by `log(num_classes)` before the static `cw=0.75`).
> **Already implemented in-code** — no new code; this is a run-only advisory.

## Result (per-task diagnostic-best; cat=next_category macro-F1, reg=next_region Acc@10-indist)

| state | arm | cat-F1 | Δcat | reg@10 | Δreg |
|---|---|---|---|---|---|
| AL s0 | baseline   | 52.38 | — | 64.34 | — |
| AL s0 | loss-scale | 52.99 | **+0.61** | 63.92 | **−0.42** |
| FL s0 | baseline   | 73.01 | — | 73.54 | — |
| FL s0 | loss-scale | 72.18 | **−0.83** | **35.73** | **−37.81** |

(FL baseline cat 73.01 == documented deterministic champion-G FL seed0 73.012 → harness reproduces G exactly.)

## Verdict: EXCLUDE — decisively harmful at scale

The AL gate-fire (cat +0.61) is a **small-state artifact of a cat↑/reg↓ rebalance**, NOT a real improvement.
At FL the **same** rebalance is **catastrophic for reg: −37.81 pp** (73.54 → 35.73), with cat also down −0.83.
The FL run is genuine (5/5 folds, 17.8 min, no NaN/crash): every fold's reg best is hit at **epoch 2-6** then
frozen at ~36 % Acc@10 with **macro-F1 ≈ 0** — a **degenerate reg head**.

**Mechanism (clean):** normalization divides reg CE by `ln(n_reg)` (FL ln 4703 ≈ 8.46) but cat CE by only
`ln 7 ≈ 1.95` — a ~4.3× relative down-weight of reg, **stacked on top of** the existing `cw=0.75` (which
already favours cat). At small AL (`ln 1109 ≈ 7.0`) the reg head survives (−0.42); at FL the much larger
4703-class reg head is gradient-starved into an early poor optimum and the onecycle/early-best lock it there.
The lever does the **opposite** of what the §8 #5 hypothesis hoped (it was aimed at *helping* reg) — it starves
reg, and the starvation worsens with region cardinality → it would be worst at CA/TX (~8.5k/6.5k regions).

## Disposition

- **NOT a v17 candidate. Recipe stays v16 / champion G.** `loss_scale_norm=False` is correct.
- **EXPLICITLY EXCLUDED on the record** (the §8 #5 / Decisions #5 null branch — here stronger than null: harmful).
  No multi-seed needed: FL seed0 alone is decisive (a −37.8 pp collapse is not a seed-variance question).
- Mirror the composite/routing exclusion wording in the freeze notes: "loss-scale normalization tested
  pre-freeze (AL+FL seed0); harmful at scale (FL reg −37.8 pp, degenerate head); excluded."
- This is the FL-scale-validation discipline paying off — AL-only would have mis-promoted it (cf. the overlap
  memo's AL +9.8 / FL +1.3 saturation lesson, same shape: small-state signal ≠ scale signal).

## Captures
`results/lane1_g01/{alabama,florida}_s0__{baseline,lossscale}/` (full_summary.json + fold*_info.json +
RUNDIR.txt). Scorer: `scripts/pre_freeze_gates/lane1_score.py`. Driver: `scripts/pre_freeze_gates/lane1_run.sh`.
