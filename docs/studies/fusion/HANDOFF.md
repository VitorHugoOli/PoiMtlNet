# Handoff — state as of 2026-04-17 (P1 complete)

Snapshot written at session close. This file is **transient**; trust `state.json` when in doubt.

---

## Study status at a glance

- **Current phase marker:** P1 (status=running in state.json but **phase gate PASSES** — awaiting `/study advance` to open P2).
- **P0 ✅ closed.** P1 ✅ **complete, full three-stage 180-test grid on AL + AZ**. P2–P6 planned.
- **Claims catalog:** 30 claims + 4 negations. C02 → `partial`, C05 → `confirmed`, C29/C30/N03 already confirmed in P0. See `CLAIMS_AND_HYPOTHESES.md`.
- **Git:** `main` is clean-ish; working-tree edits (CLAIMS, P1 SUMMARY, this file, state.json mutations from drains) are **uncommitted**. Latest commit `97f7fdb fix: fix the gradnorm`.

---

## P1 outcome (matched batch, fusion, seed 42)

| Stage | AL | AZ | Verdict |
|-------|----|----|---------|
| screen (1f×10ep) | 75 | 75 | all `matches_hypothesis` |
| promote (2f×15ep) | 10 | 10 | all `matches_hypothesis` |
| confirm (5f×50ep) | 5 | 5 | all `matches_hypothesis` |

- **AL P1c winner:** `mmoe4 × gradnorm` joint **0.4082** (cat 0.822±0.012, next 0.272±0.011).
- **AZ P1c winner:** `cgc21 × uncertainty_weighting` joint **0.4374** (cat 0.732±0.015, next 0.312±0.014).
- **Winner disagreement across states** — different arch AND different optim class. Paper must carry a state-sensitivity caveat.
- Full tables + cross-stage trajectory: `docs/studies/fusion/results/P1/SUMMARY.md`.

### Claim updates from P1

| Claim | Before P1 | After P1c | Delta |
|-------|-----------|-----------|-------|
| **C02** grad-surgery > eq on fusion | `partially_refuted` | **`partial`** | AL +0.0051 (marginal), AZ −0.0005 (static tied/better). Within noise. |
| **C05** expert-gating > FiLM base | `partial` | **`confirmed`** | Every expert arch > base on both states at screen (75 cells each). |

C03, C04 still `pending` — deferred to P3 (DGI/HGI cross-engine).

### P1-derived second-order observations worth noting for the paper

1. **C02 is effectively null at matched batch.** The +0.0051 AL advantage is within noise; AZ reverses sign. On fusion, the optimizer choice is second-order. Architecture (C05) is first-order.
2. **`equal_weight` placed within 0.5 p.p. of the AL winner** at 5f×50ep. Had the phase doc's "pause-and-replan" trigger been strict rather than "winner === equal_weight", this would have fired.
3. **NextHead overfitting signature.** Next F1 loses ~2 p.p. going from promote (2f×15ep) to confirm (5f×50ep) on 4 of 5 AL cells. Only `mmoe4 × gradnorm` avoids the drop. Early-stopping the next head or adding dropout may unlock more joint F1.
4. **Cross-state arch winners differ.** AL likes `mmoe4`/`cgc22`, AZ likes `cgc21`. Fusion engine same, state-dependence real. Treat any single "champion" with skepticism — it may not transfer to FL.

---

## Data availability snapshot (unchanged from 2026-04-15)

| State   | dgi | hgi | fusion | sphere2vec | time2vec (next) | poi2hgi | check2hgi |
|---------|:---:|:---:|:------:|:----------:|:---------------:|:-------:|:---------:|
| alabama |  ✓  |  ✓  |   ✓    |     ✓      |        ✓        |    ✗    |     ✗     |
| arizona |  ✓* |  ✓  |   ✓    |     ✗      |        ✗        |    ✗    |     ✗     |
| florida |  ✓* |  ✓  |   ✓    |     ✓      |        ✓        |    ✗    |     ✗     |

Open issues (state.json):
- `az_fl_dgi_stale` — AZ + FL DGI parquets pre-bugfix (no `placeid`). **Blocker for P3** (DGI cross-engine). Regenerate before enrolling P3 DGI cells on AZ/FL.
- `fl_fusion_scale` — FL fusion half-L2 ratio 40.99× (AL is ~15×). Likely HGI vs Sphere2Vec scale imbalance specific to FL. **Confounds C02/C19 on FL**. Investigate before P3 FL runs.

---

## Next steps (recommended order)

1. **Advance phase gate.** Run `.venv/bin/python scripts/study/study.py advance` (or `/study advance`) to flip P1 → completed and open P2. Gate passes cleanly. This is a paper-direction-safe call (neither state's winner is `equal_weight`).
2. **Commit P1 artifacts.** Working-tree changes to commit: `CLAIMS_AND_HYPOTHESES.md`, `docs/studies/fusion/results/P1/SUMMARY.md`, `docs/studies/fusion/results/P1/<180 archived test dirs>`, `docs/studies/fusion/state.json`, `docs/studies/fusion/HANDOFF.md`. (Also housekeeping: `docs/studies/fusion/CLAIMS_AND_HYPOTHESES.md`, `pipelines/train/next_head.pipe.py`, `src/losses/gradnorm/loss.py`, `tests/test_losses/test_gradnorm.py` per `git status`.)
3. **Enroll P2** — see `docs/studies/fusion/phases/P2_heads_and_mtl.md`.
   - Champion to freeze into P2: **`mmoe4 × gradnorm`** (AL winner; only config to improve screen → confirm).
   - Sensitivity check: also run P2's C06 (MTL vs single-task) on AZ's `cgc21 × uncertainty_weighting` to cover the cross-state disagreement.

## Worth-answering open questions (for P2 and beyond)

**Mechanistic (answerable in P1 data, no new runs):**
- Q1. **Why does only `mmoe4 × gradnorm` improve from promote → confirm?** Look at `results/fusion/alabama/mtlnet_lr1.0e-04_bs4096_ep50_20260416_1936/` metric-store. Is it the gating structure (mmoe4 routes next-POI to a specialist expert) or does gradnorm specifically reweight the next loss as training progresses? Could inform an ablation: mmoe4 × equal_weight vs mmoe4 × gradnorm at 5f×50ep.
- Q2. **Is the next-POI drop promote → confirm a true overfit or a schedule artifact?** OneCycleLR ramps across full 50 epochs; at 15 epochs the LR has a different profile. Per-epoch val-F1 trajectories in `MetricStore` should tell us at a glance.
- Q3. **Why does the AZ winner flip to static (`uncertainty_weighting`)?** Check AZ fusion half-L2 ratio — is it more or less imbalanced than AL? The memory note `fusion_embedding_design` says ratios matter for which optimizer helps; a direct measurement would close the loop.

**Requires P2 data (free from P2 runs, just analyze):**
- Q4. **C28 — no negative transfer** is mandatory for any reviewer. P2 will produce paired fold-level F1 for MTL vs single-task; run Wilcoxon + Cohen's d immediately.
- Q5. **C23 — wall-clock ratio.** P2 logs single-task and MTL wall times. Compute `wall_MTL / (wall_cat + wall_next)`. CBIC reported 4×; expect ours to be lower.
- Q6. **C24 — train/val gap.** Free from existing logs. If MTL shows smaller gap than single-task on the same fusion+arch, that's a bonus figure.

**Requires modest new runs (consider for P2 time budget):**
- Q7. **Cross-state champion check.** Run the AL winner on AZ, and vice versa, at 5f×50ep. 2 runs × ~35 min = ~70 min. Closes the paper's "which champion transfers?" question with one table.
- Q8. **`base × equal_weight` ceiling at 5f×50ep on AL.** We have no confirm-stage base cell. One run (~35 min) would lock C05 under the same protocol as C02.

**Deferred (P3–P6):**
- Q9. **C29 generalization audit.** Fusion contains Sphere2Vec — does the fclass shortcut still dominate category F1 on fusion, or does Sphere2Vec break it? An arm-C-style shuffle on fusion would be a Phase 3 must.
- Q10. **FL fusion scale (40.99×) before P3.** Decide: (a) regenerate FL fusion with different normalization, (b) accept and caveat, (c) drop FL from C01 verification.

---

## How to invoke next session

```
.venv/bin/python scripts/study/study.py advance   # close P1, open P2
# then
/coordinator P2                                     # set champion, list tests
/worker P2 all                                      # drain
```

Session checklist on resume:
- Confirm `state.json` current_phase = P2 (not P1).
- Confirm uncommitted P1 artifacts were either committed or intentionally left.
- Re-read `phases/P2_heads_and_mtl.md`; decide champion = mmoe4×gradnorm + (optional) cgc21×uw robustness arm.
