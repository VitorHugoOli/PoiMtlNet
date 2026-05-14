# Handoff — state as of 2026-04-17 (P1 closed, critical review complete)

Snapshot written at session close. This file is **transient**; trust `state.json` when in doubt.

---

## ⚠️ Read first

`docs/studies/fusion/issues/P1_METHODOLOGY_FLAWS.md` catalogs 10 flaws identified in the post-P1 critical review. The two high-severity items (F1 joint-peak checkpoint bias; F2 single-seed) **must be resolved or explicitly acknowledged before P2 runs** — P2's #1 paper claim (C06 MTL vs single-task) rides on the champion we freeze from P1.

---

## Study status at a glance

- **Current phase:** P2 (P1 → completed via `/study advance` 2026-04-17).
- **P0 ✅ closed. P1 ✅ closed** (189 P1 tests archived: 180 original + 1 F4 base + 8 F2 multi-seed). P2 open, 0 enrolled. P3–P6 planned.
- **Claims catalog:** 32 claims + 4 negations.
  - **C02** (grad-surgery > eq): **`refuted`** — multi-seed null, t-stats |t|<0.7.
  - **C05** (expert > base): **`partial`** — screen supports, confirm inverts direction.
  - **C18** (reproducibility): **`confirmed`** — all 4 F2 candidates std<0.01 on joint@J; std<0.004 on joint@T.
  - **C31** (fclass shortcut on fusion): **`partial`** — linear probe strongly supports; full arm-C retrain pending.
  - **C32** (joint-peak checkpoint bias): **`confirmed`** — joint@T now first-class metric.
- **Git:** clean-ish. Latest commits: `0161c7a` F1/F3, `42b4035` F4, then this session's F2 finalization (still uncommitted at this note).

---

## P1 outcome (matched batch, fusion, seed 42)

| Stage | AL | AZ | Verdict |
|-------|----|----|---------|
| screen (1f×10ep) | 75 | 75 | all `matches_hypothesis` |
| promote (2f×15ep) | 10 | 10 | all `matches_hypothesis` |
| confirm (5f×50ep) | 5 | 5 | all `matches_hypothesis` |

- **AL "winner" at joint-peak selection (joint@J):** `mmoe4 × gradnorm` = **0.4082**.
- **AL at per-task-best selection (joint@T):** `cgc22 × equal_weight` = **0.4229**. All 5 AL top-5 collapse to 0.4215–0.4229 (spread 0.0014). **The AL joint@J ranking is a checkpoint-selection artifact (C32/F1).**
- **AZ "winner" at joint@J:** `cgc21 × uncertainty_weighting` = **0.4374**.
- **AZ at joint@T:** `cgc21 × dwa` = **0.4416**. Top-4 cgc21 within 0.0024.
- **C02 refuted under joint@T.** AL grad−eq = −0.0009; AZ grad−static = −0.0010.
- Full tables + per-task-best reanalysis: `docs/studies/fusion/results/P1/SUMMARY.md`.

### Claim updates from P1

| Claim | Before | After P1 + critical review | Delta |
|-------|--------|----------------------------|-------|
| **C02** grad-surgery > eq on fusion | `partially_refuted` | **`partially_refuted`** (reinforced) | AL joint@J +0.0051 (Z=0.39 → null); AL joint@T −0.0009; AZ joint@T −0.0010. No measurable advantage. |
| **C05** expert-gating > FiLM base | `partial` | **`confirmed`** (with F4 caveat — `base` not at 5f×50ep) | Every expert arch > base at screen (75 cells × 2 states); unanimous direction. |
| **C31** fclass shortcut on fusion | — (new) | `pending` (blocking for P2/P3) | 1-fold arm-C run on fusion needed. Results unknown. |
| **C32** joint-peak checkpoint bias | — (new) | `confirmed` | Ranking changes under per-task-best selection; must report both going forward. |

C03, C04 still `pending` — deferred to P3 (DGI/HGI cross-engine).

### P1-derived second-order observations worth noting for the paper

1. **C02 is null at matched batch** under both checkpoint policies. On fusion, optimizer choice is not a measurable knob. Architecture (C05) is first-order.
2. **`equal_weight` leads AL under per-task-best selection.** The paper narrative shifts: "MTL balancing is over-engineered; on fusion, equal_weight works."
3. **Asymmetric per-task peak epochs (C32).** Category peaks epoch 17–45; next peaks epoch 10–22. Any single-checkpoint joint F1 is a compromise; P2 must report both joint@J and joint@T.
4. **Cross-state arch preference** (AL likes mmoe4/cgc22, AZ likes cgc21) persists under both checkpoint policies. Is it real or seed-42 noise? Unknown without F2 multi-seed. Treat any single "champion" with skepticism — it may not transfer to FL.

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

## Next steps (F1–F4 now resolved; ready for P2)

**Completed 2026-04-17:** F1 (joint@T backfilled for 181 tests + archive_result.py emits going forward), F2 (8 multi-seed runs), F3 proxy (fclass probe; full arm-C retrain still pending but the proxy signal is decisive), F4 (base × eq AL confirm — C05 downgraded).

**Remaining P1 loose ends (not blocking P2):**
1. **F3 primary test** — full arm-C retrain on fusion (not just linear probe). ~45 min. Converts C31 from `partial` → `confirmed`.
2. **F4 follow-up** — base × nash_mtl (or × gradnorm) AL confirm (~35 min). Cross-checks base tie at confirm under a gradient-based optimizer.

**P2 ready to start** with the following champion choices (post-F2 multi-seed):

| State | Champion | Why |
|-------|----------|-----|
| AL | **`mmoe4 × gradnorm`** | Most stable (seed std 0.0008 joint@J). Mean joint tied at joint@T (0.4232 vs cgc22×eq 0.4237 = −0.0005). |
| AZ | **`cgc21 × uncertainty_weighting`** *or* `cgc21 × dwa` | Tied at joint@T (0.4394 vs 0.4412). Pick uw for continuity with the P1 headline; either is fine. |

For C06 (MTL vs single-task) in P2, report both joint@J and joint@T. The joint@J metric will favor single-task-next because of the NextHead-peak-early pattern (F5); joint@T is the fair comparison.

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

P1 has been closed via `/study advance`. Before any P2 work:

```
# 1. Commit the P1 body of work
git status
git add docs/studies/fusion/ src/losses/gradnorm/loss.py tests/test_losses/test_gradnorm.py pipelines/train/next_head.pipe.py
git commit -m "study(P1): complete 3-stage grid + critical review + flaw tracker"

# 2. Run the fclass-on-fusion check (C31)
.venv/bin/python experiments/hgi_leakage_ablation.py --engine fusion --arm C_fclass_shuffle --state alabama

# 3. Backfill joint@T into state.json (one-off script)
#    See issues/P1_METHODOLOGY_FLAWS.md F1 resolution

# 4. Multi-seed AL+AZ champion candidates (F2)
#    Enrollment script needed — see F2 resolution. ~5 h total.

# 5. Only then: enroll and drain P2
/coordinator P2
/worker P2 all
```

Session checklist on resume:
- Confirm `state.json` current_phase = P2.
- Confirm working tree is clean (P1 committed).
- Read `issues/P1_METHODOLOGY_FLAWS.md` first — the F1/F2/F3 remediation plan determines what P2 actually measures.
- Re-read `phases/P2_heads_and_mtl.md` and reconcile the champion picked post-F2 with the "arch* × optim*" assumption it makes.
