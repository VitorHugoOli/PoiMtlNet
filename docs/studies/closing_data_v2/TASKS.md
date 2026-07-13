# closing_data_v2 — Task list & objective

> **Read this first.** This is the charter for `closing_data_v2`. It exists so the work does not drift.
> Everything here is scoped to the **v17** model on the **`check2hgi_dk_ovl`** (overlap-gated, stride-1,
> MIN_SEQ=10) substrate — the exact configuration behind the MobiWac Table 3. **We report nothing on a
> different model version, substrate, or ceiling set.** The runs live on the `closing-data/v17-ceilings-n20`
> branch / A40 checkout; this study only re-reads and re-scores them (CPU-only, no retraining).

## Objective (one sentence)

Produce and audit the **joint-best** (single deployable checkpoint) numbers for the six Table-3 "Joint (ours)"
MTL cells — the numbers the *one* saved checkpoint per fold actually delivers, both heads read at the
`geom_simple`-selected epoch — and present them side-by-side with the currently-reported **per-task
diagnostic-best** cells, so the camera-ready / response-letter can state the deployable numbers honestly and
**close the diag-best-vs-joint-best gap** the author flagged (`J1_JOINT_SCORE_RUNBOOK.md`).

### Why this gap exists (the one-paragraph background)
Table 3's MTL cells are **diagnostic-best**: category is read at its own macro-F1-best epoch and region at its
own Acc@10-best epoch — *two different epochs per fold*. That is an honest per-head headroom number, but it is
**not a single served model**. Training also saves **one joint checkpoint per fold**, gated on
`geom_simple = sqrt(cat_macroF1 · reg_top10_acc_indist)` on validation. Nobody had scored it. The paper §6.2
disclosure sentence was removed (author decision 2026-07-09) and a hidden response-letter note says the joint
checkpoint "was not re-scored … can be provided on request; do NOT claim they match." **This study provides it.**

## Scope guardrails (do NOT violate)
- **Model = v17** (= v16 + `--batch-size 8192` + `--onecycle-per-head-lr`, cat/reg/shared 1e-3/3e-3/1e-3).
- **Substrate = `check2hgi_dk_ovl`** (gated stride-1 overlap, MIN_SEQ=10). fp32. `geom_simple` selector, `min_best_epoch=0`.
- **Same rundirs as Table 3** — AL/AZ/FL n=20 (`perhead_lr_n20`), CA/TX seed-0 5f (`catx_v17_seed0_5f`),
  Istanbul n=20 (`h3_istanbul`). No new training. No re-tuned ceilings (the n=20 best-vs-best ceilings from
  `CEILINGS_N20_FINAL.md` stand).
- **Never report an MTL number without its epoch-selection convention** (diag-best vs joint-best) — the root
  cause of the original confusion (`JOINT_BEST_SCORING.md §5`).
- STL "Dedicated" ceilings need **no** joint-best re-score: a single-task run's saved checkpoint *is* its
  diag-best (one head, one best epoch). Joint-best applies only to the MTL cells.

## Task list

- [x] **T1 — Map & verify the 18 v17 MTL rundirs.** AL×4, AZ×4, FL×4 (the `new` bs8192 per-head runs, not
  `base`), CA×1, TX×1, Istanbul×4. Authoritative (state, seed)↔rundir map via PID + each rundir's
  `a40_matched_score.json` "seed". Exclude the Istanbul cascade run (`341087`). All carry 5 per-epoch CSVs. → `PROVENANCE.md`.
- [x] **T2 — Run `score_joint_best.py` on all 18 rundirs + aggregate to cells.** Sidecars written into each
  rundir. n=20 cell = mean over 4 seeds of the per-seed fold-mean (cross-seed sd); CA/TX = single-seed fold-mean
  (fold sd). → `JOINT_BEST_RESULTS.md`, `data/j1_results.json`.
- [x] **T3 — Three built-in parity gates.** (A) scorer diag-best == committed `a40_matched_score.json` (18/18);
  (B) scorer `e*` == training `primary_checkpoint.epoch` (all 90 folds, +1 indexing); (C) diag-best cell ==
  paper Table 3 (6/6). → `JOINT_BEST_RESULTS.md §Audit`.
- [x] **T4 — Independent adversarial audit (agent fan-out).** From-scratch re-derivation (no scorer/scoring.py),
  AL region tail-risk, Istanbul/FL region-margin sensitivity, completeness critic. → `AUDIT.md`.
- [x] **T5 — Corrected table + decision memo.** Side-by-side diag-best vs joint-best, Δ vs ceilings under both
  conventions, does any verdict change? → `JOINT_BEST_RESULTS.md`.
- [ ] **T6 (deferred, now CPU-only) — CA/TX joint-best at n=20.** **A1 (CA/TX v17 MTL n=20, seeds {1,7,100}) is
  DONE — completed 2026-07-11 on the A40** (`docs/results/closing_data/catx_v17_n20/`; the n=20 diag-best confirms
  the seed-0 cells within <0.13 pp). The GPU top-up it was blocked on is finished; T6's remaining step is now the
  CPU-only re-run of T2 (`score_joint_best.py`) on the CA/TX n=20 rundirs → then drop "provisional".
- [ ] **T7 (deferred, CPU) — camera-ready stats on joint-best per-fold pairs.** The paper's region-superiority
  (Wilcoxon 90% CI) + AL/AZ non-inferiority (TOST) were run on diag-best per-fold pairs. For the camera-ready
  joint-best row, re-run `superiority_wilcoxon.py` / `region_match_tost.py` on the joint-best per-fold series.
  Only Istanbul reg (margin +0.28→+0.19) is plausibly sensitive; direction is preserved.

## Deliverables
- `README.md` — landing + headline finding.
- `JOINT_BEST_RESULTS.md` — the corrected table, Δ analysis, decision memo, parity gates.
- `PROVENANCE.md` — the 18 rundirs (state, seed, PID, path) + how the cells aggregate.
- `AUDIT.md` — the independent adversarial-audit verdicts.
- `data/j1_results.json` + `score_all.py` — the machine-readable results + the reproducer.
