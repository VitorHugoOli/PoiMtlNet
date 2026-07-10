# J1 — Joint-checkpoint scoring runbook (A40)

> ✅ **EXECUTED 2026-07-09 → [`../../docs/studies/closing_data_v2/`](../../docs/studies/closing_data_v2/).**
> All 18 v17/`dk_ovl` MTL rundirs scored (CPU-only). **Finding: the served single checkpoint reproduces
> Table 3 within ≤ 0.06 pp cat / ≤ 0.11 pp reg on every dataset (largest 0.051 / 0.107, both at AZ) — no verdict changes.** Category still beats
> everywhere; region still beats at Istanbul/FL/TX/CA (Istanbul margin +0.28→+0.19, still positive, 20/20 folds)
> and matches at AL/AZ (the AL tail risk did not materialize: −0.31→−0.41, far from the −2 pp bound). Three
> self-audits (diag-parity 18/18, joint-epoch 90/90, paper-parity 6/6) + a 4-agent independent audit all pass.
> The corrected side-by-side table + decision memo: [`../../docs/studies/closing_data_v2/JOINT_BEST_RESULTS.md`](../../docs/studies/closing_data_v2/JOINT_BEST_RESULTS.md).
> CA/TX stay seed-0 provisional (n=20 blocked on the A1 GPU top-up, not on J1).

> **Why this doc exists.** The paper's Table 3 MTL cells are **per-task diagnostic-best** (category at
> its F1-best epoch, region at its own Acc@10-best epoch — two different checkpoints per fold; proof:
> AL s0 cat epochs [17,17,16,17,17] vs reg [27,25,28,22,37]). Training also saves a **single joint
> checkpoint** per fold (the `geom_simple` selector, `sqrt(cat macro-F1 × reg Acc@10-indist)` on
> validation), but nobody ever scored it. **Author decision (2026-07-09):** the submission carries NO
> rendered mention of this (the §6.2 disclosure sentence was removed); the joint-best numbers are
> produced on the A40 for the **response letter and camera-ready**, where the fix lands if the paper
> passes. Full background: [`../../docs/studies/closing_data/JOINT_BEST_SCORING.md`](../../docs/studies/closing_data/JOINT_BEST_SCORING.md).

## What to run (CPU-only, minutes total — no retraining)

The scorer recovers joint-best numbers post hoc from each rundir's per-epoch val CSVs
(`metrics/fold*_next_{category,region}_val.csv`), mirroring the training-time gate exactly
(v17 board gate = `--min-best-epoch 0`, the default). It prints joint-best AND diag-best side by side
and writes a committable `joint_best_score.json` sidecar into the rundir.

```bash
cd ~/PoiMtlNet   # A40 checkout
# One invocation per board rundir (paths per the A40's results/ layout — locate with the
# v17_completion docs; the rundir families are:)
#   1. perhead_lr_n20 AL/AZ/FL   (4 seeds × 5 folds each — docs/studies/closing_data/perhead_lr_n20.md)
#   2. catx_v17_seed0_5f CA/TX   (seed-0 × 5 folds     — docs/studies/closing_data/catx_v17_seed0_5f/)
#   3. h3_istanbul               (4 seeds × 5 folds     — docs/studies/closing_data/v17_completion/h3_istanbul/)
PYTHONPATH=src .venv/bin/python scripts/closing_data/score_joint_best.py <rundir> --tag <state>_<seed>
# Then commit the joint_best_score.json sidecars (C28 convention: commit result JSONs).
```

## What to check in the output (before quoting any number)

1. **AL region is the tail risk.** Diag-best AL region is −0.31 vs the dedicated ceiling with a TOST
   CI of (−0.48, −0.14). If the joint-best region drop is ~1 pp (the non-board smoke test saw
   −1.16 reg / −0.69 cat on a legacy run), the AL joint-best delta could approach the −2 pp TOST
   bound. **Score first, decide with numbers in hand** — never quote a joint-best number that breaks
   a claimed match without flagging it to the author.
2. Verify the scorer's `joint_epoch` per fold matches the epoch of the actually-saved joint
   checkpoint (the sidecar carries the cross-check when `standard_scores.json` artifacts exist).
3. The selector uses reg **Acc@10-indist** (as training did); reported reg = `indist × (1 − ood_frac)`
   at that epoch — same full-catalog convention as the paper.

## Where the results go  (✅ done — see the EXECUTED banner above)

- Sidecars: `joint_best_score.json` written into all 18 rundirs' result trees. **Note:** those live under
  `results/check2hgi_dk_ovl/` which is **gitignored** here, so the committable record is the aggregate
  `docs/studies/closing_data_v2/data/j1_results.json` (+ the reproducer `score_all.py`), not the per-rundir sidecars.
- Aggregate + decision memo: **[`../../docs/studies/closing_data_v2/JOINT_BEST_RESULTS.md`](../../docs/studies/closing_data_v2/JOINT_BEST_RESULTS.md)**
  (the corrected table lives in the new `closing_data_v2` study; a §Results pointer was also added to
  `JOINT_BEST_SCORING.md`).
- **Response letter:** the honest line is on record in a hidden comment at
  `src/sections/06_results.tex` (search "Response-letter note"): a single joint checkpoint per fold
  exists; numbers can be provided; never claim they match the diag-best cells.
- **Camera-ready (if accepted):** add the joint-checkpoint numbers to Table 3 (extra row or column)
  and restore a scoring-convention sentence in §6.2; the CA/TX n=20 top-up (A1) and this land
  together in one renumbering pass.

## Standing guard for future work

Every new MTL run now auto-exports `metrics/fold{N}_standard_scores.json` with BOTH conventions
(`src/tracking/scoring.py` — the single source of truth). **Never report a number without naming its
epoch-selection convention.**
