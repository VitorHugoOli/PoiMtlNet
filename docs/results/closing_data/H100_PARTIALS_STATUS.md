# H100 — final partials after studio restart (2026-06-25)

The H100 studio **restarted ~2026-06-25T06:40Z** (GPU detached, all detached jobs + the auto-committer
killed). This file records the final salvaged state. No GPU is available to resume on this box.

## CTLE-SC @ FL — ✅ PARTIAL captured (2/5 folds)
The sequential compare completed folds 0–1 before the restart. **Both axes captured** (cat extracted from
the run's fold reports, reg from the committed per-fold p1 reports):

| fold | cat macro-F1 | reg Acc@10 |
|---|---|---|
| 0 | 27.98 | 73.00 |
| 1 | 28.06 | 73.04 |
| **mean (n=2)** | **28.02** | **73.02** |

Supplementary (from the earlier parallel attempt, **reg only** — its cat raced and is unusable; f2 OOM-died):
f3 reg 71.81, f4 reg 74.54. → `baseline_compare/florida_ctle.json`.

**Reading (holds at n=2):** CTLE-SC cat ~28 ≪ Check2HGI-SC comparand **73.47** (the substrate drives next-category;
frozen CTLE can't compete), while reg Acc@10 ~73 ≈ comparand **72.71** (region is a near-tie, log_T-prior-driven,
substrate-independent). This is the §3-predicted two-axis story, stable across the two completed folds.
A full n=5 needs a GPU box (~2.5 h sequential).

## CA/TX ReHDM faithful — ✗ NOT captured (lost at restart)
Both 1-seed trains were ~epoch 1–2 when the box died; the faithful trainer writes its summary **only at
seed-end** (~8–12 h/seed, CPU-collaborator-mining-bound), so nothing landed. **Recoverable cheaply later:**
- ETLs are **built and on disk** — `output/baselines/rehdm/{california,texas}/inputs.parquet` + `vocab.json`
  (CA 6149 regions / 377k train traj; TX 4900 / 514k) → no re-ETL needed.
- The byte-identical collate optimization is already merged (PR #45).
- To finish: on a GPU box, `python -m research.baselines.rehdm.train --state {california,texas} --folds 1
  --epochs 50 --batch-size 128 --seed 42 --tag REHDM_{state}_1seed_50ep`. Remains "footnote infeasible" on the
  deadline unless a GPU box is free for ~8–12 h/seed.

## Lessons captured (for the parallel CTLE-SC tooling in PR #45)
`mac_baseline_compare.py --only-fold` + isolated `OUTPUT_DIR` parallelization is **not safe as-is**: (1) 5
concurrent reg-scorings OOM-killed 2 folds (each materialises a ~4.8 GB 4703-class logit on CPU → cap
concurrency to ~2–3), and (2) `run_cat` reads the *newest shared `results/` rundir* → a cat race across
concurrent folds (needs `RESULTS_ROOT` isolation per fold too, not just `OUTPUT_DIR`). Reg was fine (tag-isolated).
Sequential is the reliable path until those two fixes land.

## Net board impact
FL role-2 block is complete bar the CTLE-SC n=5 top-up (the n=2 partial here already supports the headline). CA/TX
ReHDM faithful stays a footnote candidate. Everything else from this H100 session (comparand, CTLE-E2E, A2,
CSLSL set-a fence, Istanbul stride-1 champion + n=20 + ceilings) is committed via PR #45 (merged).
