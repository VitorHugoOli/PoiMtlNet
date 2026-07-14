# M2 Pro handoff — simple, no-GPU analysis (v17 board)

> **Machine role (user rule): the M2 Pro does the simple analysis** — no training, no big GPU. Re-scoring saved
> logits/JSONs, the pre-registered stats, the CPU-only leak audit, and the doc/LaTeX/submission work. Everything here
> is CPU-bound and hours-scale. Recipe discipline (for any re-score): track [`README.md`](README.md).

> **Sequencing (updated 2026-07-11 — M1 is now FULLY UNBLOCKED).** The ceilings are n=20 at all 6 datasets and the
> v17 MTL is now n=20 at **all 6** — the **CA/TX** cells landed with **A1 done 2026-07-11 on the A40** (`docs/results/closing_data/catx_v17_n20/`;
> CA cat 77.052 ±0.006 / reg 65.693 ±0.017, TX cat 77.239 ±0.014 / reg 67.062 ±0.007 — confirms seed-0 within
> <0.13 pp). → **M1-full can run NOW at all 6.** (Prior state: M1-partial at AL/AZ/FL/Istanbul while CA/TX waited on
> A1, ex-H1.) M2–M5 are independent — anytime.

## Queue

### M1 · v17 stats: Wilcoxon + TOST + per-cell Holm — **M1-full NOW at all 6 (A1 done 2026-07-11)**
**NOW (M1-full):** on the committed n=20 artifacts (no GPU, no waiting): pair the v17 MTL (AL/AZ/FL
`perhead_lr_n20.md`; Istanbul `h3_istanbul/`; **CA/TX `docs/results/closing_data/catx_v17_n20/` — A1 done 2026-07-11
on the A40**) against the n=20 best-vs-best ceilings (`CEILINGS_N20_FINAL.md`,
**AZ = 56.43 corrected**) → per-cell Wilcoxon (cat superiority) + TOST (reg matches) + Holm across the full 6-dataset
family; write the verdicts to a new `v17_completion/stats_n20/` record. The remaining step is to re-run
`m1_stats_n20.py` on the CA/TX n=20 per-fold (verdicts not expected to move — the n=20 confirms seed-0 within <0.13 pp).
Original spec (kept):
- Re-score every §6.2 cell at n=20 via the matched scorer (`scripts/closing_data/h100_score_matched.py` /
  `r0_matched_rescore.py` read saved logits — no re-training).
- Re-run **superiority Wilcoxon** (`scripts/closing_data/superiority_wilcoxon.py`) + the state-level sign test, and
  **region TOST** (`scripts/closing_data/region_match_tost.py`); add **per-cell Holm** (clears 0.05 only at n=20).
- Update paper §5.3/§6.2 + Table 3: drop "n=5 (seed 0) provisional / no-per-cell-Holm / pooled-fold fallback / seed-0
  development-bias"; recompile. Keep the honest verdict verbs ("outperforms"/"matches" bound to their test).
- **This is the one analysis that lifts the paper from provisional to paper-grade** — do it the moment H1/H2 are in.

### M2 · A4 transductive-leak audit → CA/TX/Istanbul  — coverage (CPU, A4 is CPU≡MPS)
Extend the train-users-only rebuild audit (null at AL/AZ/FL: reg |Δ|≤0.33, cat |Δ|≤0.29) to ≥1 large state + Istanbul.
**Code-add first** (shapefiles): `Resources.TL_CA/TL_TX` TIGER tracts in `src/configs/paths.py` + the `SHAPEFILES`
dict in `scripts/pre_freeze_gates/a4_build.py`; Istanbul → the mahalle geojson. Then:
```bash
for f in 0 1 2 3 4; do python scripts/pre_freeze_gates/a4_build.py --state <state> --seed 0 --fold $f; done
python scripts/pre_freeze_gates/a4_eval.py --state <state> --seed 0
python scripts/pre_freeze_gates/a4_cat_eval.py --state <state> --seed 0
```
~3 h/fold (heavier at CA/TX). Acceptance: ≥1 large state |Δ|≲0.5 pp both axes → add a row to `pre_freeze_gates/A4_RESULTS.md`,
extend §5.2. Changes no verdict. Caveat: A4 tests the design_k substrate; note it if reporting for Istanbul.

### M3 · Bridging-metrics re-score (reg Acc@1/@5/MRR; cat Acc@1)  — coverage
Fill the 3 ladder rows in `articles/[mobiwac]/BRIDGING_METRICS.md` by re-scoring **saved logits** (no re-training).
**Blocker:** the k>10 metrics weren't serialized and the MTL/HMT-GRN per-fold logits are gitignored — if the run-machine
artifacts aren't reachable, this needs a short GPU re-forward (A40) first, then the M2 Pro does the scoring. Nice-to-have.

### M4 · STAN precision-mix disclosure (S1) + v4-collapse guard  — hygiene (the STAN track)
Faithful STAN is **done + citable** (AL 60.72 / AZ 49.86 / FL 72.99 / Istanbul 61.86), but the Table-3 cells mix
precision/version: **AL/AZ = v5_compiled fp32, FL = v6_opt bf16, Istanbul = v5_bf16c bf16** (same faithful recipe;
v6 = v5 + bit-identical perf opts; bf16 A/B quality-neutral ≤0.07 pp — defensible, matches the board precision policy).
- Add **one sentence** to `docs/baselines/next_region/stan.md` (and the Table 3 STAN footnote if room) disclosing the
  mix, so a reviewer isn't surprised.
- Add a **guard note** that the old v4/seed-42 STAN numbers (AL 34.46 / AZ 38.96, **below the Markov floor**) are a
  **superseded under-trained collapse — never cite** (they still sit in `faithful_stan_*_v4.json` on disk).

### M5 · Stale-doc fixes + submission mechanics  — doc/LaTeX/EDAS
- Stale-doc fixes (from the scrape): `PAPER_PLAN §10` says faithful-STAN-FL "in-flight" → **it's DONE (72.99)**;
  `RUN_MATRIX §0` + `p3_board.sh DEFAULT_STATES` include **georgia** → out of paper scope (pass explicit `--states`);
  Table 3 `--` markers (bare ReHDM-Istanbul vs `--`$^{\dagger}$ CA/TX) → standardize.
- Submission: EDAS Step-3 manuscript upload (paper #1571313639, 10-page fee variant); reconfirm the MobiWac deadline;
  apply the accepted Germano edits (`REVIEW_GERMANO.md`, 29 Accept + 29 Partial with concrete "Edit:"s).

## Not here
Any training / substrate build / n=20 run → H100 or A40. The M2 Pro touches only saved artifacts + prose + stats.
