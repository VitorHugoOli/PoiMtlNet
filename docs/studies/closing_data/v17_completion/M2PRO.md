# M2 Pro handoff — simple, no-GPU analysis (v17 board)

> **Machine role (user rule): the M2 Pro does the simple analysis** — no training, no big GPU. Re-scoring saved
> logits/JSONs, the pre-registered stats, the CPU-only leak audit, and the doc/LaTeX/submission work. Everything here
> is CPU-bound and hours-scale. Recipe discipline (for any re-score): track [`README.md`](README.md).

> **Sequencing (does the M2 Pro wait for the A40/H100? — mostly, but not entirely).** **M1** (the re-score + stats
> payoff) MUST wait for **H1 (H100)** + **H2 (A40)** to land — it consumes their JSONs, so it is genuinely last on the
> critical path. **But M4 + M5** (STAN disclosure, stale-doc fixes, submission mechanics) **and M2 + M3** (A4-leak,
> bridging) **are independent — start them now, in parallel with the GPU work.** So: not idle until the GPUs finish;
> only M1 blocks on them.

## Queue

### M1 · n=20 re-score + the two pre-registered tests + drop "provisional"  — **the payoff of H1/H2**
Once H1 (CA/TX v17 MTL n=20) + H2 (STL cat ceiling n=20) land, on saved fold JSONs (no GPU):
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
