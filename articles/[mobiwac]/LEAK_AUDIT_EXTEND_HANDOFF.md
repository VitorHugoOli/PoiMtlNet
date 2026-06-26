# MobiWac 2026: Transductive-leak audit — EXTEND coverage (handoff)

> **STATUS: NON-BLOCKING strengthening.** The transductive-leak audit (A4) is **already done at AL and FL** and the
> disclosure gate is **ON NULL on both axes** (region inflation ≈ 0, category inflation ≈ 0 on the in-coverage
> subset). The paper's §5.2 now cites the real measured numbers, so the leak rebuttal is **sourced and defensible
> today**. This handoff only **widens the coverage** (the reviewers noted the audit is on 2 of 6 datasets); it is a
> strengthening, not a submission blocker. **Source of truth:** [`docs/studies/pre_freeze_gates/A4_RESULTS.md`](../../docs/studies/pre_freeze_gates/A4_RESULTS.md).

## What the audit measures (and why the paper needs it)
The check-in-level representation is trained once over **all** check-ins, including the visits that later land in
validation folds (it is transductive). The rebuttal to the prior topological-leak rejection is: rebuild the
representation **per fold on training users only**, re-run both tasks, and show the downstream numbers barely move.
A4 is exactly that experiment.

**Already on disk (AL, FL, seed 0, 5 folds):**
| state | region: full / train-only / Δ | category (POI proxy, in-coverage): full / train-only / Δ | in-coverage |
|---|---|---|---|
| AL | 61.89 / 62.22 / **−0.33** | 29.07 / 28.78 / **+0.29** | 66.8% |
| FL | 69.97 / 70.08 / **−0.12** | 36.20 / 36.19 / **+0.00** | 86.9% |

Verdict (A4_RESULTS §Verdict): inflation ≈ 0 on both axes; one-paragraph defusal; headline numbers do NOT need
re-anchoring. The §5.2 prose already states these four numbers + the in-coverage caveat.

## The two real caveats the paper carries (state them, do not hide them)
1. **Category is a POI-level proxy on the in-coverage subset** (~67% of val rows at AL, ~87% at FL): val sequences
   whose input POIs are all train-covered, POI-level embeddings, `next_gru` macro-F1. The exact check-in-level §0.1
   setup cannot be measured on a transductive substrate (val-user check-ins at train-unseen POIs have no embedding —
   the inductive wall).
2. **The borrowed HGI Delaunay / POI2Vec spatial scaffolding is held fixed** in the rebuild (POI-spatial priors, not
   a check-in-transductive channel). So A4 bounds the **check-in-transductive** channel, which is the one the leak
   objection is about; the POI-spatial channel is the same for every learned baseline.

The residual (the contextual per-visit component + the cold-POI remainder) is bounded by the **inductive-Check2HGI
future work** (`docs/research/future_work.md §2`); the available evidence points to negligible inflation.

## EXTEND task — widen coverage (priority order)
House rules: seed 0, the SAME `StratifiedGroupKFold(seed)` split the reg eval uses; per-fold **train-only** log_T;
CPU is fine (the A4 eval validated CPU ≡ MPS). Each state's substrate rebuild is ~3 h (design_k, CPU). Commit the
result row into `A4_RESULTS.md` + a one-line finding; do NOT merge to main without the orchestrator's audit.

### 1. AZ — ready NOW (no code change)
AZ has a shapefile entry in `a4_build.py` (`SHAPEFILES = {alabama, arizona, florida}`), so it runs as-is. Per fold:
```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
for f in 0 1 2 3 4; do
  $PY scripts/pre_freeze_gates/a4_build.py --state arizona --seed 0 --fold $f      # ~3h/fold (CPU)
done
$PY scripts/pre_freeze_gates/a4_eval.py     --state arizona --seed 0   # reg Acc@10 full vs train-only
$PY scripts/pre_freeze_gates/a4_cat_eval.py --state arizona --seed 0   # cat macro-F1 POI-proxy, in-coverage
```
**Acceptance:** AZ region Δ and category Δ both within fold noise (≈ AL/FL, |Δ| ≲ 0.5 pp); add the AZ row to
`A4_RESULTS.md` and to §5.2 of the paper (extend "Alabama and Florida" → "Alabama, Arizona, and Florida").

### 2. CA / TX / Istanbul — need a shapefile entry first (modest code add)
`a4_build.py` only knows AL/AZ/FL tract shapefiles (`SHAPEFILES` dict + `Resources.TL_*`). To run CA/TX:
- Add `Resources.TL_CA` / `Resources.TL_TX` (TIGER tract shapefiles; CA/TX tracts) to `src/configs/paths.py` and the
  `SHAPEFILES` dict in `a4_build.py`. Istanbul uses the **mahalle** polygons (`output/check2hgi/istanbul/` provenance;
  not a TIGER tract) — point the build at the mahalle geojson the Istanbul substrate already uses.
- CA/TX HGI scaffolding must exist under `output/hgi/{california,texas}/` (built for the Tbl-2 cells, PR #52) — the
  build symlinks them; verify the `poi2vec_*` / `poi_embeddings` files are present before launching.
- Large-state rebuild is heavier (more check-ins); run one fold first as a smoke, confirm the inductive gap
  (regions absent from train) is small for coarse tracts, then fan out the 5 folds.
**Acceptance:** at least one large state (CA or TX) added with |Δ| within noise on both axes, so the null is shown
to hold at scale, not only at the small/mid states.

### 3. (Optional) multi-seed
A4 is seed-0 only, matching the n=5 board. A {1,7,100} top-up is only worth it if the headline goes multi-seed
post-deadline; otherwise seed-0 at more states is the better marginal coverage.

## After the runs: paper-doc updates
- `A4_RESULTS.md`: append the new state rows + a one-line verdict update.
- `articles/[mobiwac]/src/sections/05_setup.tex` §5.2: extend the state list in the leak sentence; if a large state
  shows a non-trivial Δ (it should not, per AL/FL), DISCLOSE it and re-anchor rather than hide it.
- Recompile (`pdflatex paper_skeleton ×2 + bibtex`); confirm §5.2 still renders and the page count is unchanged.

## Do NOT
- Do not call this a submission blocker — the gate is already ON NULL at AL+FL and §5.2 is sourced.
- Do not claim a check-in-level category measurement on the transductive substrate (it cannot be done — the cat
  number is the POI-proxy on the in-coverage subset; keep that caveat in the paper).
- Do not invent values for states not yet run; the four numbers in §5.2 are AL+FL only until the rows land.
