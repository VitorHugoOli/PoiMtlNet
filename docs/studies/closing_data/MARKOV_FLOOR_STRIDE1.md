# Protocol-matched Markov-1 region floor (stride-1 board windowing)

**Status: COMPLETE 2026-07-18 (CPU-only, local M4 Pro). NOT yet integrated — paper/board
untouched; edits below are PROPOSALS pending author approval.**

## What this is

The paper's region floor (Table 3's "Markov floor" anchor, `markov_1step_region` in
`docs/results/P0/simple_baselines/<state>/next_region.json`) was computed on the frozen
NON-overlapping substrate windows (stride-9, MIN_SEQ=5, emit_tail=True), while every model
cell lives on the BOARD windowing (gated stride-1 overlap, window=9, MIN_SEQ=10,
emit_tail=False). `articles/[mobiwac]/src/sections/06_results.tex:106` discloses the mismatch
("computed under a non-overlapping windowing of the same data, indicative rather than
protocol-matched"). This recompute puts the SAME first-order region-level Markov floor on the
board windowing and folds so the caveat can be retired at camera-ready.

- **Script**: `scripts/closing_data/compute_markov_floor_stride1.py`
  (`scripts/compute_simple_baselines.py` was NOT modified; nothing under `docs/results/P0/`
  was touched).
- **Results**: `docs/results/closing_data/markov_floor_stride1/<state>.json` (six files).
- **Protocol**: windows = board gated stride-1 (window 9, stride 1, MIN_SEQ 10,
  emit_tail False; every window has 9 real visits, target = genuine next visit); folds =
  user-grouped `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, strat key =
  next_category via inverse `CATEGORIES_MAP`, groups = `str(userid)` — mirroring
  `compute_simple_baselines.py` exactly; baseline = train-fold-only P(next region | last
  region) transition counts, top-10 by `Counter.most_common` (train-insertion-order
  tie-break), global train top-10 fallback for unseen last-regions; val-fold Acc@K/MRR,
  mean ± std over the 5 folds. Metric semantics identical to the old floor JSONs.
- **Windows verified two ways per state**: (a) per-user content equivalence against the
  repo's canonical `data.inputs.core.generate_sequences` (the function the board builders
  call) — PASS at all six; (b) the mandatory count gate vs the paper's Table 1 Windows
  column (`tbl1_datasets.tex`) — see below.
- **poi→region mapping** (windowing-independent, so reusing frozen v11 artifacts is
  protocol-neutral): the check2hgi graph pickle
  `output/check2hgi/<state>/temp/checkin_graph.pt` for AL/AZ/FL/CA/Istanbul; for **Texas**
  (graph disk-reclaimed) the mapping was REBUILT from raw lat/lon +
  `data/miscellaneous/tl_2022_48_tract_TX` by replicating
  `research/embeddings/check2hgi/preprocess.py:_assign_regions` verbatim — the rebuild
  reproduces the substrate EXACTLY (160,938 POIs mapped, 0 dropped, 6,553 regions == Table 1;
  windows exact).

## Old vs new floor (Acc@10, fraction; ×100 for points)

| Dataset | Old floor (non-overlap) | **New floor (stride-1)** ± fold-std | Δ (pp) |
|---|---|---|---|
| Istanbul | .5252 | **.6506** ± .0038 | +12.54 |
| AL | .4701 | **.6226** ± .0311 | +15.25 |
| AZ | .4296 | **.5123** ± .0138 | +8.27 |
| FL | .6505 | **.7247** ± .0113 | +7.42 |
| TX | .5494 | **.6010** ± .0051 | +5.17 |
| CA | .5209 | **.5909** ± .0041 | +7.00 |

The floor moves up everywhere, as expected: stride-1 makes the last→next transition evidence
much denser, and the gated windowing guarantees a real last visit and a genuine next-visit
target in every window. (Full acc1/acc5/mrr aggregates and per-fold arrays are in the JSONs;
e.g. acc1 ranges 28.95 (AZ) to 50.53 (FL).)

## New margin vs the paper's joint region cells (points, Acc@10)

Joint cells (board §1): Ist 75.44 / AL 69.80 / AZ 59.56 / FL 77.42 / TX 67.07 / CA 65.69.

| Dataset | Joint | New floor | **Margin (joint − floor)** | Old-floor margin |
|---|---|---|---|---|
| Istanbul | 75.44 | 65.06 | **+10.38** | +22.92 |
| AL | 69.80 | 62.26 | **+7.54** | +22.79 |
| AZ | 59.56 | 51.23 | **+8.33** | +16.60 |
| FL | 77.42 | 72.47 | **+4.95** | +12.37 |
| TX | 67.07 | 60.10 | **+6.97** | +12.13 |
| CA | 65.69 | 59.09 | **+6.60** | +14.60 |

The joint model still clears the protocol-matched floor at **all six datasets**, by
**+4.9 to +10.4 points** (was "+12 to +23" over the non-matched floor). The floor itself now
spans **51 to 72** Acc@10 (was "43 to 65"). Note AL: the new floor (62.26) also sits above the
old AL joint-vs-floor framing's floor by a wide margin, and the joint margin (+7.54) remains
comfortably positive; FL is the tightest at +4.95.

## Sanity-gate outcome per state (windows vs Table 1)

| Dataset | Computed | Table 1 | Ratio | Verdict |
|---|---|---|---|---|
| AL | 96,326 | 96,326 | 1.0000 | **EXACT** |
| AZ | 200,895 | 200,895 | 1.0000 | **EXACT** |
| FL | 1,274,418 | 1,274,418 | 1.0000 | **EXACT** |
| CA | 2,925,466 | 2,925,466 | 1.0000 | **EXACT** |
| TX | 3,830,414 | 3,830,414 | 1.0000 | **EXACT** (rebuilt tract mapping; 0 unmapped POIs) |
| Istanbul | 270,217 | 271,666 | 0.9947 | PASS (−0.53%), residual fully attributed ↓ |

**Istanbul residual (−1,449 windows, 0.53%)**: the local Istanbul substrate
(`output/check2hgi/istanbul/` — graph metadata AND `embeddings.parquet` both give 270,217)
is the pre-rebuild stride-1 base; Table 1's 271,666 is the A40 `dk_ovl` substrate REBUILD
(tbl1_datasets.tex hidden comment: "Windows updated 2026-07-08 to the rebuilt dk_ovl
substrate count (271,666; was 270,217 on the old stride-1 base -- h3_istanbul/RESULTS.md)";
`STATS_T1.md` documents 270,217 for this exact corpus). Corpus totals match Table 1
EXACTLY (462,615 check-ins / 23,694 users / 29,816 POIs / 520 mahalles); only the per-user
window distribution of the A40 rebuild differs slightly and is not reproducible from the
artifacts on this machine. Impact on a count-based floor: negligible relative to the ±0.38 pp
fold-std; the Istanbul cell is honest but carries this one-line vintage caveat.

## Proposed replacement for `06_results.tex:106-109` (PROPOSAL ONLY — no articles/ edit made)

Current:

> A simple first-order Markov region floor reaches $43$ to $65$ Acc@10 across the datasets
> (computed under a non-overlapping windowing of the same data, indicative rather than
> protocol-matched); the joint model exceeds it by $12$ to $23$ points at all six datasets.

Proposed:

> A simple first-order Markov region floor, recomputed under the same sliding-window
> protocol and fold splits as our models, reaches $51$ to $72$ Acc@10 across the datasets;
> the joint model exceeds it by $5$ to $10$ points at all six datasets.

(If the author prefers exact bounds over rounded ones: "by $4.9$ to $10.4$ points". FL is
the tightest margin at $+4.95$; rounding it to "5" is a 0.05-point overstatement — the
"about $5$ to $10$" phrasing avoids that if preferred. The category Markov-9 floor row is
untouched: it stays deliberately row-matched to POI-RGNN's non-overlap ETL.)

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/closing_data/compute_markov_floor_stride1.py
# or --state <alabama|arizona|istanbul|florida|california|texas>
```

Runtime: seconds per small state, ~13 s CA, ~19 s TX (CPU-only, RAM-bounded; one state per
process). Sources per state are recorded in each JSON (`source` block), incl. the Texas
mapping-rebuild provenance.
