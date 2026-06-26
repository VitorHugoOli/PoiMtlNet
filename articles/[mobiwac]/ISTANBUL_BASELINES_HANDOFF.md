# MobiWac 2026: Istanbul Baselines Handoff (for the executing agent)

> **Why this exists.** Table 3 (`src/tables/tbl3_results.tex`) has the Istanbul row filled for **our** STL and
> MTL numbers, but every **baseline** cell is `--` (Markov, POI-RGNN, ReHDM, STAN). This handoff is to run those
> four baselines **for Istanbul, locally on Apple-Silicon MPS** (user decision 2026-06-25), under the board
> protocol, and fill the row. No new architecture; these are existing baselines re-pointed at `--state istanbul`.
>
> **Read first:** [`BASELINE_HANDOFF.md`](BASELINE_HANDOFF.md) (the locked baseline plan + the three roles),
> [`BASELINE_AUDIT.md`](BASELINE_AUDIT.md), the per-baseline docs
> [`docs/baselines/next_category/poi_rgnn.md`](../../docs/baselines/next_category/poi_rgnn.md),
> [`docs/baselines/next_region/rehdm.md`](../../docs/baselines/next_region/rehdm.md),
> [`docs/baselines/next_region/stan.md`](../../docs/baselines/next_region/stan.md), and the board
> [`docs/studies/closing_data/RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md) §1/§4.

---

## 0 · Scope and the cells to fill

Istanbul, mahalle region taxonomy (520 regions, the board's PRIMARY def), seed 0 × 5 folds (n=5):

| Task | Baseline | Role | Code | Cost on MPS |
|---|---|---|---|---|
| next-category | **Markov-9-cat** floor | SOTA floor | `scripts/compute_markov_kstep_cat.py` | seconds (CPU) |
| next-category | **POI-RGNN** (faithful) | category SOTA | `research/baselines/poi_rgnn/` | ~minutes |
| next-region | **STAN** (substrate-bound) | region SOTA | `scripts/p1_region_head_ablation.py --heads next_stan` | ~10–30 min |
| next-region | **ReHDM** (faithful) | region SOTA | `research/baselines/rehdm/` | **LONG POLE (hours)** — see §4 |

Markov-1 **region** floor is DROPPED (BASELINE_HANDOFF §4) — do NOT add it.

---

## 1 · House rules / protocol (Istanbul-specific)

- **Region target = mahalle (520).** The board's Istanbul substrate + STL/MTL all use the mahalle def
  (`output/check2hgi/istanbul/`, the stride-1 GCN substrate). Do NOT use the H3 (2,585) variant for the table row.
- **Folds:** user-disjoint `StratifiedGroupKFold(5, seed=42)`, the SAME folds the board uses — so the baseline is
  paired with our STL/MTL. The substrate-bound runs inherit folds automatically; the faithful runs build their own
  ETL but MUST use the same 5-fold user-disjoint split (the scripts already do this — verify `--folds 5`).
- **Seed 0 × 5 folds (n=5).** Mark every cell `n=5 (seed 0) provisional`. The {1,7,100} top-up is post-deadline.
- **MPS execution (read the memory notes):**
  - **fp32 only.** MPS float16 autocast ADDS overhead and can NaN-collapse a fold; do NOT enable autocast. If a
    script exposes an AMP flag, set the disable gate (`DISABLE_AMP=1`). Verify healthy late best-epochs per fold.
  - **No `torch.compile`** on MPS (unvalidated; caution per memory).
  - Env for every run: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1`.
  - Wrap long runs in `caffeinate -i env ...` to prevent sleep-induced SIGBUS.
- **Leak-free per fold:** any graph / transition / collaborator statistic is computed on TRAIN rows of the fold
  only (the scripts already enforce this; ReHDM had a target-leak bug that was fixed — see `rehdm.md` audit).
- **Faithful = baseline's own raw inputs, no leak of our pretrained substrate into a SOTA row** (BASELINE_HANDOFF
  §3). The one substrate-bound exception is STAN (§4 below), by design.

Standard env block (paste once):
```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
ENV='PYTHONPATH=. DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data
     OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1'
```
(POI-RGNN/STAN docs use `PYTHONPATH=src`; ReHDM uses `PYTHONPATH=.` — keep each as written in its own doc.)

---

## 2 · Markov-9-cat (next-category floor) — do first, trivial

A `docs/results/P0/simple_baselines/istanbul/` directory already exists — **check it first**; Markov-9-cat may
already be computed (if so, just read the value and tabulate).

If not present, run (CPU, seconds):
```bash
env $ENV "$PY" scripts/compute_markov_kstep_cat.py --state istanbul   # confirm flags against the script's --help
```
(Cross-check `scripts/compute_simple_baselines.py` for the exact entrypoint/flags — one of these produces the
Markov-9-cat best-K floor used for the Gowalla states: AL 20.50 / AZ 23.92 / FL 29.74 / CA 27.59 / TX 28.67.)

**Acceptance:** a single macro-F1 floor value for Istanbul, well below our STL cat ceiling (53.20).

---

## 3 · POI-RGNN (next-category SOTA, faithful) — fast

Code: `research/baselines/poi_rgnn/`. AL was ~70 s on M4 Pro MPS, so Istanbul is ~minutes.
```bash
caffeinate -i env PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$PY" -m research.baselines.poi_rgnn.etl --state istanbul

caffeinate -i env PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$PY" -m research.baselines.poi_rgnn.train \
    --state istanbul --folds 5 --epochs 35 --tag FAITHFUL_POIRGNN_istanbul_5f35ep
```
**Caveat:** POI-RGNN uses category + hour-of-week + distance/duration buckets from the raw trail; it needs
lat/lon + datetime in `data/checkins/Istanbul.parquet` (present). If a column name differs for Massive-STEPS vs
Gowalla, the ETL will error — fix the column mapping in `research/baselines/poi_rgnn/etl.py`, do not silently
drop rows.
**Acceptance:** Istanbul macro-F1 far below our MTL cat (59.89) — for Gowalla our MTL beats POI-RGNN by +40…+48 pp;
expect the same order at Istanbul.

---

## 4 · Region SOTA: STAN and ReHDM (the two harder ones)

### 4a · STAN (substrate-bound) — the recommended Istanbul region-SOTA point

**Decision needed (flagged):** the Gowalla board uses STAN `stl_hgi` (STAN on the HGI substrate). **HGI is NOT
built for Istanbul** (`output/hgi/istanbul/` does not exist). Three options, in preference order:

1. **`stl_check2hgi` (RECOMMENDED).** Run STAN's bi-layer attention on the EXISTING Istanbul check2hgi substrate.
   This is the board substrate for Istanbul, needs no new embedding build, and is framed honestly as "the STAN
   architecture given our pretrained substrate." Command:
   ```bash
   caffeinate -i env PYTHONPATH=src DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
     "$PY" -u scripts/p1_region_head_ablation.py \
       --state istanbul --heads next_stan --folds 5 --epochs 50 \
       --input-type region --region-emb-source check2hgi \
       --tag STAN_CHECK2HGI_istanbul_5f50ep
   ```
   **Note the framing in the paper:** for Istanbul, STAN is reported on check2hgi (not HGI), because Istanbul has
   no HGI build. State this in the Table-B footnote so it is not silently mixed with the Gowalla `stl_hgi` column.
2. Build HGI for Istanbul first, then `--region-emb-source hgi` (matches Gowalla exactly, but is a new embedding
   build — only if HGI-Istanbul is wanted for other reasons).
3. STAN `faithful` (raw POI tokens, from scratch) — a strawman (below Markov-1 at AL) and slow; do NOT headline.
   Skip unless a reviewer demands a from-raw STAN.

**Comparability caveat:** the Istanbul STL region ceiling (74.80, `next_stan_flow` on check2hgi) is OUR dedicated
head on the same substrate. STAN `stl_check2hgi` is the published STAN architecture on the same substrate, so it
is a fair external point and should land at or below our ceiling. Keep them as separate columns; do not let STAN
`stl_check2hgi` be read as the ceiling itself.

### 4b · ReHDM (faithful) — the long pole

Code: `research/baselines/rehdm/`. Faithful is the SOTA-row variant (matches the Gowalla AL/AZ/FL ReHDM cells).
**Cost:** FL faithful was ~30 h at batch 32 on M4 Pro; Istanbul (270 k windows) is ~AZ-scale, expect several
hours. Use the FL trick (batch 128 + lr/max_lr scaled 4×, the linear-scaling rule, validated within 1σ on AL/AZ)
to cut wall-time, and `caffeinate`:
```bash
caffeinate -i env PYTHONPATH=. DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$PY" -m research.baselines.rehdm.etl --state istanbul

caffeinate -i env PYTHONPATH=. DATA_ROOT=$DATA_ROOT OUTPUT_DIR=$OUTPUT_DIR \
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
  "$PY" -u -m research.baselines.rehdm.train \
    --state istanbul --folds 5 --epochs 50 \
    --batch-size 128 --max-len 20 --max-intra 3 --max-inter 3 \
    --tag REHDM_istanbul_5f50ep
```
**Caveats:** (1) ReHDM's faithful target must be the **mahalle GEOID** (not quadkey-L10, which it keeps only as an
INPUT feature) — confirm the ETL maps the Istanbul target to the same mahalle taxonomy the board uses, or the row
is not comparable. (2) ReHDM NaN'd at higher LR on MPS in prior runs; if you hit NaN, drop max_lr (the doc used
5e-4) before widening anything. (3) If faithful proves infeasible on MPS within the deadline, the documented
fallback is `train_stl_study --engine check2hgi` (the substrate-bound ReHDM), but mark it clearly as a DIFFERENT
comparability class (substrate-bound, not faithful-from-raw) — do not blend it into the faithful column.
**Acceptance:** Istanbul Acc@10 below our MTL reg (74.28); for Gowalla our MTL beats ReHDM by +3.75…+11.6 pp.

---

## 5 · Comparands (Istanbul board numbers, for acceptance)

From `RESULTS_BOARD.md §1` (Istanbul, mahalle, stride-1, fp32, n=20 for ours):

| | our STL ceiling | our MTL champion |
|---|---|---|
| next-category (macro-F1) | **53.20** | **59.89** (+6.69) |
| next-region (Acc@10) | **74.80** | **74.28** (−0.52, matches within 2 pp) |

Sources: `docs/results/second_dataset/istanbul/istanbul_stride1_s0_stl_cat_ceiling.json`,
`docs/results/P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json`,
`docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json`.
Every baseline should sit **below** our MTL (and below or at our STL) on both tasks.

---

## 6 · Where the numbers go

1. **Commit the JSONs** (one per baseline, the scripts write them under `docs/results/baselines/` for faithful and
   `docs/results/P1/` for the STAN substrate-bound run; Markov under `docs/results/P0/simple_baselines/istanbul/`).
   Never cite a fold that NaN-collapsed; re-run that fold first.
2. **Per-baseline docs:** add the Istanbul row to the Source-JSONs tables in `poi_rgnn.md`, `rehdm.md`, `stan.md`,
   and to `docs/baselines/next_category/comparison.md` + `next_region/comparison.md`.
3. **Table 3** (`articles/[mobiwac]/src/tables/tbl3_results.tex`): replace the four `--` in the Istanbul row with
   the new values (Markov, POI-RGNN under next-category; ReHDM, STAN under next-region). Add the footnote that the
   Istanbul STAN point is on check2hgi (no HGI build) if you took option 4a-1.
4. **`RESULTS_BOARD.md §4`:** update the Istanbul baseline status from missing to done, with the JSON paths.
5. Recompile the paper (`pdflatex main` ×2) and confirm the Istanbul row renders.

---

## 7 · Acceptance checklist (the gate)
- [ ] Markov-9-cat Istanbul value (read existing or computed), below 53.20.
- [ ] POI-RGNN Istanbul 5f committed; macro-F1 ≪ our MTL 59.89.
- [ ] STAN Istanbul 5f committed (state which variant; default `stl_check2hgi`); Acc@10 ≤ our reg ceiling, < our MTL 74.28.
- [ ] ReHDM Istanbul 5f committed (faithful, mahalle target); Acc@10 < our MTL 74.28. (If faithful infeasible on
      MPS, substrate-bound fallback used AND labeled as a different class.)
- [ ] No fp16/NaN-collapsed fold cited; fp32 verified; healthy late best-epochs per fold.
- [ ] Table 3 Istanbul row filled; per-baseline docs + comparison.md + RESULTS_BOARD §4 updated; paper recompiles.
- [ ] Every cell marked `n=5 (seed 0) provisional`.

## 8 · Traps (each cost real time before)
- **HGI-Istanbul does not exist** → use STAN `stl_check2hgi`, not `stl_hgi` (§4a). Do not silently fall back to a
  zero/empty HGI substrate.
- **ReHDM target taxonomy** must be mahalle, not quadkey — quadkey is an input feature only (§4b).
- **MPS autocast** silently NaN-collapses folds → fp32, verify per-fold best-epochs are late, not epoch 4–5.
- **Massive-STEPS column names** may differ from Gowalla in the raw parquet → the faithful ETLs (POI-RGNN, ReHDM)
  may need a column-mapping fix; do not let them drop rows silently.
- **Faithful vs substrate-bound** are different comparability classes; never blend them in one column.
