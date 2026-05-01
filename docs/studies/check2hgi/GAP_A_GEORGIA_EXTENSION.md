# Gap A — Georgia extension (H100, paste-in)

**Context.** CA + TX baselines are closing on the Lightning H100 (see `GAP_A_RUNPOD_HANDOFF_PROMPT.md`'s 2026-04-30 closure note). This file extends the same flow to **Georgia** so all five Gap A states (AL, AZ, FL, CA, TX, **GA**) land in the comparison tables.

GA is small (smaller than AL/AZ in checkin count) — wall-clock is 30–60 % of FL per cell. Run it on the same H100 instance once a slot frees.

Substrate gdrive folder is already wired in `scripts/runpod_fetch_data.sh georgia` (folder ID `1v5xiJRzIQfMT8yk-J11sax5uf7Mct5vo`). HGI substrate folder for GA is **not** wired yet — fetch from the same Drive bucket as CA/TX (ask user for the HGI/GA folder ID if absent).

GA TIGER FIPS = **13**. Shapefile URL: `https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_13_tract.zip`.

---

## 1 · Bootstrap deltas (vs CA/TX handoff §3)

```bash
# Substrate (check2hgi)
bash scripts/runpod_fetch_data.sh georgia

# HGI substrate — replace <GA_HGI_FOLDER_ID> when known
mkdir -p output/hgi/georgia
cd output/hgi/georgia && gdown --folder <GA_HGI_FOLDER_ID> && cd -

# Raw checkins + TIGER shapefile
mkdir -p data/checkins data/miscellaneous/tl_2022_13_tract_GA
gdown <GA_PARQUET_ID> -O data/checkins/Georgia.parquet   # ask user if not on Drive yet
cd data/miscellaneous/tl_2022_13_tract_GA && \
  curl -O https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_13_tract.zip && \
  unzip -o tl_2022_13_tract.zip && cd -
```

Same env exports as the parent handoff (`PYTHONPATH=src`, `DATA_ROOT`, `OUTPUT_DIR`, `mkdir -p logs/gap_a`).

---

## 2 · Run order — paste block

```bash
STATE=georgia
STAG=GEORGIA

# 2.1 — Floors (CPU, <5 min each)
python3 scripts/compute_simple_baselines.py --state "$STATE" \
  2>&1 | tee "logs/gap_a/MARKOV1_REGION_${STAG}_5f.log"

python3 scripts/compute_markov_kstep_cat.py --state "$STATE" \
  2>&1 | tee "logs/gap_a/MARKOV_KCAT_${STAG}_5f_k1to9.log"

# 2.2 — POI-RGNN faithful (5f×35ep)
TAG="FAITHFUL_POIRGNN_${STATE}_5f35ep"
python3 -m research.baselines.poi_rgnn.etl --state "$STATE" \
  2>&1 | tee "logs/gap_a/${TAG}_etl.log"
python3 -m research.baselines.poi_rgnn.train \
  --state "$STATE" --folds 5 --epochs 35 --batch-size 400 --seed 42 \
  --tag "$TAG" 2>&1 | tee "logs/gap_a/${TAG}.log"

# 2.3 — MHA+PE faithful (5f×11ep)
TAG="FAITHFUL_MHAPE_${STATE}_5f11ep"
python3 -m research.baselines.mha_pe.etl --state "$STATE" \
  2>&1 | tee "logs/gap_a/${TAG}_etl.log"
python3 -m research.baselines.mha_pe.train \
  --state "$STATE" --folds 5 --epochs 11 --batch-size 400 --seed 42 \
  --tag "$TAG" 2>&1 | tee "logs/gap_a/${TAG}.log"

# 2.4 — STAN faithful (5f×50ep)
TAG="FAITHFUL_STAN_${STATE}_5f50ep"
python3 -m research.baselines.stan.etl --state "$STATE" \
  2>&1 | tee "logs/gap_a/${TAG}_etl.log"
python3 -m research.baselines.stan.train \
  --state "$STATE" --folds 5 --epochs 50 --batch-size 2048 --seed 42 \
  --tag "$TAG" 2>&1 | tee "logs/gap_a/${TAG}.log"

# 2.5 — STAN substrate variants (matched-head)
for ENGINE in check2hgi hgi; do
  TAG="STL_${STAG}_${ENGINE}_stan_5f50ep"
  python3 scripts/p1_region_head_ablation.py \
    --state "$STATE" --heads next_stan \
    --folds 5 --epochs 50 --seed 42 --input-type region \
    --region-emb-source "$ENGINE" \
    --tag "$TAG" 2>&1 | tee "logs/gap_a/${TAG}.log"
done
```

**OOM mitigation.** GA is smaller than FL — defaults should fit. If `next_stan` OOMs on the HGI variant, drop to `--batch-size 1024`.

---

## 3 · Aggregate

After all runs land, mirror the CA/TX aggregation (parent handoff §5):

- Create `docs/studies/check2hgi/baselines/next_category/results/georgia.json`
- Create `docs/studies/check2hgi/baselines/next_region/results/georgia.json`
- Add **GA** column to both `comparison.md` summary tables (next to TX)
- Flip the GA cells in `baselines/README.md` §"Status board" from 🔴 → ✅

Use `florida.json` / `california.json` as your schema reference.

---

## 4 · Hard constraints

- Don't change canonical hparams (POI-RGNN 5f×35ep b400; MHA+PE 5f×11ep b400; STAN 5f×50ep b2048).
- Stay on `worktree-check2hgi-mtl`; don't push to `main`.
- Don't run ReHDM (out of scope per parent handoff §1).
- Don't touch AL/AZ/FL/CA/TX numbers when adding the GA column — append, don't replace.
- Stage explicitly (no `git add -A`); single commit summarising GA.

## 5 · Acceptance

GA closed when: both `georgia.json` aggregate files exist and validate against schema v1, comparison tables show a populated GA column, status board shows ✅ for the GA cells in the rows you ran, one commit + push to `worktree-check2hgi-mtl`, summary table posted.
