# Gap A — RunPod Handoff Prompt (CA + TX external baselines)

**Paste the section below into the RunPod Sonnet agent.** Self-contained brief; agent should NOT need to ask anything except for the gdrive folder IDs in §2.4. Branch is `worktree-check2hgi-mtl`. Don't push to `main`.

> **2026-04-30 closure on Lightning H100 (no RunPod needed; raw Gowalla parquets + TIGER 2022 tract shapefiles fetched into `data/checkins/` and `data/miscellaneous/` directly).**
>
> **Closed:**
> - All `next_category` floors (majority, top_k_popular, Markov-1-POI, Markov-K-cat) for CA + TX.
> - All `next_region` floors (majority, Markov-1-region) for CA + TX.
> - `next_region` STAN STL variants (`stl_check2hgi`, `stl_hgi`) for CA + TX.
> - `next_category` MHA+PE faithful for CA + TX.
>
> **Still running on Lightning H100 (5f×35ep / 5f×50ep, ~6-way GPU contention slows wall-clock):**
> - §4.2 POI-RGNN faithful (CA + TX)
> - §4.4 STAN faithful (CA + TX)
>
> This file is now historical. The whole RunPod path was rendered unnecessary because pyogrio + the TIGER URL pattern (`https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_<FIPS>_tract.zip`) made local execution feasible. Keep this file for the next reproduction; ignore the §3.4 ask-the-user step (it's no longer blocking).

---

## You are picking up the Check2HGI study to close Gap A

You are running on a RunPod instance with a single A100 (≥24 GB VRAM). Your job is to close **Gap A** of the Check2HGI study: produce the **California + Texas** numbers for the 5-state external-baseline tables that already cover Alabama, Arizona, and Florida. The repo, the baseline code, and the result schemas are all landed; you are executing, not designing.

You inherit a clean branch `worktree-check2hgi-mtl`. **Do not push to `main`.** Stay on the worktree branch throughout.

---

## 1 · Scope — exactly what to produce

Two tasks × two states × four baseline families. Cells marked ✅ already exist for AL+AZ+FL and **must not be re-run**.

### Task A — `next_category` (macro-F1 primary)

| Baseline | AL | AZ | FL | **CA** | **TX** | Cost per state |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Majority + Markov-1-POI floors | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | <5 min CPU |
| Markov-K-cat floors (k=1..9) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | <5 min CPU |
| **POI-RGNN faithful** (5f×35ep) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | ~25 min A100 (FL ref) |
| **MHA+PE faithful** (5f×11ep) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | ~10 min A100 (FL ref) |

### Task B — `next_region` (Acc@10 primary)

| Baseline | AL | AZ | FL | **CA** | **TX** | Cost per state |
|---|:-:|:-:|:-:|:-:|:-:|---|
| Markov-1-region floor | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | <5 min CPU |
| **STAN faithful** (5f×50ep) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | ~50 min A100 (FL ref) |
| **STAN stl_check2hgi** (5f×50ep) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | ~30 min A100 |
| **STAN stl_hgi** (5f×50ep) | ✅ | ✅ | ✅ | 🔴 **run** | 🔴 **run** | ~30 min A100 |
| ReHDM faithful (paper-protocol, 5 seeds) | ✅ | ✅ | 🔴 (~30 h, deferred) | ❌ **skip** | ❌ **skip** | — |
| ReHDM stl_check2hgi (5f×50ep, study protocol) | ✅ AL only | 🔴 | 🔴 | ❌ **skip** | ❌ **skip** | — |
| ReHDM stl_hgi (5f×50ep, study protocol) | ✅ AL,AZ | 🔴 | 🔴 | ❌ **skip** | ❌ **skip** | — |

> **ReHDM is OUT OF SCOPE for Gap A.** It was deferred at FL (~30 h faithful) and the AL+AZ STL data is sufficient for the comparison-table footnote. Only run ReHDM if the user explicitly tells you to — the "comparison.md" tables already document it as deferred.

**Total budget estimate (CA + TX, A100 24 GB):**
- Floors: ~30 min total (CPU)
- POI-RGNN ×2 + MHA+PE ×2: ~1.5 h
- STAN faithful ×2 + STL variants ×4: ~4 h
- **Grand total: ~6 h wall-clock + ~$3-5** on A100 24 GB pod.

CA is ~2× FL data; TX is ~3×. Assume FL × 2 for CA, FL × 3 for TX as worst-case scaling. STAN at TX may push toward ~1.5 h per cell → budget ~8 h to be safe.

---

## 2 · First read (~30 min, in this order)

1. `docs/studies/check2hgi/baselines/README.md` — protocol + status board + result-JSON schema
2. `docs/studies/check2hgi/baselines/next_category/comparison.md` — what the AL+AZ+FL columns look like; you'll fill the CA+TX columns the same way
3. `docs/studies/check2hgi/baselines/next_region/comparison.md` — same for next-region
4. `docs/studies/check2hgi/baselines/next_category/{poi_rgnn,mha_pe}.md` — per-baseline reproduction commands
5. `docs/studies/check2hgi/baselines/next_region/stan.md` — STAN reproduction command + variant CLI
6. `research/baselines/README.md` — code layout (each baseline owns its own ETL+model+train under `research/baselines/<name>/`)

After these you should know:
- The result JSONs land in `docs/studies/check2hgi/results/baselines/<tag>_<state>_<n>f<ep>ep_<TAG>.json`
- The aggregate state JSONs land in `docs/studies/check2hgi/baselines/next_<task>/results/<state>.json`
- The schema (v1) for state JSONs is in `baselines/README.md` §"`<state>.json` schema (v1)"
- All baselines use `StratifiedGroupKFold(seed=42)` on `userid`, stratified by `target_category` — same as the rest of the study

---

## 3 · Pod setup

### 3.1 · Bootstrap

```bash
# Clone + checkout
cd /workspace   # (or whatever the runpod default workdir is)
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl

# Python deps (Python 3.12 expected; check `python3 --version`)
pip install -U pip
pip install -r requirements.txt
# scikit-learn 1.8.0 is pinned — verify with: python3 -c "import sklearn; print(sklearn.__version__)"
# It must be 1.8.0+ (the StratifiedGroupKFold(shuffle=True) bugfix matters).

# PyG wheels matched to torch+CUDA — only needed if you'll run the substrate variants
# (STAN stl_check2hgi / stl_hgi do NOT use PyG; only the in-house MTL pipeline does).
# Check current torch+CUDA:
python3 -c "import torch; print(torch.__version__, torch.version.cuda)"
# If you DO need PyG (you don't for Gap A baselines, but just in case):
#   pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VER}+cu${CUDA_VER}.html

# Sanity check
nvidia-smi --query-gpu=name,memory.total --format=csv
git log -1 --oneline
```

### 3.2 · Required data layout

Each baseline either reads raw checkins (faithful variants) or substrate embeddings (STAN stl_*). You need both:

```
data/checkins/California.parquet                        # 142 MB raw Gowalla
data/checkins/Texas.parquet                             # 182 MB raw Gowalla
data/miscellaneous/tl_2022_06_tract_CA/                 # 50 MB TIGER shapefile bundle
data/miscellaneous/tl_2022_48_tract_TX/                 # 50 MB TIGER shapefile bundle
output/check2hgi/california/                            # ~2 GB substrate (for STAN stl_check2hgi)
output/check2hgi/texas/                                 # ~3 GB substrate
output/hgi/california/                                  # ~2 GB substrate (for STAN stl_hgi)
output/hgi/texas/                                       # ~3 GB substrate
```

Total disk: ~12 GB for both states. Verify free space: `df -h /workspace`.

### 3.3 · Fetch substrate outputs (existing helper)

`scripts/runpod_fetch_data.sh` already has the gdrive folder IDs for CA + TX substrates wired. From the repo root:

```bash
bash scripts/runpod_fetch_data.sh california
bash scripts/runpod_fetch_data.sh texas
# Mirrors output/check2hgi/california + output/check2hgi/texas to disk.

# For HGI substrate variants you also need the HGI outputs.
# These IDs are NOT yet wired into runpod_fetch_data.sh — use gdown directly:
mkdir -p output/hgi/california output/hgi/texas
cd output/hgi/california && gdown --folder 1nMNaFgEEc1RwoH_o8_wasL9ENOkOCdKJ && cd -
cd output/hgi/texas      && gdown --folder 1g43xNSlJZBXStt3YGruOvCZTI-_WW4OQ && cd -
```

### 3.4 · Fetch raw checkins + TIGER shapefiles (USER MUST PROVIDE IDs)

The raw checkins parquets and the TIGER tract shapefiles are NOT yet on the documented gdrive map. **Ask the user for these gdrive file/folder IDs, in this exact form:**

> "I need the gdrive IDs for: (a) `data/checkins/California.parquet`, (b) `data/checkins/Texas.parquet`, (c) the folder `data/miscellaneous/tl_2022_06_tract_CA/`, (d) the folder `data/miscellaneous/tl_2022_48_tract_TX/`. Could you upload these to Drive and paste the IDs?"

If the user's data is on a Lightning Studio they can also pipe through `rclone` from there to the RunPod — ask them which path they prefer. Once you have the IDs, fetch:

```bash
mkdir -p data/checkins data/miscellaneous/tl_2022_06_tract_CA data/miscellaneous/tl_2022_48_tract_TX
gdown <CA_PARQUET_ID> -O data/checkins/California.parquet
gdown <TX_PARQUET_ID> -O data/checkins/Texas.parquet
cd data/miscellaneous/tl_2022_06_tract_CA && gdown --folder <CA_SHAPEFILE_FOLDER_ID> && cd -
cd data/miscellaneous/tl_2022_48_tract_TX && gdown --folder <TX_SHAPEFILE_FOLDER_ID> && cd -

# Sanity-check sizes
ls -lh data/checkins/{California,Texas}.parquet
du -sh data/miscellaneous/tl_2022_*_tract_*
# Expected: CA ~142 MB, TX ~182 MB; each shapefile dir ~50 MB.
```

### 3.5 · Environment

```bash
export PYTHONPATH=src
export DATA_ROOT=$(pwd)/data
export OUTPUT_DIR=$(pwd)/output
mkdir -p logs/gap_a
```

Verify the path resolution before launching anything:

```bash
python3 -c "from configs.paths import DATA_ROOT, OUTPUT_DIR; print(DATA_ROOT); print(OUTPUT_DIR)"
# Should print the absolute paths above.
```

---

## 4 · Run order (sequential — keeps logs clean)

> Each command writes a `logs/gap_a/<TAG>.log` and a result JSON under `docs/studies/check2hgi/results/baselines/`. Do NOT background — run sequentially with `tee` so you can spot errors. Reuse the same TAG conventions the existing AL/AZ/FL JSONs use (see filenames in `docs/studies/check2hgi/results/baselines/`).

### 4.1 · Floors first (cheap; CPU-only; both states in one block)

```bash
# next_region Markov-1-region floor (existing aggregator script)
for STATE in california texas; do
  TAG="MARKOV1_REGION_${STATE^^}_5f"
  python3 scripts/compute_simple_baselines.py --state "$STATE" \
    2>&1 | tee "logs/gap_a/${TAG}.log"
done

# next_category Markov-K-cat floor
for STATE in california texas; do
  TAG="MARKOV_KCAT_${STATE^^}_5f_k1to9"
  python3 scripts/compute_markov_kstep_cat.py --state "$STATE" \
    2>&1 | tee "logs/gap_a/${TAG}.log"
done
```

Outputs land in `docs/studies/check2hgi/results/P0/simple_baselines/<state>/next_region.json` and `next_category_markov_kstep.json`. Verify both wrote.

### 4.2 · POI-RGNN faithful (5f × 35ep)

```bash
for STATE in california texas; do
  STAG="${STATE^^}"
  TAG="FAITHFUL_POIRGNN_${STAG,,}_5f35ep"

  # ETL — one pass per state (~5–10 min)
  python3 -m research.baselines.poi_rgnn.etl --state "$STATE" \
    2>&1 | tee "logs/gap_a/${TAG}_etl.log"

  # Train
  python3 -m research.baselines.poi_rgnn.train \
    --state "$STATE" --folds 5 --epochs 35 --batch-size 400 --seed 42 \
    --tag "$TAG" \
    2>&1 | tee "logs/gap_a/${TAG}.log"
done
```

Result JSON: `docs/studies/check2hgi/results/baselines/faithful_poi_rgnn_<state>_5f_35ep_${TAG}.json`. Cross-check that schema matches the existing AL/AZ/FL files.

### 4.3 · MHA+PE faithful (5f × 11ep)

```bash
for STATE in california texas; do
  STAG="${STATE^^}"
  TAG="FAITHFUL_MHAPE_${STAG,,}_5f11ep"

  python3 -m research.baselines.mha_pe.etl --state "$STATE" \
    2>&1 | tee "logs/gap_a/${TAG}_etl.log"

  python3 -m research.baselines.mha_pe.train \
    --state "$STATE" --folds 5 --epochs 11 --batch-size 400 --seed 42 \
    --tag "$TAG" \
    2>&1 | tee "logs/gap_a/${TAG}.log"
done
```

### 4.4 · STAN faithful (5f × 50ep)

```bash
for STATE in california texas; do
  STAG="${STATE^^}"
  TAG="FAITHFUL_STAN_${STAG,,}_5f50ep"

  # ETL needs the TIGER shapefile + raw checkins
  python3 -m research.baselines.stan.etl --state "$STATE" \
    2>&1 | tee "logs/gap_a/${TAG}_etl.log"

  python3 -m research.baselines.stan.train \
    --state "$STATE" --folds 5 --epochs 50 --batch-size 2048 --seed 42 \
    --tag "$TAG" \
    2>&1 | tee "logs/gap_a/${TAG}.log"
done
```

### 4.5 · STAN substrate variants (matched-head, study protocol, no faithful ETL)

These run STAN's architecture but ingest the substrate embeddings instead of raw inputs. They use `scripts/p1_region_head_ablation.py` (the in-house substrate-comparison harness already used by Phase 1+2).

```bash
for STATE in california texas; do
  STAG="${STATE^^}"
  for ENGINE in check2hgi hgi; do
    TAG="STL_${STAG}_${ENGINE}_stan_5f50ep"
    python3 scripts/p1_region_head_ablation.py \
      --state "$STATE" --heads next_stan \
      --folds 5 --epochs 50 --seed 42 --input-type region \
      --region-emb-source "$ENGINE" \
      --tag "$TAG" \
      2>&1 | tee "logs/gap_a/${TAG}.log"
  done
done
```

> **Note** — the `comparison.md` table calls these `stl_check2hgi` / `stl_hgi`. The result JSON lands in `docs/studies/check2hgi/results/P1/region_head_<state>_region_5f_50ep_${TAG}.json`. When you fill in the state JSONs, map this run to the `baselines.stan.stl_<engine>` block.

### 4.6 · OOM mitigation (TX especially)

If STAN OOMs on TX (~9.9K regions × 2048 batch on a 24 GB GPU), drop to `--batch-size 1024`. Wall-clock unchanged (samples/sec roughly constant). Don't drop further than 1024.

If `p1_region_head_ablation.py` OOMs, drop `--batch-size` further (e.g. `--batch-size 512`), though below 1024 the throughput penalty starts to matter. Read `--help` to confirm available flags.

---

## 5 · Aggregate the results

After all runs land:

### 5.1 · Build the per-state aggregate JSONs

For each state in `{california, texas}`, create or update:

- `docs/studies/check2hgi/baselines/next_category/results/<state>.json`
- `docs/studies/check2hgi/baselines/next_region/results/<state>.json`

following the v1 schema in `baselines/README.md`. Read the existing `florida.json` files for both tasks as your template — copy the structure, swap in the new numbers and the new `tag` / `date` / `source_json` fields.

Per-fold metric extraction:
- POI-RGNN, MHA+PE, STAN faithful → from the `aggregate.*_mean` / `aggregate.*_std` fields of the per-tag summary JSON
- STAN stl_* → from `docs/studies/check2hgi/results/P1/region_head_<state>_*.json::aggregate.*_mean`
- Floors → from `docs/studies/check2hgi/results/P0/simple_baselines/<state>/next_*.json`

### 5.2 · Refresh the `comparison.md` summary tables

Edit:

- `docs/studies/check2hgi/baselines/next_category/comparison.md` — fill the **CA** and **TX** columns of the macro-F1 table (top of file) and the Acc@1 table; append per-state Markov-K-cat detail rows
- `docs/studies/check2hgi/baselines/next_region/comparison.md` — fill the **CA** and **TX** columns of the Acc@10 summary; expand the per-baseline detail tables (STAN `faithful` / `stl_check2hgi` / `stl_hgi`)

The two existing FL columns are your reference for formatting (rounding, ± σ presentation). Don't change AL/AZ/FL numbers.

### 5.3 · Refresh the status board

In `docs/studies/check2hgi/baselines/README.md` §"Status board (live)", flip the corresponding 🔴 cells to ✅ for the rows you closed.

---

## 6 · Commit + push

Stage exactly the files you wrote/edited — do NOT use `git add -A` (large output dirs are gitignored anyway, but the habit prevents accidents).

```bash
git status --short
git add docs/studies/check2hgi/baselines/next_category/results/{california,texas}.json \
        docs/studies/check2hgi/baselines/next_region/results/{california,texas}.json \
        docs/studies/check2hgi/baselines/next_category/comparison.md \
        docs/studies/check2hgi/baselines/next_region/comparison.md \
        docs/studies/check2hgi/baselines/README.md \
        docs/studies/check2hgi/results/baselines/*_{california,texas}_*.json \
        docs/studies/check2hgi/results/P0/simple_baselines/{california,texas}/ \
        docs/studies/check2hgi/results/P1/region_head_{california,texas}_*.json

git commit -m "$(cat <<'EOF'
study(check2hgi): Gap A — CA + TX external baselines closed

POI-RGNN, MHA+PE faithful (5f×35/11ep) + STAN faithful + stl variants
(5f×50ep) at California + Texas. Markov-K-cat + Markov-1-region floors
at both states. Tables in baselines/{next_category,next_region}/comparison.md
filled for CA + TX columns; status board updated.

ReHDM remains deferred (paper-protocol faithful ~30h+; AL/AZ STL data
sufficient as comparison footnote per existing decision).
EOF
)"

git pull --rebase origin worktree-check2hgi-mtl   # remote may have moved
git push origin worktree-check2hgi-mtl
```

If pre-commit hooks fail: fix the underlying issue and create a NEW commit. **Don't** pass `--no-verify`. **Don't** amend.

---

## 7 · Drive backup for gitignored artefacts (optional but recommended)

Run dirs under `results/<engine>/<state>/` and the per-fold per-tag baseline JSONs under `docs/studies/check2hgi/results/baselines/` are gitignored if they exceed the limit (the latter are usually small enough to commit; the former are not). Bundle anything gitignored:

```bash
DATE=$(date +%Y-%m-%d)
BUNDLE=/workspace/gap_a_drive_bundle_$DATE
mkdir -p $BUNDLE/{logs,results}

cp -r logs/gap_a $BUNDLE/logs/
cp -r results/baselines/{california,texas}* $BUNDLE/results/ 2>/dev/null || true

cd /workspace
tar czf gap_a_drive_bundle_$DATE.tar.gz $(basename $BUNDLE)/
ls -lh gap_a_drive_bundle_$DATE.tar.gz
```

Tell the user the path; they download via the RunPod file UI and upload to Drive `mestrado_data/PoiMtlNet/gap_a_archives/`.

---

## 8 · Hard constraints — do NOT violate

- **Don't change canonical hparams.** Pinned: POI-RGNN 5f×35ep batch 400 lr 1e-3; MHA+PE 5f×11ep batch 400 lr 7e-4; STAN faithful 5f×50ep batch 2048 max_lr 3e-3 d_model 128 dropout 0.3; STAN stl_* same as `p1_region_head_ablation.py` defaults. Same as AL/AZ/FL — do NOT tune.
- **Don't push to `main`.** Stay on `worktree-check2hgi-mtl`.
- **Don't run ReHDM** (in or out of scope) without the user explicitly green-lighting it. Faithful is ~30 h × 2 states; not paper-blocking per the existing decision.
- **Don't overwrite AL/AZ/FL data.** You're appending CA + TX cells, not replacing existing rows.
- **Don't `git add -A`.** Stage explicitly (see §6).
- **Don't bypass pre-commit hooks** with `--no-verify`. If a hook fails, investigate and fix.
- **Don't skip the floor runs in §4.1.** They're cheap and the comparison tables need them as reference rows.
- **Don't rebuild ETL outputs unless the data changes.** ETL is idempotent and writes to `output/baselines/<name>/<state>/`; if you re-run the trainer, ETL skips re-doing existing files.

---

## 9 · Communication style

- Don't narrate environment exploration verbosely. One short status line per phase: "Bootstrap done", "POI-RGNN CA done — F1 = X.XX", etc.
- Post a single concise summary table at the end (one row per cell × state), not raw JSONs.
- If a baseline result looks anomalous (e.g. POI-RGNN CA F1 < 20% when AL was 23.80, or STAN faithful CA Acc@10 < 30% when AL was 34.46), **flag it loudly and stop** before committing. Likely causes: wrong shapefile, ETL failed silently, fold split mismatch, raw checkins file truncated.
- If anything blocks (OOM, missing shapefile, gdown fails), surface it immediately with the exact command + error. Don't retry blindly.

---

## 10 · Acceptance criteria — Gap A is closed when

1. ✅ `docs/studies/check2hgi/baselines/next_category/results/{california,texas}.json` exist, follow the v1 schema, contain `floors.{majority_class, markov_1_poi, markov_k_cat}` + `baselines.{poi_rgnn.faithful, mha_pe.faithful}` blocks with per-fold means + stds.
2. ✅ `docs/studies/check2hgi/baselines/next_region/results/{california,texas}.json` exist with `floors.{markov_1_region}` + `baselines.stan.{faithful, stl_check2hgi, stl_hgi}` blocks.
3. ✅ `comparison.md` summary tables show CA + TX columns populated (no more 🔴) for the rows you closed.
4. ✅ Status board in `baselines/README.md` reflects the new ✅ cells.
5. ✅ One commit on `worktree-check2hgi-mtl` summarising the work; pushed.
6. ✅ Final summary message to the user with a CA + TX cell-by-cell table (mean ± σ for each).

---

## 11 · The user

Vitor Hugo (vho2009@hotmail.com / vitor.oliveira@altbank.co). Brazilian time zone, replies primarily in English. Auto mode is enabled — proceed without asking for routine decisions, **but pause** on:

- A scientifically surprising result (see §9).
- The ReHDM scope question (don't run it without explicit OK).
- Any destructive action (force-push, branch delete, `rm -rf`).

If the gdown IDs in §3.4 aren't immediately available, ask once and wait — that's the only blocking question. CA+TX substrate IDs in §3.3 are already wired in.

Good luck. Branch `worktree-check2hgi-mtl`. The AL/AZ/FL columns of the comparison tables are your spec — match the format and you're done.
