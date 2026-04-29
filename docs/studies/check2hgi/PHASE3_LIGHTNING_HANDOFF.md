# Phase 3 — Lightning Pod Handoff (concrete step-by-step)

**For the agent picking up Phase 3 in a fresh Lightning.ai pod with A100/H100.** Start here.

This guide assumes a clean Lightning Studio with no Drive access. Data comes from the user's gdown folders (links below). Repo state is on GitHub branch `worktree-check2hgi-mtl`.

---

## 0 · Quick context

Phase 2 closed the **STL** half of the substrate comparison at all 5 states (AL+AZ+FL+CA+TX) with CH16 confirmed at max-n=5 Wilcoxon p=0.0312 each, and CH15 reframed (substrate-equivalent on reg under matched MTL head, TOST non-inf at all 5 states).

**Phase 3 task:** close MTL CH18 at CA + TX. AL+AZ+FL already confirmed CH18 (MTL B3 wins on cat F1 and reg Acc@10 only when paired with Check2HGI substrate; HGI substrate breaks both heads). The remaining 4 cells are:

| Cell | Wall-clock estimate (A100 40 GB) |
|---|---|
| MTL B3 CA × C2HGI | ~30-40 min |
| MTL B3 CA × HGI (counterfactual) | ~30-40 min |
| MTL B3 TX × C2HGI | ~40-50 min |
| MTL B3 TX × HGI (counterfactual) | ~40-50 min |

Sequential ~3 h on single A100; parallel on 4× A100 ~50 min.

---

## 1 · Pod requirements

- **Lightning Studio with ≥1× A100 (40 GB)**. 4× A100 gives ~4× wall-clock speedup at ~same total cost.
- **Disk: ≥30 GB free** (~12 GB upstream parquets + ~10 GB run dirs + headroom).
- **Linux + Python 3.12 + CUDA 12.x.** PyTorch 2.8+cu128 confirmed working; older PyTorch + CUDA 11.x should also work.

T4 (15 GB) was already proven insufficient — see PHASE3_TRACKER.md §2 for OOM diagnostic.

---

## 2 · Bootstrap — clone + deps + data

Run one-shot bootstrap:

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl
bash scripts/setup_lightning_pod.sh
```

The bootstrap script does:
1. `pip install` core deps (cvxpy 1.6.4 pinned, ecos, geopandas, etc.)
2. `pip install` PyG wheels matched to the installed torch+CUDA version
3. `gdown` the 4 upstream data folders into `output/{check2hgi,hgi}/{california,texas}/`
4. Verify pre-flight checklist (parquets + transition matrices present)

If you need to gdown manually, the URLs are:

| Folder | gdrive ID |
|---|---|
| `output/check2hgi/california` | `1ZLL8FHPeO7I-3DEfVBogW1C1eFE76ttv` |
| `output/check2hgi/texas` | `1bLfFDEOM1BJ2ELoQUnd_qMXFpxGsZ7UF` |
| `output/hgi/california` | `1nMNaFgEEc1RwoH_o8_wasL9ENOkOCdKJ` |
| `output/hgi/texas` | `1g43xNSlJZBXStt3YGruOvCZTI-_WW4OQ` |

Manual gdown:
```bash
gdown --folder "https://drive.google.com/drive/folders/<ID>" -O output/<engine>/<state>
```

---

## 3 · Pre-flight verification

Before launching training, confirm:

```bash
# GPU
nvidia-smi --query-gpu=name,memory.total --format=csv
# expect: A100 (or H100), >=40 GB

# Repo branch
git branch --show-current  # → worktree-check2hgi-mtl

# Data on disk (8 parquets + 2 transition matrices = 10 files)
for state in california texas; do
  for engine in check2hgi hgi; do
    ls output/$engine/$state/{embeddings,region_embeddings}.parquet \
       output/$engine/$state/input/{next,next_region}.parquet \
       2>/dev/null | wc -l
  done
  ls output/check2hgi/$state/region_transition_log.pt
done
# expect: 4, 4, 4, 4 (counts), then transition .pt files exist
```

Quick smoke test (2 epochs × 1 fold on CA c2hgi MTL — ~5 min):

```bash
PYTHONPATH=src OUTPUT_DIR=$(pwd)/output \
python3 -u scripts/train.py \
  --task mtl --state california --engine check2hgi \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$(pwd)/output/check2hgi/california/region_transition_log.pt \
  --batch-size 2048 \
  --folds 1 --epochs 2 --seed 42 --no-checkpoints
```

If the smoke runs cleanly to completion (val_acc reported, no OOM), the pod is ready for full grid.

---

## 4 · Launch the grid

### 4.1 · Parallel (4× A100, ~50 min wall-clock)

```bash
nohup bash scripts/run_phase3_mtl_parallel.sh \
  > logs/phase3/orchestrator.log 2>&1 &
echo $! > logs/phase3/orchestrator.pid
```

Launches 4 background processes, one per GPU. Monitor:

```bash
tail -f logs/phase3/orchestrator.log logs/phase3/MTL_B3_*.log
```

### 4.2 · Sequential (1× A100, ~3-5 h wall-clock)

```bash
nohup bash scripts/run_phase3_mtl_grid.sh \
  > logs/phase3/orchestrator.log 2>&1 &
echo $! > logs/phase3/orchestrator.pid
```

Runs CA c2hgi → CA hgi → TX c2hgi → TX hgi sequentially with fail-fast (aborts if any cell exits non-zero).

### 4.3 · Single-cell (manual)

```bash
# scripts/run_phase3_mtl_cell.sh STATE ENGINE GPU_ID
bash scripts/run_phase3_mtl_cell.sh california check2hgi 0
```

---

## 5 · Post-run finalization

When all 4 cells finish:

```bash
python3 scripts/finalize_phase3.py
```

This:
1. Extracts per-fold metrics from each run dir's `folds/foldN_info.json::diagnostic_best_epochs.{next_category,next_region}.metrics`
2. Writes per-fold JSONs:
   - `docs/studies/check2hgi/results/phase1_perfold/{CA,TX}_{check2hgi,hgi}_mtl_{cat,reg}.json`
3. Runs paired tests:
   - `docs/studies/check2hgi/results/paired_tests/{california,texas}_mtl_{cat_f1,reg_acc10,reg_mrr}.json`
4. Prints cross-state CH18 status board.

If acceptance passes (cat F1 + reg Acc@10 paired Wilcoxon p < 0.05 per state), update:
- `PHASE3_TRACKER.md` — flip CA/TX MTL rows 🔴 → 🟢
- `research/SUBSTRATE_COMPARISON_FINDINGS.md` — append "Phase 3 — CH18 cross-state closure" section
- `PHASE2_TRACKER.md` — note that the CH18 outstanding work is now closed
- `CLAIMS_AND_HYPOTHESES.md` — CH18 status `confirmed AL+AZ+FL+CA+TX`

---

## 6 · Drive backup after closure

Run-dirs (`results/<engine>/<state>/mtlnet_*/`) and logs are gitignored. After Phase 3 closes, bundle and upload to Drive:

```bash
BUNDLE=/teamspace/studios/this_studio/phase3_drive_bundle_$(date +%Y-%m-%d)
mkdir -p $BUNDLE/results $BUNDLE/logs
cp -r results/{check2hgi,hgi}/{california,texas}/mtlnet_*  $BUNDLE/results/  # adjust paths if needed
cp -r logs/phase3 $BUNDLE/logs/

# Manifest
cat > $BUNDLE/MANIFEST.md << 'M'
Phase 3 MTL CH18 closure run dirs + training logs.
Per-fold + paired-test JSONs are in git (not in this bundle).
M

cd /teamspace/studios/this_studio
tar czf phase3_drive_bundle_$(date +%Y-%m-%d).tar.gz $(basename $BUNDLE)/
```

Download via Lightning Files panel, upload to Drive `mestrado_data/PoiMtlNet/phase3_archives/`.

Then commit the small artifacts (per-fold + paired tests + doc updates) to git:

```bash
git add docs/studies/check2hgi/results/phase1_perfold/{CA,TX}_*_mtl_*.json \
        docs/studies/check2hgi/results/paired_tests/{california,texas}_mtl_*.json \
        docs/studies/check2hgi/PHASE3_TRACKER.md \
        docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md \
        docs/studies/check2hgi/PHASE2_TRACKER.md \
        docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md
git commit -m "study(check2hgi): Phase 3 MTL CH18 closed at 5/5 states"
git push origin worktree-check2hgi-mtl
```

---

## 7 · Troubleshooting

### OOM despite A100

Memory creep can happen if PyTorch's allocator fragments. The orchestrators set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` automatically. If still OOMing:

1. Halve batch size: `MTL_BATCH_SIZE=1024 bash scripts/run_phase3_mtl_grid.sh`
2. Use 80 GB A100 if available

### Slow training (>2× expected)

Common causes:
- Data on slow filesystem (gdown'd to a non-NVMe path) — copy parquets to `/tmp/output_local/` and `export OUTPUT_DIR=/tmp/output_local`
- Non-deterministic CUBLAS — set `CUBLAS_WORKSPACE_CONFIG=:4096:8`

### Folds don't match between c2hgi and hgi

The MTL fold creator stratifies on `next_category` (which is identical between engines for the same state — see `PHASE2_TRACKER §7` Lightning verification note). If folds drift, run:

```bash
python3 scripts/study/freeze_folds.py --state california --engine check2hgi --task mtl_check2hgi
python3 scripts/study/freeze_folds.py --state california --engine hgi --task mtl_check2hgi
# then re-launch — both engines load the frozen indices
```

### Long runtime on T4

Don't. T4 is proven OOM at every batch size for CA. Use A100.

---

## 8 · Quick reference

**Orchestrator scripts (all under `scripts/`):**

| Script | Purpose |
|---|---|
| `setup_lightning_pod.sh` | Bootstrap fresh pod (deps + gdown data) |
| `run_phase3_mtl_cell.sh STATE ENGINE GPU_ID` | Launch one MTL cell, pinned to a specific GPU |
| `run_phase3_mtl_grid.sh` | Sequential CA→TX (single GPU) |
| `run_phase3_mtl_parallel.sh` | 4 cells in parallel across GPUs |
| `finalize_phase3.py` | Extract per-fold + run paired tests + status board |

**Canonical CLI (don't change):**

```bash
python3 -u scripts/train.py \
  --task mtl --state {california,texas} --engine {check2hgi,hgi} \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/<state>/region_transition_log.pt \
  --batch-size 2048 --folds 5 --epochs 50 --seed 42 --no-checkpoints
```

**Per-fold extraction key:** `diagnostic_best_epochs.next_category.metrics.{f1,accuracy}` and `diagnostic_best_epochs.next_region.metrics.{top10_acc_indist,mrr_indist,f1}`.

**Phase 3 closure when:** `paired_tests/{california,texas}_mtl_{cat_f1,reg_acc10}.json` both show Wilcoxon p < 0.05 with positive Δ̄.
