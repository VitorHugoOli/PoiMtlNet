# Phase 3 Scope D — Lightning Pod Handoff (concrete step-by-step)

**For the agent picking up Phase 3 in a fresh Lightning.ai pod with A100/H100.** Start here.

This guide assumes a clean Lightning Studio with no Drive access. Data comes from the user's gdown folders. Repo state is on GitHub branch `worktree-check2hgi-mtl`.

---

## 0 · Quick context

Phase 2 closed STL cat (`next_gru`) at 5/5 states (CH16 confirmed) + reframed CH15 with reg STL `next_getnext_hard` (TOST non-inf). But Phase 2 reg STL + MTL data carried **F44 transition-matrix leakage** (full-dataset prior includes val-fold edges).

**Phase 3 task — Scope D: re-run all reg STL + MTL cells with leakage-free per-fold transition matrices, across all 5 states**, so the substrate-comparison evidence chain is end-to-end clean and consistent.

| Phase 3 step | Cells | A100 40 GB ETA |
|---|---|---|
| 1. Per-fold transition build (CPU) | 25 matrices (5 states × 5 folds) | ~25 min total |
| 2a. Reg STL re-run (`_pf` suffix) | 10 cells (5 states × 2 engines) | 4× GPU: ~50 min · 1× GPU: ~225 min |
| 2b. MTL B3 re-run (`_pf` suffix) | 10 cells | 4× GPU: ~110 min · 1× GPU: ~350 min |
| 3. Finalize (extract + paired tests) | CPU | ~10 min |

**Total on 4× A100 (40 GB): ~2.7 hours wall-clock, ~$13 total cost.**

---

## 1 · Pod requirements

- **Lightning Studio with ≥1× A100 (40 GB).** 4× A100 gives ~3× wall-clock speedup at near-flat total cost.
- **Disk: ≥40 GB free** (~25 GB upstream parquets across 5 states + ~15 GB run dirs).
- **Linux + Python 3.12 + CUDA 12.x.** PyTorch 2.8+cu128 confirmed working.

T4 (15 GB) is **not enough** — see PHASE3_TRACKER §2.

---

## 2 · Bootstrap — clone + deps + data

### 2.1 · Clone

```bash
cd /teamspace/studios/this_studio
git clone https://github.com/VitorHugoOli/PoiMtlNet.git
cd PoiMtlNet
git checkout worktree-check2hgi-mtl
```

### 2.2 · Bootstrap with all 5 states

The setup script accepts gdrive IDs as env vars. CA + TX IDs are hardcoded as defaults; AL/AZ/FL must be passed explicitly:

```bash
STATES="alabama arizona florida california texas" \
  AL_C2HGI_GDID=<alabama c2hgi folder id> \
  AL_HGI_GDID=<alabama hgi folder id> \
  AZ_C2HGI_GDID=<arizona c2hgi folder id> \
  AZ_HGI_GDID=<arizona hgi folder id> \
  FL_C2HGI_GDID=<florida c2hgi folder id> \
  FL_HGI_GDID=<florida hgi folder id> \
  bash scripts/setup_lightning_pod.sh
```

The bootstrap:
1. `pip install` core deps (cvxpy 1.6.4 pinned, ecos, geopandas, etc.)
2. `pip install` PyG wheels matched to torch+CUDA
3. `gdown` 10 upstream folders (5 states × 2 engines) into `output/{check2hgi,hgi}/{state}/`
4. Verify pre-flight checklist (40 parquets + 5 transition matrices on disk)

If you only want to run a subset of states, omit the corresponding env vars; the bootstrap will skip what's missing and report which cells will be runnable.

### 2.3 · Known gdrive folder IDs (provided by user 2026-04-29)

| Folder | gdrive ID |
|---|---|
| `output/check2hgi/california` | `1ZLL8FHPeO7I-3DEfVBogW1C1eFE76ttv` |
| `output/check2hgi/texas` | `1bLfFDEOM1BJ2ELoQUnd_qMXFpxGsZ7UF` |
| `output/hgi/california` | `1nMNaFgEEc1RwoH_o8_wasL9ENOkOCdKJ` |
| `output/hgi/texas` | `1g43xNSlJZBXStt3YGruOvCZTI-_WW4OQ` |
| `output/check2hgi/alabama` | (ask user) |
| `output/hgi/alabama` | (ask user) |
| `output/check2hgi/arizona` | (ask user) |
| `output/hgi/arizona` | (ask user) |
| `output/check2hgi/florida` | (ask user) |
| `output/hgi/florida` | (ask user) |

If user can't provide AL/AZ/FL gdown URLs, run Phase 3 only on `STATES="california texas"` and document that AL/AZ/FL leakage-free is deferred.

---

## 3 · Pre-flight verification

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
git branch --show-current   # → worktree-check2hgi-mtl

# Parquets + transition matrices for all 5 states
for state in alabama arizona florida california texas; do
  for engine in check2hgi hgi; do
    ls output/$engine/$state/{embeddings,region_embeddings}.parquet \
       output/$engine/$state/input/{next,next_region}.parquet \
       2>/dev/null | wc -l
  done
done
# expect: each "wc -l" prints 4 (4 parquets per state×engine)

# Smoke test (2-epoch 1-fold CA c2hgi MTL on a single GPU, ~10 min)
PYTHONPATH=src OUTPUT_DIR=$(pwd)/output \
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -u scripts/train.py \
  --task mtl --state california --engine check2hgi \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$(pwd)/output/check2hgi/california/region_transition_log.pt \
  --batch-size 2048 --folds 1 --epochs 2 --seed 42 --no-checkpoints
```

If the smoke runs cleanly, the pod is ready.

---

## 4 · Run Phase 3

### Step 1: Per-fold transition matrices (CPU, ~5 min per state)

```bash
bash scripts/build_phase3_per_fold_transitions.sh
```

This builds `output/check2hgi/<state>/region_transition_log_fold{1..5}.pt` from `StratifiedGroupKFold(seed=42)` train-only edges. Idempotent — skips states already complete.

### Step 2: Full grid (parallel)

```bash
nohup bash scripts/run_phase3_parallel.sh > logs/phase3/orchestrator.log 2>&1 &
echo $! > logs/phase3/orchestrator.pid

# Monitor
tail -f logs/phase3/orchestrator.log
```

Runs all reg STL cells (10), then all MTL cells (10), parallelized across detected GPUs.

If you want a smaller scope:
```bash
STATES="california texas" bash scripts/run_phase3_parallel.sh
```

### Step 3: Finalize

```bash
python3 scripts/finalize_phase3.py
```

Outputs:
- `docs/studies/check2hgi/results/phase1_perfold/<STATE>_<engine>_{reg_gethard_pf_5f50ep, mtl_{cat,reg}_pf}.json`
- `docs/studies/check2hgi/results/paired_tests/<state>_{reg_acc10, reg_mrr, mtl_cat_f1, mtl_reg_acc10, mtl_reg_mrr}_pf.json`
- Cross-state CH15 + CH18 status board (printed)

---

## 5 · Acceptance + doc updates

If acceptance passes (per state, both reg STL TOST non-inf at δ=2pp AND MTL cat-F1 + reg-Acc@10 paired Wilcoxon p < 0.05), update:

- **`PHASE3_TRACKER.md`** — flip cells 🔴 → 🟢 in §Status board.
- **`research/SUBSTRATE_COMPARISON_FINDINGS.md`** — append "Phase 3 — Scope D leakage-free CH15+CH18 closure" section with cross-state tables.
- **`CLAIMS_AND_HYPOTHESES.md`** — CH15 + CH18 status: `confirmed leakage-free at AL+AZ+FL+CA+TX`.
- **`FOLLOWUPS_TRACKER.md`** — close F44.

---

## 6 · Drive backup

After Phase 3 closes, bundle gitignored artefacts:

```bash
DATE=$(date +%Y-%m-%d)
BUNDLE=/teamspace/studios/this_studio/phase3_drive_bundle_$DATE
mkdir -p $BUNDLE/{results,logs,output_per_fold_transitions}

# Run dirs (reg STL + MTL with _pf suffix)
for state in alabama arizona florida california texas; do
  for engine in check2hgi hgi; do
    cp -r results/$engine/$state/mtlnet_* $BUNDLE/results/ 2>/dev/null || true
  done
done
# Reg STL run dirs land in P1 (in git) — only mtlnet_* needs Drive.

# Logs
cp -r logs/phase3 $BUNDLE/logs/

# Per-fold transition matrices (gitignored under output/)
for state in alabama arizona florida california texas; do
    [ -f "output/check2hgi/$state/region_transition_log_fold1.pt" ] || continue
    mkdir -p $BUNDLE/output_per_fold_transitions/$state
    cp output/check2hgi/$state/region_transition_log_fold*.pt \
       $BUNDLE/output_per_fold_transitions/$state/
done

cd /teamspace/studios/this_studio
tar czf phase3_drive_bundle_$DATE.tar.gz $(basename $BUNDLE)/
```

Download the tarball via Lightning Files panel, upload to Drive `mestrado_data/PoiMtlNet/phase3_archives/`.

Then commit small artefacts:
```bash
cd PoiMtlNet
git add docs/studies/check2hgi/results/phase1_perfold/*_pf*.json \
        docs/studies/check2hgi/results/paired_tests/*_pf.json \
        docs/studies/check2hgi/results/P1/*_pf*.json \
        docs/studies/check2hgi/{PHASE3_TRACKER.md,research/SUBSTRATE_COMPARISON_FINDINGS.md,FOLLOWUPS_TRACKER.md,CLAIMS_AND_HYPOTHESES.md}
git commit -m "study(check2hgi): Phase 3 Scope D — leakage-free reg STL + MTL CH18 closed at 5/5"
git push origin worktree-check2hgi-mtl
```

---

## 7 · Troubleshooting

### "per-fold matrix missing"

The reg STL / MTL cell scripts verify per-fold matrices before launching. If they're missing, run Step 1 first:
```bash
STATES="<state>" bash scripts/build_phase3_per_fold_transitions.sh
```

### OOM despite A100

Use `MTL_BATCH_SIZE=1024` env var to halve batch size. Wall-clock unchanged (samples/sec ~constant).

### Slow training (>2× expected)

- Data on slow filesystem: `cp -r output /tmp/output_local && export OUTPUT_DIR=/tmp/output_local`
- Non-deterministic CUBLAS workspace: already handled by orchestrator env vars

### Folds don't match Phase 2

The per-fold transition builder uses `StratifiedGroupKFold(seed=42, n_splits=5)` — same as the trainer's FoldCreator. Mismatches indicate sklearn version drift; verify with `python3 -c "import sklearn; print(sklearn.__version__)"` (expected 1.6+).

### TX MTL still OOMs at bs=2048 on A100 40 GB

If unexpected: try `MTL_BATCH_SIZE=1024 bash scripts/run_phase3_mtl_cell.sh texas check2hgi 0`. Should not happen on A100 — there's ~15 GB headroom over T4 ceiling.

---

## 8 · Quick reference

**Orchestrator scripts (all under `scripts/`):**

| Script | Purpose |
|---|---|
| `setup_lightning_pod.sh` | Bootstrap fresh pod (deps + gdown data, state-flexible) |
| `build_phase3_per_fold_transitions.sh` | Build per-fold transition matrices (CPU) |
| `run_phase3_reg_stl_cell.sh STATE ENGINE GPU_ID` | Reg STL single cell with `--per-fold-transition-dir` |
| `run_phase3_mtl_cell.sh STATE ENGINE GPU_ID` | MTL B3 single cell with `--per-fold-transition-dir` |
| `run_phase3_grid.sh` | Sequential 20-cell grid (single GPU) |
| `run_phase3_parallel.sh` | Auto-dispatch parallel grid |
| `finalize_phase3.py` | Extract per-fold + paired tests + status board |

**Canonical CLI snippets:**

```bash
# reg STL with per-fold transition (used by run_phase3_reg_stl_cell.sh)
python3 scripts/p1_region_head_ablation.py \
  --state <state> --heads next_getnext_hard \
  --folds 5 --epochs 50 --seed 42 --input-type region \
  --region-emb-source <engine> \
  --override-hparams d_model=256 num_heads=8 \
    transition_path=$OUTPUT_DIR/check2hgi/<state>/region_transition_log.pt \
  --per-fold-transition-dir $OUTPUT_DIR/check2hgi/<state> \
  --tag STL_<STATE>_<engine>_reg_gethard_pf_5f50ep

# MTL B3 with per-fold transition (used by run_phase3_mtl_cell.sh)
python3 scripts/train.py \
  --task mtl --state <state> --engine <engine> \
  --task-set check2hgi_next_region --model mtlnet_crossattn \
  --mtl-loss static_weight --category-weight 0.75 \
  --reg-head next_getnext_hard --cat-head next_gru \
  --reg-head-param d_model=256 --reg-head-param num_heads=8 \
  --reg-head-param transition_path=$OUTPUT_DIR/check2hgi/<state>/region_transition_log.pt \
  --per-fold-transition-dir $OUTPUT_DIR/check2hgi/<state> \
  --batch-size 2048 --folds 5 --epochs 50 --seed 42 --no-checkpoints
```

**Per-fold extraction key:** same as Phase 2 (`diagnostic_best_epochs.next_category.metrics.{f1,accuracy}` and `diagnostic_best_epochs.next_region.metrics.{top10_acc_indist,mrr_indist,f1}`).

**Phase 3 closure when:** `paired_tests/{<state>_reg_acc10_pf, <state>_mtl_cat_f1_pf, <state>_mtl_reg_acc10_pf}.json` all show acceptance criteria met across ≥4 of 5 states.
