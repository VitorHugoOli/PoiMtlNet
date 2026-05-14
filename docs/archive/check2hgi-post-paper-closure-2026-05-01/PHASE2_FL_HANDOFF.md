# Phase 2 — FL → CA/TX handoff (mid-flight, 2026-04-28)

Resume note for the next agent session. FL grid was launched on Colab T4 from `notebooks/colab_phase2_grid.ipynb`. CA/TX still queued.

## State at handoff

- **FL Phase-2 grid in flight on Colab T4**, 7 experiments sequential, account using Drive root `/content/drive/MyDrive/mestrado_data/PoiMtlNet/`.
- Cell ⑤ launcher running. Logs land at `MyDrive/mestrado_data/PoiMtlNet/results/phase2_logs/florida_F36*.log`.
- Diagnosis mid-run: STL cells appear **I/O-bound, not GPU-bound** because parquets are read directly from Drive each fold (~487 MB on FL). Local NVMe (`/content/...`) is ~10× faster than Drive mount.
- User decided to let FL finish as-is (don't kill), then re-init agent in a fresh Colab session for CA/TX.

## When re-init'd (start here)

### 1 · Harvest FL artefacts from Drive

After cell ⑤ shows `✅ Phase-2 grid for florida complete` (or 7 individual `EXIT … rc=0`), the following files exist on Drive at `MyDrive/mestrado_data/PoiMtlNet/results/`:

| Experiment | Drive path (under `results/`) |
|---|---|
| F36a probe × 2 | `<repo>/docs/studies/check2hgi/results/probe/florida_{check2hgi,hgi}_last.json` (writes inside the cloned repo, NOT Drive — see note below) |
| F36b cat STL × 2 | `check2hgi/florida/next_*/` and `hgi/florida/next_*/` run dirs |
| F36c reg STL × 2 | `P1/region_head_florida_region_5f_50ep_STL_FLORIDA_*.json` |
| F36d MTL counterfactual | `hgi/florida/mtlnet_*/` run dir |

> **Probe outputs caveat:** the probe script writes to the cloned repo's `docs/studies/check2hgi/results/probe/` (inside `/content/PoiMtlNet/`), which is *not* on Drive. Either copy them to Drive before the Colab session ends, or re-run the probes locally (CPU-only, ~2 min each).

### 2 · Land FL data in the host repo

On the M4 (host repo at this branch), copy from Drive:

```bash
# Probe outputs (head-free)
cp <DRIVE>/results/probe/florida_*.json docs/studies/check2hgi/results/probe/

# Per-fold cat STL extraction (matches the AL/AZ pattern)
python3 - <<'PYEOF'
import json
from pathlib import Path
DRIVE = Path('<DRIVE>/results')
OUT = Path('docs/studies/check2hgi/results/phase1_perfold')
for engine in ('check2hgi', 'hgi'):
    rd = sorted((DRIVE / engine / 'florida').glob('next_*'), key=lambda p: p.stat().st_mtime)
    if not rd: continue
    folds_dir = rd[-1] / 'folds'
    out = {}
    for i in range(5):
        fp = folds_dir / f'fold{i}_info.json'
        m = json.load(fp.open())['diagnostic_best_epochs']['next']['metrics']
        out[f'fold_{i}'] = {'f1': m['f1'], 'accuracy': m['accuracy']}
    (OUT / f'FL_{engine}_cat_gru_5f50ep.json').write_text(json.dumps(out, indent=2))
    print(f'wrote FL_{engine}_cat_gru_5f50ep.json')
PYEOF

# Reg STL — already in flat per-fold format, just copy
cp <DRIVE>/results/P1/region_head_florida_*.json docs/studies/check2hgi/results/P1/

# MTL counterfactual (HGI substrate) per-fold extraction (cat + reg)
python3 - <<'PYEOF'
import json
from pathlib import Path
DRIVE = Path('<DRIVE>/results')
OUT = Path('docs/studies/check2hgi/results/phase1_perfold')
rd = sorted((DRIVE / 'hgi' / 'florida').glob('mtlnet_*'), key=lambda p: p.stat().st_mtime)[-1]
cat, reg = {}, {}
for i in range(5):
    m = json.load((rd / 'folds' / f'fold{i}_info.json').open())['diagnostic_best_epochs']
    cat[f'fold_{i}'] = {'f1': m['next_category']['metrics']['f1'],
                        'accuracy': m['next_category']['metrics']['accuracy']}
    rm = m['next_region']['metrics']
    reg[f'fold_{i}'] = {'f1': rm.get('f1', None), 'acc1': rm['top1_acc'],
                        'acc5': rm['top5_acc'], 'acc10': rm.get('top10_acc_indist', rm['top10_acc']),
                        'mrr': rm.get('mrr_indist', rm['mrr'])}
(OUT / 'FL_hgi_mtl_cat.json').write_text(json.dumps(cat, indent=2))
(OUT / 'FL_hgi_mtl_reg.json').write_text(json.dumps(reg, indent=2))
print('wrote FL_hgi_mtl_{cat,reg}.json')
PYEOF
```

### 3 · Run paired tests for FL (per `PHASE2_TRACKER.md §5`)

```bash
PFD=docs/studies/check2hgi/results/phase1_perfold

# Cat F1 paired Wilcoxon (matched-head)
python3 scripts/analysis/substrate_paired_test.py \
  --check2hgi $PFD/FL_check2hgi_cat_gru_5f50ep.json \
  --hgi       $PFD/FL_hgi_cat_gru_5f50ep.json \
  --metric f1 --task cat --state florida \
  --output docs/studies/check2hgi/results/paired_tests/florida_cat_f1.json

# Reg Acc@10 + MRR with TOST δ=2pp
for METRIC in acc10 mrr; do
  python3 scripts/analysis/substrate_paired_test.py \
    --check2hgi <DRIVE>/results/P1/region_head_florida_region_5f_50ep_STL_FLORIDA_check2hgi_reg_gethard_5f50ep.json \
    --hgi       <DRIVE>/results/P1/region_head_florida_region_5f_50ep_STL_FLORIDA_hgi_reg_gethard_5f50ep.json \
    --metric $METRIC --task reg --state florida --tost-margin 0.02 \
    --output docs/studies/check2hgi/results/paired_tests/florida_${METRIC}_reg_${METRIC}.json
done
```

### 4 · Update PHASE2_TRACKER status board for FL row

Switch the FL row in `docs/studies/check2hgi/PHASE2_TRACKER.md` from 🔴 → 🟢, populate combined paired-test column, commit + push.

### 5 · Apply local-SSD optimisation for CA/TX

The cat-STL slowness on FL was caused by reading the 487 MB `next.parquet` directly from Drive each fold. **Before** launching CA/TX in the next Colab session, insert a copy step right after cell ④:

```python
# Insert as cell ④.5 — copy parquets to /content/output (local NVMe SSD)
import shutil, os
LOCAL_OUT = Path('/content/output')
for engine in ('check2hgi','hgi'):
    src = OUTPUT_DIR / engine / STATE
    dst = LOCAL_OUT / engine / STATE
    dst.mkdir(parents=True, exist_ok=True)
    for sub in ('embeddings.parquet','region_embeddings.parquet','region_transition_log.pt'):
        p = src / sub
        if p.exists() and not (dst/sub).exists():
            shutil.copy(p, dst/sub); print(f'  ✓ {engine}/{state}/{sub} ({p.stat().st_size/1e6:.0f} MB)')
    (dst/'input').mkdir(exist_ok=True)
    for sub in ('next.parquet','next_region.parquet'):
        p = src / 'input' / sub
        if p.exists() and not (dst/'input'/sub).exists():
            shutil.copy(p, dst/'input'/sub); print(f'  ✓ {engine}/{state}/input/{sub} ({p.stat().st_size/1e6:.0f} MB)')
os.environ['OUTPUT_DIR'] = str(LOCAL_OUT)
OUTPUT_DIR = LOCAL_OUT  # rebind in notebook scope
TRANSITION_PATH = OUTPUT_DIR / 'check2hgi' / STATE / 'region_transition_log.pt'
print('OUTPUT_DIR rebound to:', OUTPUT_DIR)
```

Then re-build the EXPERIMENTS list (rerun cell ④ which references `OUTPUT_DIR`/`TRANSITION_PATH`) before the launcher in cell ⑤.

Expected speedup: cat-STL ~5× faster (Drive seq read ~50 MB/s vs `/content/` NVMe ~500 MB/s); MTL+gethard a smaller win (it's GPU-bound, not I/O-bound).

> **Run dirs still need to write to Drive** for persistence across sessions. Either keep `RESULTS_ROOT = DRIVE_ROOT/'results'` (same as before — small writes are fine), or set `RESULTS_ROOT = LOCAL_OUT/'..'/results` and rsync at end of session.

### 6 · Launch CA, then TX

After CA grid completes, repeat steps 1–4 with `STATE = 'california'` substituted. Same for TX. PHASE2_TRACKER FL/CA/TX rows all 🟢 closes Phase 2.

## Memory: do not lose

- Repo branch: `worktree-check2hgi-mtl` (current; has both substrate-comparison scripts + perf patches).
- Drive account using `MyDrive/mestrado_data/PoiMtlNet/` (NOT `mestrado/PoiMtlNet/` — different account).
- Notebook driver: `notebooks/colab_phase2_grid.ipynb` (cell IDs may shift if user inserts cells; lookup by source content).
- Phase-1 reference (AL+AZ) lives at `docs/studies/check2hgi/research/SUBSTRATE_COMPARISON_FINDINGS.md`. FL/CA/TX should land alongside it.
