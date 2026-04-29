#!/usr/bin/env bash
# Phase 3 — bootstrap a fresh Lightning.ai pod for MTL CH18 closure on CA + TX.
# Idempotent: skips already-installed deps and already-downloaded data.
#
# Prereqs: clone this repo, checkout worktree-check2hgi-mtl, then run this.
#
# Usage:
#   bash scripts/setup_lightning_pod.sh
set -eu
cd "$(dirname "$0")/.."

echo "===================================================================="
echo "Phase 3 — Lightning pod bootstrap"
echo "===================================================================="

# ── 1. Verify branch + GPU ──────────────────────────────────────────────
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "worktree-check2hgi-mtl" ]; then
    echo "[warn] On branch $BRANCH (expected worktree-check2hgi-mtl)"
fi

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || {
    echo "[error] nvidia-smi not available — pod has no GPU?"
    exit 1
}

# Detect torch + CUDA version (for PyG wheel selection)
TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0].split('.')[0:2])" 2>/dev/null \
            | tr -d "[],' " | sed 's/\(.\)/\1./;s/\.\(.\)$/\1/')
TORCH_FULL=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
CUDA_VER=$(python3 -c "import torch; v = torch.version.cuda or ''; print(v.replace('.', ''))" 2>/dev/null || echo "")

echo ""
echo "torch: $TORCH_FULL   cuda: $CUDA_VER"

# ── 2. Install Python deps (no-deps, to preserve torch) ─────────────────
echo ""
echo "=== Installing Python deps ==="

# Core deps. cvxpy 1.6.4 is pinned (1.8.x needs numpy 2.x which conflicts).
pip install --no-deps --quiet \
    pyarrow h5py geopandas shapely numba pytorch-warmup \
    cvxpy==1.6.4 ecos==2.0.14 fvcore xxhash psutil tqdm

# llvmlite must be >= 0.47 for numba; pip resolver sometimes picks too-old version.
pip install --no-deps --quiet "llvmlite>=0.47.0,<0.48.0"

# PyG wheels matched to torch+CUDA. Lightning's torch is typically a recent build.
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_FULL%+*}+cu${CUDA_VER}.html"
echo "PyG wheel index: $PYG_INDEX"
pip install --no-deps --quiet \
    torch_geometric torch_scatter torch_sparse torch_cluster \
    -f "$PYG_INDEX" || {
    echo "[warn] PyG wheel install failed for $PYG_INDEX — trying generic"
    pip install --no-deps --quiet torch_geometric
}

echo ""
echo "=== Verifying imports ==="
python3 -c "
import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster
import pytorch_warmup, cvxpy, ecos, geopandas, shapely, numba, h5py, fvcore
import pyarrow, networkx, torchmetrics, psutil, scipy, sklearn, pandas, numpy, matplotlib, tqdm, xxhash
print(f'torch {torch.__version__}  cuda {torch.cuda.is_available()}  devices {torch.cuda.device_count()}')
print('all imports OK')
"

# ── 3. Download upstream data via gdown (skip if present) ───────────────
echo ""
echo "=== Downloading upstream parquets ==="

declare -A FOLDERS=(
    ["check2hgi/california"]="1ZLL8FHPeO7I-3DEfVBogW1C1eFE76ttv"
    ["check2hgi/texas"]="1bLfFDEOM1BJ2ELoQUnd_qMXFpxGsZ7UF"
    ["hgi/california"]="1nMNaFgEEc1RwoH_o8_wasL9ENOkOCdKJ"
    ["hgi/texas"]="1g43xNSlJZBXStt3YGruOvCZTI-_WW4OQ"
)

for path in "${!FOLDERS[@]}"; do
    target="output/$path"
    if [ -f "$target/embeddings.parquet" ] && [ -f "$target/input/next.parquet" ]; then
        echo "  ✓ $target — already present, skipping"
        continue
    fi
    echo "  ↓ downloading $target (gdrive ${FOLDERS[$path]})"
    mkdir -p "$target"
    gdown --folder "https://drive.google.com/drive/folders/${FOLDERS[$path]}" -O "$target" 2>&1 \
        | grep -vE "^[ \t]*[0-9]+%|Building|Retrieving|Processing" | tail -10
done

# ── 4. Pre-flight verification ──────────────────────────────────────────
echo ""
echo "=== Pre-flight verification ==="
ALL_OK=true
for state in california texas; do
    for engine in check2hgi hgi; do
        for f in embeddings.parquet region_embeddings.parquet input/next.parquet input/next_region.parquet; do
            p="output/$engine/$state/$f"
            if [ -f "$p" ]; then
                printf "  ✓ %-50s %s\n" "$engine/$state/$f" "$(du -h $p | cut -f1)"
            else
                printf "  ✗ %-50s MISSING\n" "$engine/$state/$f"
                ALL_OK=false
            fi
        done
    done
    p="output/check2hgi/$state/region_transition_log.pt"
    if [ -f "$p" ]; then
        printf "  ✓ %-50s %s\n" "check2hgi/$state/region_transition_log.pt" "$(du -h $p | cut -f1)"
    else
        printf "  ✗ %-50s MISSING\n" "check2hgi/$state/region_transition_log.pt"
        ALL_OK=false
    fi
done

mkdir -p logs/phase3

if [ "$ALL_OK" = "true" ]; then
    echo ""
    echo "===================================================================="
    echo "Bootstrap complete. Ready to launch:"
    echo "  bash scripts/run_phase3_mtl_parallel.sh   # 4× GPU parallel"
    echo "  bash scripts/run_phase3_mtl_grid.sh       # single-GPU sequential"
    echo "===================================================================="
else
    echo ""
    echo "[error] Some files missing. Re-run gdown for the failed folders:"
    echo "  gdown --folder 'https://drive.google.com/drive/folders/<ID>' -O output/<engine>/<state>"
    exit 1
fi
