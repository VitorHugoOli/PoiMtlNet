# Sphere2Vec (sphereM variant)

A spherical-RBF location encoder ported from the source notebook
`Location Encoders/A General-Purpose Location Representation Learning over a Spherical Surface for Large-Scale Geospatial Predictions (Sphere2Vec-sphereM).ipynb`.

Reference: Mai et al., 2023 — https://arxiv.org/abs/2306.17624

## Architecture (faithful port)

```
Input coords [lat, lon] (degrees)
    │
    ▼ deg → rad → unit-sphere Cartesian (x, y, z)
SpherePositionEncoder    (frozen RBF: 256 random unit centroids × 32 log-spaced scales)
    │  → [B, 256·32 = 8192]
Linear input_projector    8192 → 512
    │
MultiLayerFeedForwardNN   512 → 512 → 128   (1 hidden layer, ReLU, dropout=0.5, layernorm, skip)
    │
Linear projector          128 → 64
    │
F.normalize(dim=-1)
    ▼
Output [B, 64]   (L2-normalized)
```

The RBF centroids are random-initialized and then **frozen via `register_buffer`** —
the spherical position encoder is not trained, only the projector + MLP + final
projector are.

## Training signal (caveat)

The contrastive task is intentionally simple:

- **Positive pairs**: `coord_i + Gaussian noise(σ=0.01°)` (~1.1 km perturbation).
- **Negative pairs**: `(coord_i, random_other_coord)`.
- **Loss**: BCE on cosine-similarity logits with temperature `τ = 0.15`.

This produces a smooth-by-distance embedding space, but the supervision is weak
(the model never sees real proximity structure beyond noise vs. random). It is a
faithful port of the reference algorithm, **not** a reimplementation. Tuning is
out of scope for this package.

## ⚠️ Important: dropout is ACTIVE during inference (notebook bug, preserved)

The source notebook (cell 12) calls
```python
loc_embeds = model(torch.Tensor(coords))
```
**without** `model.eval()` and **without** `torch.no_grad()`. The model still
has its two `nn.Dropout(p=0.5)` layers active, so each forward pass on the
same coordinate yields a *different* embedding. The downstream
`groupby+mean` averages out some of this noise, but POIs visited only a few
times end up with measurably noisy embeddings.

This package preserves that behavior **by default** so the migrated output is
faithful to the source. To get deterministic embeddings instead, pass
`args.eval_inference=True` to `create_embedding`. The unit-test
`test_eval_inference_produces_unit_norm` documents both code paths.

You can spot the difference in the output `embeddings.parquet`:

| Mode | Per-row L2 norm | Determinism |
|---|---|---|
| `eval_inference=False` (default, **matches notebook**) | < 1.0 (averaging unit vectors with dropout noise) | Stochastic without seeds |
| `eval_inference=True` (deterministic fix) | ≈ 1.0 (model.eval() means same coord → same unit-norm embedding) | Deterministic with seeds |

## Notable port decisions

| Notebook | This package | Reason |
|---|---|---|
| No random seeds | `args.seed` threaded through `torch`/`numpy`/`random` and a `worker_init_fn` for `DataLoader` workers | Reproducibility + enables equivalence testing |
| `device='cuda' if cuda else 'cpu'` hard-coded inside model | `device` arg, default from `configs.globals.DEVICE` | MPS support |
| Forces CPU in training cell | Honors `args.device` | MPS / CUDA support |
| `class LayerNorm(nn.Module)` defined but unused | Dropped | Dead code; pipeline already uses `nn.LayerNorm` |
| `loc_embeds = model(torch.Tensor(coords))` (one giant call, train mode) | Batched eval (`batch_size=10000`), still in train mode by default | Memory safety on real states; default behavior matches notebook |
| Per-checkin training (no dedup) → groupby `placeid` mean | **Same** (faithful port) | Preserves training distribution |
| CSV + per-row in-train-mode dropout-noisy embeddings | `embeddings.parquet` (parquet, not CSV) but values match cell 14 bit-for-bit under fixed seeds | Matches project storage convention; values preserved |
| Final column order `[placeid, "0"..."63", category]` | `[placeid, category, "0"..."63"]` (intentional deviation) | Matches project convention (`space2vec`, `hgi`, `poi2hgi` all use this) |
| Model state_dict never saved | Saved to `sphere2vec_model.pt` | Reproducibility + checkpoint resume |

## Usage

### Pipeline
```bash
python pipelines/embedding/sphere2vec.pipe.py
```

### Programmatic
```python
from argparse import Namespace
import torch
from configs.globals import DEVICE
from embeddings.sphere2vec import create_embedding

args = Namespace(
    dim=64, spa_embed_dim=128,
    num_scales=32, min_scale=10, max_scale=1e7, num_centroids=256,
    ffn_hidden_dim=512, ffn_num_hidden_layers=1, ffn_dropout_rate=0.5,
    ffn_act="relu", ffn_use_layernormalize=True, ffn_skip_connection=True,
    epoch=50, batch_size=64, lr=1e-3, tau=0.15, pos_radius=0.01,
    seed=42, device=DEVICE, num_workers=2,
)
create_embedding(state="Alabama", args=args)
```

### CLI
```bash
PYTHONPATH=src:research python -m embeddings.sphere2vec.sphere2vec --state Alabama
```

## Output

`output/sphere2vec/{state}/embeddings.parquet`

| Column | Type | Description |
|---|---|---|
| `placeid` | str | POI identifier |
| `category` | str | Original category (mode of per-checkin categories) |
| `0` … `63` | float32 | 64-d L2-normalized embedding |

`output/sphere2vec/{state}/sphere2vec_model.pt` — trained `state_dict`.

## Files

- `__init__.py` — public API
- `sphere2vec.py` — `create_embedding(state, args)` entry point
- `model/Sphere2VecModule.py` — model classes (verbatim port)
- `model/dataset.py` — `ContrastiveSpatialDataset`
- `README.md` — this file
- `CLAUDE.md` — short agent pointer

## Equivalence test

`tests/test_embeddings/_sphere2vec_reference.py` is a frozen verbatim copy of the
notebook's model code. `tests/test_embeddings/test_sphere2vec.py` seeds both
versions identically and asserts forward-pass equivalence at every layer
(`SpherePositionEncoder`, `SphereLocationEncoder`, `SphereLocationContrastiveModel`)
plus loss equivalence and a tiny end-to-end smoke test on synthetic data.
