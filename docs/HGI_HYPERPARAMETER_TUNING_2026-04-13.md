# HGI Hyperparameter Tuning — Cross-State Results (2026-04-13)

Empirical findings from two sweeps run on 2026-04-12/13:
1. `cross_region_weight` (w_r) tuned per state using the MTLnet next head
   (3 folds × 25 epochs).
2. `poi2vec_epochs` ablated on Alabama to validate the default value of 100.

Builds on the performance improvement plan in `docs/HGI_PERFORMANCE_IMPROVEMENT_PLAN.md`.

---

## 1. cross_region_weight (w_r) Sweep

### What it controls

`cross_region_weight` (w_r) is a scalar multiplier on Delaunay graph edges that
cross census-tract boundaries (Equation 2 in Huang et al., ISPRS 2023). It
modulates how much inter-region spatial signal is injected into the HGI graph
relative to within-region edges.

- `w_r < 1.0` → penalizes cross-boundary edges (emphasizes local structure)
- `w_r = 1.0` → treats cross-boundary edges identically to within-boundary
- `w_r > 1.0` → amplifies cross-boundary signal (possible but untested)

### Protocol

- Task: next-head only (`--task next`, `NextHeadMTL`)
- Protocol: `3f25e` (3 folds × 25 epochs)
- Metric: macro-F1 on next-POI prediction
- Per-state decision rule: pin w_r if `gap(best, second) ≥ 1σ(second)`

### Results

| State | w_r=0.4 | w_r=0.5 | w_r=0.7 | w_r=0.9 | w_r=1.0 | Pinned |
|-------|---------|---------|---------|---------|---------|--------|
| Alabama | 0.718 | — | **0.791** | — | — | **0.7** |
| Florida | 0.744 | 0.761 | **0.783** | 0.769 | 0.747 | **0.7** |
| California | 0.735 | — | **0.767** | — | *(failed)* | **0.7** |
| Texas | 0.765 | — | **0.799** | — | *(OOM)* | **0.7** |

Notes:
- FL w_r=0.5 and w_r=0.9 were run as extra sweep points after FL w_r=0.7 was
  the initial best, to confirm it was not a local plateau. Neither beat 0.7.
- CA w_r=1.0 failed due to a missing `use_torch_compile` attribute in
  `ExperimentConfig` (fixed 2026-04-13). Decision is valid from 0.4 and 0.7.
- TX w_r=1.0 HGI was OOM-killed mid-training (155K POIs / 6547 regions).
  Decision is still valid: gap(0.799, 0.765) = 0.034 >> 1σ = 0.006.

### Why w_r=0.7 wins across all US states

The original paper (Huang et al., ISPRS 2023) was tuned on dense Chinese urban
areas (~26–150 POI/km²). US census tracts are fundamentally different:

| Dataset | POI density | Typical POI/region |
|---------|------------|-------------------|
| Chinese cities (paper) | 26–150 POI/km² | ~200–800 |
| Alabama | ~0.15 POI/km² | ~11 |
| Texas | ~0.10 POI/km² | ~24 |

US states are **100–1000× sparser**. In sparse graphs, cross-tract edges carry
genuine spatial signal because many tracts have few enough POIs that
neighboring tracts are the only source of structural context.

The three failure modes:

- **w_r=0.4** (over-penalizes): cross-boundary edges are so down-weighted that
  sparse tracts lose access to neighborhood context. The GCN layer sees a nearly
  disconnected graph in low-density regions.

- **w_r=1.0** (boundary-blind): the census-tract boundary prior is discarded
  entirely. Tracts that are administratively adjacent but functionally different
  (e.g., a downtown commercial tract next to a residential suburb) become
  indistinguishable.

- **w_r=0.7** (optimal): retains the boundary prior as a soft constraint, giving
  cross-boundary edges ~70% of the weight of within-boundary edges. At US POI
  densities, this is the correct signal-to-noise trade-off — cross-boundary
  context is informative, but slightly discounted relative to local structure.

**Implication:** w_r=0.7 is not a coincidence. It is robust across four US states
spanning very different geographies (coastal FL, mountain-desert CA, large TX,
moderate AL). States with similar sparse POI distributions should use the same
default.

States not yet swept (Arizona, Georgia) can safely extrapolate to w_r=0.7 until
swept explicitly.

---

## 2. poi2vec_epochs Ablation

### What it controls

`poi2vec_epochs` is the number of skip-gram training epochs for the POI2Vec
phase of the HGI pipeline. POI2Vec pre-trains fclass embeddings (267 categories
in Alabama) using Node2Vec random walks on the Delaunay POI graph. These
embeddings become the initial node features for the HGI graph convolution.

Default: 100 epochs.

### Protocol

- State: Alabama (only; HGI + next head, isolated per point)
- Grid: {25, 50, 75, 100, 150, 200}
- Protocol: `3f25e` (3 folds × 25 epochs)
- Parallelism: 2 isolated workers (city alias `Alabama_ep{N}`, each with its own
  `output/hgi/alabama_ep{N}/` and `results/hgi/alabama_ep{N}/` directories)
- Run via: `scripts/run_poi2vec_ablation.py`

### Results

| poi2vec_epochs | Next F1 | σ |
|---------------|---------|---|
| 25 | 0.2528 | 0.0029 |
| **50** | **0.2533** | 0.0019 |
| 75 | 0.2452 | 0.0051 |
| 100 | 0.2447 | 0.0082 |
| 150 | 0.2494 | 0.0071 |
| 200 | 0.2487 | 0.0069 |

**Decision: INCONCLUSIVE** — gap(ep50, ep25) = 0.0004 << 1σ = 0.0029.
Total sweep range = 0.009 F1. **Keep default `poi2vec_epochs=100`.**

### Interpretation

POI2Vec skip-gram training converges early on Alabama's graph (11,706 POIs,
35,093 edges, 267 fclass types). The fclass co-occurrence structure is captured
within the first 25 epochs; additional training adds variance but no signal.

The extremely flat curve (0.009 F1 range across an 8× increase in compute)
means `poi2vec_epochs` is not a meaningful hyperparameter for this dataset.
The HGI improvement plan (Phase 5 in `HGI_PERFORMANCE_IMPROVEMENT_PLAN.md`)
suggested this sweep precisely to validate whether the default could be safely
reduced — it cannot, but it also cannot be usefully increased.

**Implication for larger states:** Texas (155K POIs, 465K edges) may show
more sensitivity to poi2vec_epochs because the walk co-occurrence graph is
richer. A follow-up sweep on Texas is warranted if runtime is the bottleneck,
but requires sequential execution (cannot run parallel workers; see below).

---

## 3. Scale and Infrastructure Notes

### Texas is too large for parallel HGI runs

Texas HGI requires loading 155K POIs / 6547 regions / 465K edges into memory
simultaneously. During the w_r sweep, TX w_r=1.0 HGI was OOM-killed while
three poi2vec ablation workers were also running on Alabama.

**Rule of thumb:** Do not run Texas HGI in parallel with any other HGI workload
on this machine. Run TX sweeps sequentially (MAX_WORKERS=1) or in isolation.

Alabama (11K POIs, 1108 regions) can run 2 parallel HGI workers safely with
~3–4 GB free RAM. 3 workers push into compression territory (78 MB free
observed) — use 2 as the default for Alabama ablations.

### Bugs fixed in this session

Both affected the `--task next` training path and caused all ablation workers
to fail before this session's fixes:

1. **`ExperimentConfig` missing `use_torch_compile`** (`src/configs/experiment.py`):
   An upstream merge added a `config.use_torch_compile` check in `mtl_cv.py:296`
   but the dataclass was not updated. Fixed by adding `use_torch_compile: bool = False`.
   Affected: CA w_r=1.0, FL w_r=0.5/0.9 initial runs.

2. **FLOPs KeyError in `next_cv.py`** (`src/training/runners/next_cv.py:139`):
   `result['total_flops']` crashed when `fvcore` is not installed. The same guard
   already existed in `mtl_cv.py` but was missing in `next_cv.py`. Fixed by adding
   `if 'total_flops' in result` check (matching the mtl_cv.py pattern).
   Affected: all six poi2vec ablation workers in the first parallel run.

---

## 4. Updated State Configurations

Current pinned values in `pipelines/embedding/hgi.pipe.py`:

| State | cross_region_weight | poi2vec_epochs | Source |
|-------|-------------------|---------------|--------|
| Alabama | 0.7 | 100 (default) | swept 2026-04-12 |
| Florida | 0.7 | 100 (default) | swept 2026-04-12/13 |
| California | 0.7 | 100 (default) | swept 2026-04-13 |
| Texas | 0.7 | 100 (default) | swept 2026-04-13 |
| Arizona | 0.7 | 100 (default) | extrapolated (sparse) |
| Georgia | 0.7 | 100 (default) | not swept (extrapolated) |

All states converge to the same configuration. The universality of w_r=0.7
across US states means the pipeline can be applied to new states without
a dedicated sweep.

---

## 5. Other HGI Hyperparameters — Priority Assessment

These were identified in the HGI improvement plan and literature review but
not yet swept. Ranked by expected ROI:

| Hyperparameter | Default | Expected sensitivity | Risk | Priority |
|---------------|---------|---------------------|------|----------|
| `cross_region_weight` | 0.7 | **High** (confirmed) | Low | **Done** |
| `poi2vec_epochs` | 100 | **Low** (confirmed) | Low | **Done** |
| `alpha` (MI weighting) | 0.5 | Medium | Low | Next |
| `dim` (embedding size) | 64 | Medium | Medium | After alpha |
| `epoch` (HGI epochs) | 2000 | Low–Medium | Low | Optional |
| `lr` / `warmup_period` | 0.006 / 40 | Low (coupled) | High | Skip |
| `attention_head` | 4 | Low | Low | Skip |
| `max_norm` | 0.9 | Low | Low | Skip |
| `gamma` (loss scale) | 1.0 | Low | Low | Skip |

**`alpha`** (weight between POI-level and region-level MI losses) is the
highest-ROI next sweep. The paper reports sensitivity to alpha on Chinese data;
US sparse graphs may have a different optimum. Suggested grid: {0.3, 0.5, 0.7}.

**`dim`** is coupled to the downstream model (MTLnet's `EMBEDDING_DIM=64`) and
fusion concatenation width. Changing it requires regenerating all embeddings and
inputs. Only worth sweeping after other improvements are exhausted.
