# bayesagg_mtl

Why this?
- BayesAgg-style weighting approximates uncertainty-aware gradient aggregation
  while staying lightweight enough for staged ablations.

Runtime mapping:
- Registry key: `bayesagg_mtl` (alias: `bayesagg`)
- Runtime class: `losses.mtl_baselines.BayesAggMTLLoss`

Source:
- In-repo implementation: `src/losses/mtl_baselines.py`
- Variant notes: `docs/mtl_optimizers/bayesagg_mtl/README.md`
