# BayesAgg-MTL

`bayesagg_mtl` implements a Bayesian uncertainty-inspired gradient aggregation
approximation for our current two-task setup. It estimates per-task gradient
uncertainty from running second moments on shared-parameter gradients and uses
precision-weighted task scalarization.

## Source

- Paper: [Bayesian Uncertainty for Gradient Aggregation in Multi-Task Learning (ICML 2024)](https://arxiv.org/abs/2402.04005)
- Official code: [ssi-research/BayesAgg_MTL](https://github.com/ssi-research/BayesAgg_MTL)
- Core reference module: [BayesAgg_MTL/BayesAggMTL.py](https://github.com/ssi-research/BayesAgg_MTL/blob/main/BayesAgg_MTL/BayesAggMTL.py)
