# DB-MTL

`db_mtl` implements a Dual-Balancing-style weighting that applies log transform
to task losses, maintains a moving buffer of task gradients, and rescales tasks
by inverse buffered gradient magnitude to reduce imbalance.

## Source

- Paper: [Dual-Balancing for Multi-Task Learning](https://arxiv.org/abs/2308.12029)
- LibMTL reference implementation: [LibMTL/weighting/DB_MTL.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/DB_MTL.py)
