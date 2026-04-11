# FairGrad

`fairgrad` implements a FairGrad-style scalarization that derives task weights from
the shared-parameter gradient interaction matrix (`G G^T`). It solves the FairGrad
fixed-point condition with a projected iterative solver and applies the resulting
weights to the task losses.

## Source

- Paper: [Fair Resource Allocation in Multi-Task Learning (ICML 2024)](https://openreview.net/forum?id=KLmWRMg6nL)
- LibMTL reference implementation: [LibMTL/weighting/FairGrad.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/FairGrad.py)
