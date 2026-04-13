# ExcessMTL

`excess_mtl` implements an ExcessMTL-style robust update that accumulates squared
shared gradients per task and updates task weights with an exponential robust-risk
step. This follows the excess-risk weighting idea while fitting the current loss
registry API.

## Source

- Paper: [Robust Multi-Task Learning with Excess Risks (ICML 2024)](https://openreview.net/forum?id=JzWFmMySpn)
- LibMTL reference implementation: [LibMTL/weighting/ExcessMTL.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/ExcessMTL.py)
