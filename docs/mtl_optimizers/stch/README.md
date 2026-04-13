# STCH

`stch` implements Smooth Tchebycheff scalarization with warmup and nadir-vector
normalization. During warmup it optimizes summed log-losses; after warmup it
applies smoothed max-risk scalarization across normalized task losses.

## Source

- Paper: [Smooth Tchebycheff Scalarization for Multi-Objective Optimization (ICML 2024)](https://openreview.net/forum?id=m4dO5L6eCp)
- LibMTL reference implementation: [LibMTL/weighting/STCH.py](https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/STCH.py)
