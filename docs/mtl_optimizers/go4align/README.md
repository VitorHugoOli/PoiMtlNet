# GO4Align

`go4align` implements a loss-oriented GO4Align-inspired weighting strategy that
tracks per-task risk (`loss / EMA(loss)`) and dynamic task interactions from
short-window risk correlation, then converts risk-guided indicators to task
weights via temperature-scaled softmax.

## Source

- Paper: [GO4Align: Group Optimization for Multi-Task Alignment (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c98987c5ec4f30920d7190dc699e3daf-Abstract-Conference.html)
- OpenReview page: [GO4Align](https://openreview.net/forum?id=8vCs5U9Hbt)
