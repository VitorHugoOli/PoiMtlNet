# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-06T16:16:30.172314+00:00 · phase **wave0** · commit `82bca519`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D D | . . . | . . . | . . . | 5 |
| alabama | D . D | . . . | . . . | . . . | 5 |
| arizona | D D ~ | . . . | . . . | . . . | 0 |
| florida | . . . | . . . | . . . | . . . | 0 |
| texas | . . . | . . . | . . . | . . . | 0 |
| california | . . . | . . . | . . . | . . . | 0 |

## Timing

- cells done: **7 / 72**
- measured wall-clock total: **1.20 h**
- cat: n=3, mean 4.7 min, max 7.8 min
- reg: n=2, mean 5.6 min, max 8.1 min
- joint: n=2, mean 23.3 min, max 28.3 min
- naive estimate for the remaining 65 cells (serial): **11.1 h**

## Running now

- arizona s0 joint (pid 625368, since 2026-08-06T16:16:30+00:00)

## Environment

- GPU free: 45489 MiB · /dados free: 2192 GB · /home free: 51 GB

