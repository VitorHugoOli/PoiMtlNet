# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-10T20:33:18.189572+00:00 · phase **done** · commit `5075d77d`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D D | D D D | D D D | D D D | 20 |
| alabama | D D D | D D D | D D D | D D D | 20 |
| arizona | D D D | D D D | D D D | D D D | 20 |
| florida | D D D | D D D | D D D | D D D | 20 |
| texas | D D D | D D D | . . D | . . D | 20 |
| california | D D D | D D D | . . D | . . D | 20 |

## Timing

- cells done: **64 / 72**
- measured wall-clock total: **54.95 h**
- cat: n=20, mean 25.7 min, max 91.5 min
- reg: n=20, mean 23.6 min, max 98.3 min
- joint: n=24, mean 96.3 min, max 375.1 min
- naive estimate for the remaining 8 cells (serial): **6.9 h**

## Environment

- GPU free: 45489 MiB · /dados free: 2192 GB · /home free: 37 GB

