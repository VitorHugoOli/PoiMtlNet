# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-06T19:37:21.989061+00:00 · phase **wave1** · commit `cce3ff65`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D D | . . . | . . . | . . . | 5 |
| alabama | D D D | . . . | . . . | . . . | 5 |
| arizona | D D D | . . . | . . . | . . . | 5 |
| florida | D D D | . . . | . . . | . . . | 5 |
| texas | ~ . . | . . . | . . . | . . . | 0 |
| california | . . . | . . . | . . . | . . . | 0 |

## Timing

- cells done: **12 / 72**
- measured wall-clock total: **4.59 h**
- cat: n=4, mean 11.4 min, max 31.7 min
- reg: n=4, mean 10.5 min, max 27.8 min
- joint: n=4, mean 46.9 min, max 121.7 min
- naive estimate for the remaining 60 cells (serial): **23.0 h**

## Running now

- texas s0 cat (pid 719485, since 2026-08-06T19:36:56+00:00)

## Environment

- GPU free: 19834 MiB · /dados free: 2192 GB · /home free: 50 GB

