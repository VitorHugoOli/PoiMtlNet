# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-07T21:17:33.990335+00:00 · phase **wave1** · commit `da179081`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D D | D D D | . . . | . . . | 10 |
| alabama | D D D | D D D | . . . | . . . | 10 |
| arizona | D D D | D D D | . . . | . . . | 10 |
| florida | D D D | D D ~ | . . . | . . . | 5 |
| texas | D D D | . . . | . . . | . . . | 5 |
| california | D D D | . . . | . . . | . . . | 5 |

## Timing

- cells done: **29 / 72**
- measured wall-clock total: **23.22 h**
- cat: n=10, mean 20.2 min, max 69.8 min
- reg: n=10, mean 26.8 min, max 98.3 min
- joint: n=9, mean 102.6 min, max 376.1 min
- naive estimate for the remaining 43 cells (serial): **34.4 h**

## Running now

- florida s1 joint (pid 126329, since 2026-08-07T21:17:33+00:00)

## Environment

- GPU free: 45489 MiB · /dados free: 2192 GB · /home free: 40 GB

