# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-06T17:07:29.905072+00:00 · phase **wave0** · commit `c17ee729`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D D | . . . | . . . | . . . | 5 |
| alabama | D D D | . . . | . . . | . . . | 5 |
| arizona | D D D | . . . | . . . | . . . | 5 |
| florida | D ~ . | . . . | . . . | . . . | 0 |
| texas | . . . | . . . | . . . | . . . | 0 |
| california | . . . | . . . | . . . | . . . | 0 |

## Timing

- cells done: **10 / 72**
- measured wall-clock total: **2.10 h**
- cat: n=4, mean 11.4 min, max 31.7 min
- reg: n=3, mean 4.8 min, max 8.1 min
- joint: n=3, mean 21.9 min, max 28.3 min
- naive estimate for the remaining 62 cells (serial): **13.0 h**

## Running now

- florida s0 reg (pid 639632, since 2026-08-06T17:07:29+00:00)

## Environment

- GPU free: 45489 MiB · /dados free: 2192 GB · /home free: 51 GB

