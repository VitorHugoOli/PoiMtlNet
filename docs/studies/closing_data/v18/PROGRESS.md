# v18 — PROGRESS

> Rewritten after every completed cell. `updated_at` 2026-08-09T06:28:18.755200+00:00 · phase **wave0** · commit `e351d4b0`

## Matrix — 6 states × 4 seeds × 3 families

`.` pending · `~` running · `D` done · `F` failed. Families in each cell: cat / reg / joint.

| state | seed 0 | seed 1 | seed 7 | seed 100 | n (joint) |
|---|---|---|---|---|---|
| istanbul | D D ~ | . D . | . . . | . . . | 0 |
| alabama | D D ~ | . D . | . . . | . . . | 0 |
| arizona | . D . | . D . | . . . | . . . | 0 |
| florida | . D . | . D . | . . . | . . . | 0 |
| texas | . D . | . . . | . . . | . . . | 0 |
| california | . D . | . . . | . . . | . . . | 0 |

## Timing

- cells done: **12 / 72**
- measured wall-clock total: **4.78 h**
- cat: n=2, mean 9.4 min, max 13.4 min
- reg: n=10, mean 26.8 min, max 98.3 min
- naive estimate for the remaining 60 cells (serial): **23.9 h**

## Running now

- alabama s0 joint (pid 712015, since 2026-08-09T06:20:18+00:00)
- istanbul s0 joint (pid 713692, since 2026-08-09T06:28:18+00:00)

## Environment

- GPU free: 35946 MiB · /dados free: 2192 GB · /home free: 38 GB

