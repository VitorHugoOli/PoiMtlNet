# mtl_frontier — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** A40 · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 0 (exploration). Blocks: `closing_data` P2 FREEZE (a promoted lever → v17 → re-pin before freeze).
- Runs in parallel with `second_dataset` (Mac) and `closing_data` P1a (reading).

## First-wave queue
| ID | Lever | State | Verdict |
|---|---|---|---|
| R1 | log_C co-location prior + probability-chain | not started | — |
| R2 | STEM-AFTB gating sweep | not started | — |
| R3 | live cross-task distillation | not started | — |

Later waves (R4–R9) gated on first-wave outcomes — see `AGENT_PROMPT.md`.

**R10 (★ user-requested) — Memory-Caching / GRM gating at the layer level** (arXiv:2602.24281, no code).
Second-wave architectural lever adjacent to R2: GRM-gated / SSC-routed read between the dual towers
(primary), and GRM/Memory-Soup fusion across Check2HGI hierarchy levels (speculative, STL-first).
"On the layers, not the transformers." Run R2 first; promote ≥0.3 pp over G, multi-seed. See `AGENT_PROMPT.md §R10`.

## Promote-gate convention
≥0.3 pp either head, multi-seed {0,1,7,100} → STOP for user (recipe → v17) → register in `closing_data` G0.2.
Null → log here + `../log.md` row; do not silently fold into the freeze.

## Decisions log
- 2026-06-14 — scaffolded from `docs/research/mtl_frontier.md` §4 (R1–R9). Optimizer aisle declared closed
  (19-arm null + Kurin/Xin/Mueller); only R9 residual sanity arms remain citable-cheap.
