# second_dataset — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** Mac (ETL/scoring) + CUDA box (Phase V) · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 0 (Phase E ETL — parallel, NO freeze dependency) + Level 4 (Phase V validation — needs frozen champion).
- Does NOT block the freeze; the Mac builds it concurrently with `mtl_frontier` / `pre_freeze_gates`.

## Dataset decision
- **Recommended: Massive-STEPS NYC** — confirm with user before ETL. Rationale: only candidate with
  categories + coords + shipped temporal split (= the protocol bridge). Default fallback if rejected:
  FSQ-TKY 2014. See `AGENT_PROMPT.md` / `future_work.md §8`.

## Queue
| Phase | Item | State |
|---|---|---|
| E | acquire + parse NYC to repo schema | not started |
| E | Foursquare→7-root category map (versioned artifact) | not started |
| E | coords→TIGER tract join + cardinality | not started |
| E | folds + shipped temporal split + substrate inputs + priors | not started |
| V | champion G + STL ceilings + Markov floor, 4 seeds (validation only) | blocked on freeze |

## Conventions
- Mac = ETL + scoring only; no heavy CUDA training. Phase V waits for a CUDA box + the frozen champion.
- Scope = validation phase, one city, headline cells — NOT the full closing_data matrix.

## Decisions log
- 2026-06-14 — scaffolded from `docs/research/future_work.md §8`. Massive-STEPS NYC recommended pending user confirm.
