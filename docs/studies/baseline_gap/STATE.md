# baseline_gap — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** A40 (training) + Mac (ETL/scoring) · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 0/1: the *triage decision* (which baselines become RUN_MATRIX rows/columns) feeds `closing_data`
  P1b and must land before the freeze. Implementation runs in parallel; final runs fold into P3 (M1 board).
- Does NOT change the champion recipe or substrate identity → only the RUN_MATRIX decision gates the freeze.

## Queue
| Tier | ID | Baseline | State |
|---|---|---|---|
| 1 | B1 | CTLE substrate column (matched heads) | not started |
| 1 | B2 | POI2Vec / skip-gram substrate columns | not started |
| 2 | B3 | HMT-GRN-style external MTL baseline (cat+region) | not started |
| 2 | B4 | Cascade category→region baseline | not started |
| 2 | B5 | Flashback / DeepMove (region targets) | not started |
| 3 | — | TALE / Geo-Teaser / CACSR / LBSN2Vec / LLM zero-shot / true-PLE | conditional (cheap/time-permitting) |

## Conventions
- **Comparability regime (binding):** baselines run on the FROZEN base — same substrate (v14/v17), same
  adopted windowing (overlapping/stride), same folds/seeds/priors. End-to-end baselines (B3/B4/B5) build
  their own inputs, so they must mirror the adopted windowing/splits and are **blocked on the
  overlapping-window ADOPT/KEEP decision + the P2 freeze**; budget one re-run if overlap is adopted. See
  `AGENT_PROMPT.md §Comparability regime`.
- Substrate-column baselines (B1/B2) use matched heads + frozen substrate; end-to-end baselines (B3/B4/B5)
  are faithful reimplementations to the `docs/baselines/` fidelity bar.
- CTLE pre-trains on train-portion-only (its protocol) — no transductive full-corpus advantage.
- Second-dataset (Massive-STEPS) external baselines = OUT of scope unless the user pulls one in.
- Maintain a fairness ledger (tuning budget vs Check2HGI) per `baseline_gap_analysis.md §1.4`.
- Multi-seed {0,1,7,100}; paired Wilcoxon; report n and p.

## Decisions log
- 2026-06-14 — scaffolded as the corrective for the audit gap: the pre-freeze family folded only A2
  (feature-concat) into `pre_freeze_gates`, leaving the substantive external baselines (CTLE, POI2Vec
  columns, HMT-GRN-style MTL, cascade, Flashback/DeepMove) unowned. Triaged from `baseline_gap_analysis.md`.
