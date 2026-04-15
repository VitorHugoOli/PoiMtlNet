# Quick Reference — One-page overview

This is the "TL;DR" page for anyone (human or agent) who needs to get oriented fast. For depth, see `README.md`, `MASTER_PLAN.md`, and the phase docs.

---

## What we're doing

A clean-slate empirical study of MTLnet on newly-regenerated (bug-fixed) embeddings, organized around ~20 explicit claims. Each claim has a specific test. Every test feeds results back into a central catalog so the paper has evidence-backed statements only.

---

## Phase order

1. **P0** — Prep. Embeddings regenerated, integrity checks, tooling built, fold indices frozen, sanity run. **~2-4 h**.
2. **P1** — 5 archs × 20 optimizers grid on fusion (AL + AZ). Finds best (arch, optim). **~4-8 h**.
3. **P2** — 9 category × 10 next heads, + MTL vs single-task (critical control for C06). **~8-10 h**.
4. **P3** — Best config on {DGI, HGI, Fusion} + CBIC config on same → proves CBIC's failure was config-specific. **~6-10 h**.
5. **P4** — Hyperparameter sensitivity on the champion. **~7-8 h**.
6. **P5** — Mechanistic / diagnostic (gradient cosine, per-category F1, stats package). **~4-6 h**.
7. **P6** — Canonical MTL claims revisited: convergence, no negative transfer, transferable representations (Caruana/Ruder). **~3-5 h, parallel to P5**.

**Total:** 36-48 h wall-clock, 5-7 days with parallelism.

---

## Datasets

- **Alabama** — fast primary (22 min / 5f×50ep run)
- **Arizona** — fast replication (~25 min / run)
- **Florida** — slow heavy validation (~4 h / run)
- Optional: California, Texas, Georgia for external-baseline matching

### Data scale vs published papers (verified 2026-04-15)

Raw Gowalla check-ins (our ETL, `data/checkins/`):

| State | Check-ins | POIs | Users |
|-------|-----------|------|-------|
| Alabama | 113,846 | 11,848 | 3,858 |
| Arizona | 236,450 | 20,666 | 7,869 |
| Florida | 1,407,034 | 76,544 | 21,052 |

Processed inputs (after sequence window=9, min-5-visits filter):

| State | POIs (model) | Active users | Sequences |
|-------|-------------|--------------|-----------|
| Alabama | 11,706 | 1,623 | 12,699 |
| Arizona | 20,440 | 2,134 | 25,083 |
| Florida | 74,862 | 10,531 | 154,640 |

**CBIC 2025** used Florida + DGI (64-dim). Dataset numbers (`N_users`, `N_poi`, `N_checkins`) are **unfilled template variables** in the LaTeX source — numbers above are the ground truth from our ETL.
CBIC results: cat macro F1 ≈ 47.7%, next macro F1 ≈ 34% (Florida/DGI, 5-fold CV).

**CoUrb 2026** uses Florida, California, Texas + HGI+Sphere2Vec+Time2Vec (192-dim fusion):

| State | Check-ins (CoUrb) | POIs (CoUrb) | Users (CoUrb) | Δ vs our ETL |
|-------|-------------------|--------------|---------------|--------------|
| Florida | 990,518 | 65,009 | 20,301 | ~30% fewer check-ins, ~15% fewer POIs |
| California | 2,535,573 | 148,314 | 36,106 | ~20% fewer |
| Texas | 3,355,419 | 135,570 | 37,522 | — (not in our ETL yet) |

**The ~30% gap for Florida is real**: CoUrb applies a stricter preprocessing filter (higher min-check-ins threshold or narrower date range). Before a head-to-head comparison with CoUrb, either align preprocessing or flag the discrepancy explicitly in the paper.

CoUrb results on Florida: Category F1 up to 73% (vs CBIC's 47.7%), Next F1 up to 40% — improvement attributed to richer fusion, not different data size.

---

## Key claims (abbreviated)

- **C01** Fusion > HGI-only on champion config
- **C02** Gradient-surgery > equal weight on fusion (at matched batch)
- **C05** Expert gating > FiLM hard sharing
- **C06** MTL > single-task on joint score (THE critical control — P2)
- **C11** Embedding quality dominates other choices
- **C12** CBIC's config fails on all embeddings, not just DGI
- **C15-17** Champion is robust to hyperparameters
- **C19-20** Scale imbalance causes source-level gradient conflict (mechanism)
- **C22-23** Modern MTL reaches target F1 without CBIC's 4× wall-clock penalty
- **C24-27** Canonical MTL benefits (regularization, noise robustness, transfer)
- **C28** **No negative transfer** — MTL per-task F1 ≥ single-task (mandatory reviewer shield)

Full list: `CLAIMS_AND_HYPOTHESES.md`.

---

## Coordinator cheat sheet

State lives in `state.json`. Actions (planned `/study` Skill):
- `status` — where are we?
- `next` — run next pending test
- `import <dir>` — archive a result from another machine
- `validate <phase>` — run integrity checks

Integrity contract: `coordinator/integrity_checks.md`.
State schema: `coordinator/state_schema.md`.

---

## Folder map

```
docs/studies/
├── README.md                 entry point
├── QUICK_REFERENCE.md        (this file)
├── MASTER_PLAN.md            detailed strategy
├── CLAIMS_AND_HYPOTHESES.md  authoritative claim catalog
├── COORDINATOR.md            coordinator spec
├── state.json                runtime state (created in P0)
├── phases/
│   ├── P0_preparation.md
│   ├── P1_arch_x_optimizer.md
│   ├── P2_heads_and_mtl.md
│   ├── P3_embedding_cross.md
│   ├── P4_hyperparams.md
│   ├── P5_remaining_claims.md
│   └── P6_convergence_and_mtl_claims.md
├── coordinator/
│   ├── integrity_checks.md
│   └── state_schema.md
└── results/                  populated as study runs
    └── Pk/<test_id>/{summary.json, metadata.json}
```

---

## What to do right now (2026-04-14)

1. **Wait for new embeddings.** User is regenerating. No scientific runs until they're ready.
2. **Meanwhile, build P0 tooling:**
   - `scripts/study/validate_inputs.py`
   - `scripts/study/archive_result.py`
   - `scripts/study/launch_test.py`
   - `/study` Skill (min viable)
3. **Once embeddings are ready:** run P0.2 (integrity) + P0.4 (sanity CBIC replication) + P0.8 (`scripts/study/freeze_folds.py --default-set`) → if all pass, commit state.json and begin P1.

---

## Exit criteria for each level

- **Phase:** all planned tests have `status ∈ {archived, failed, surprising}` and the gate passes.
- **Study:** all claims have `status ≠ pending` and a new `docs/PAPER_FINDINGS.md` is written.
- **Paper:** draft ready in `articles/BRACIS_2026/` with all claims backed by evidence pointers.

---

## If something surprising happens

1. Coordinator pauses. State.json records an open issue.
2. Human reviews: is this a bug, variance, or a real refutation?
3. If real: append a new C-claim (C22+) with the observation, plan a follow-up test.
4. If bug: fix, re-run, record post-mortem in EXECUTION_LOG.
5. Never silently ignore.

---

## BRACIS deadline context

- Paper submission: **2026-04-20**
- Today: **2026-04-13**
- Feasible minimum: P0 + P1 + P2 + P3 = ~25-30 h. If parallelized aggressively, can hit BRACIS.
- P4 + P5 go to journal extension if time-constrained.
