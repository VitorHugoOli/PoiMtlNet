# MTLnet Master Study — Entry Point

**Status:** P0 active (2026-04-14)
**Trigger:** label bug in previous dataset → embeddings being regenerated → all prior results are suspect and must be redone. This is our opportunity to design a comprehensive, claim-driven study.

> **If you're starting a fresh session**, read [`HANDOFF.md`](HANDOFF.md) **first** for the current data-availability snapshot, the exact runbook when embeddings arrive, and what blocks P1 enrollment.

## Why this study exists

We had three separate concerns converging:

1. **Prior data is compromised.** A labeling bug in the dataset invalidated comparisons with published baselines. Embeddings are being regenerated.
2. **The critical review flagged holes.** Several claims in our progress report lack the controls needed to defend them (see `docs/REPORT_CRITICAL_REVIEW.md`).
3. **We need a cleaner narrative for BRACIS/journal.** Right now the claims and evidence aren't tightly linked. Each claim should point to a specific test result.

This study re-runs everything from scratch with clean embeddings, organized around a catalog of explicit claims/hypotheses, each with a specific test that confirms or refutes it.

## How to navigate

```
docs/studies/
├── README.md                     ← you are here
├── MASTER_PLAN.md                ← overall strategy, phase order, parallelism
├── CLAIMS_AND_HYPOTHESES.md      ← the claim catalog (C01..Cnn), each links to a test
├── COORDINATOR.md                ← how the coordinator agent orchestrates execution
├── state.json                    ← (created at runtime) current step, test status
├── phases/
│   ├── P0_preparation.md         ← tooling, integrity checks, baseline verification
│   ├── P1_arch_x_optimizer.md    ← 5 archs × 20 optimizers grid + embedded claim tests
│   ├── P2_heads_and_mtl.md       ← 9×10 head ablation + MTL vs single-task
│   ├── P3_embedding_cross.md     ← DGI/HGI/Fusion cross-comparison (incl. CBIC config)
│   ├── P4_hyperparams.md         ← champion config sensitivity
│   └── P5_remaining_claims.md    ← catch-all for claims not resolved in P1-P4
├── coordinator/
│   ├── integrity_checks.md       ← data validation spec
│   └── state_schema.md           ← JSON state file schema
└── results/                      ← JSON summaries archived per test (git-tracked)
    └── P1_arch_x_optim/
        └── AL_dsk42_al_seed42/
            ├── summary.json
            └── metadata.json
```

## Current phase

Look at `state.json` for the authoritative answer. If it doesn't exist yet, we're in **Phase 0 (Preparation)** — waiting for new embeddings, building tooling.

## Workflow per phase

1. **Plan:** read the phase doc (e.g. `phases/P1_*.md`) to see which tests it runs and which claims they validate.
2. **Validate preconditions:** the coordinator checks data integrity (see `coordinator/integrity_checks.md`).
3. **Execute:** run tests sequentially on Alabama / in parallel on other machines.
4. **Archive results:** summary.json + metadata go into `results/<phase>/<test_id>/`.
5. **Analyze:** the coordinator compares results to expected (per hypothesis), flags surprises.
6. **Update claims:** each claim in `CLAIMS_AND_HYPOTHESES.md` gets status updated with pointer to evidence.
7. **Decide next:** if results are surprising, propose new hypotheses/tests; otherwise proceed to next phase.

## Deliverable

At the end of all phases: a new `docs/PAPER_FINDINGS.md` (replacing the current one) synthesizing all validated claims with evidence pointers, ready to back the BRACIS paper and future journal extension.

## Previous study artifacts (kept for reference, not for reuse)

- `docs/PAPER_FINDINGS.md` — old findings (pre-label-bug) — treat as hypotheses to revalidate, not as evidence
- `docs/full_ablation_study/` — prior 5-stage protocol — methodology reference, numbers outdated
- `docs/REPORT_CRITICAL_REVIEW.md` — the review that shaped this new study
- `docs/plans/` — earlier piecemeal plans — superseded by this master plan
