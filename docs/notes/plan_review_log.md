# Refactoring Plan Review Log

> **This is historical context, not implementation guidance.** The authoritative
> plan is [REFACTORING_PLAN.md](../REFACTORING_PLAN.md). The split
> protocol specification is [SPLIT_PROTOCOL.md](../SPLIT_PROTOCOL.md).
>
> This document records 23 rounds of cross-review between two independent
> analyses (Claude and Codex) that produced the final merged plan. Some entries
> reference rules that were later superseded — always defer to the main
> documents for the current protocol.

---

## Source Plans

- **Plan A (Claude):** `REFACTORING_PLAN.md` — Focused on mechanical
  duplication analysis, concrete code samples, bug discovery, unified Trainer
  class (later revised to shared helpers + thin runners).
- **Plan B (Codex):** Independent analysis emphasizing semantic/correctness
  issues, data contract hardening, import-time side effects, research/src
  separation for embedding trainers.

## Merge Decisions

| Conflict | Resolution |
|----------|------------|
| Unified Trainer (A) vs task-level folders (B) | Shared helpers + thin runners |
| Hydra/OmegaConf (B) vs dataclasses (A) | Dataclasses (zero dependencies) |
| `poimtl` rename (B) vs keep layout (A) | Keep current namespace |
| `pyproject.toml` Phase 5 (A) vs Phase 1 (B) | Phase 0 |
| Config before data contracts (A) vs data first (B) | Data contracts first |
| Flat experiments/ (A) vs hierarchy (B) | Hierarchy (configs/baselines/ablations/archive) |
| Embeddings in src/ (A) vs research/ (B) | research/embeddings/ |

## Key Corrections Across 23 Rounds

1. **Unified Trainer → shared helpers + thin runners.** A single Trainer with
   branching would become a new god object. (Round 1)
2. **MTL split = hard gate.** "Fix or document" is too weak for data leakage.
   Elevated to Phase 0. (Round 1)
3. **BPR/SC classification.** "Old pipelines must produce identical results"
   contradicts semantic fixes. Every item classified. (Round 1)
4. **Regression fixtures before refactoring.** Phase 0, not Phase 6. (Round 1)
5. **BPR\* (conditional BPR).** Items that are BPR only if a precondition
   holds (e.g., config values match hardcoded ones). (Round 2)
6. **Layered safety net.** MPS nondeterminism makes simple metric comparison
   too brittle. Three layers: unit → shape → coarse metrics. (Round 2)
7. **MTL split protocol specified.** User-level splits via
   StratifiedGroupKFold. Acceptance constraints. Seed regeneration. (Round 2)
8. **Schema versioning.** Added to ExperimentConfig and RunManifest. (Round 2)
9. **Shim deprecation policy.** DeprecationWarning + pytest enforcement.
   Removed by end of Phase 6. (Round 2)
10. **Phase 5/6 swap.** Tree migration before script consolidation so new
    scripts target final import paths. (Round 2)
11. **MTL split operational algorithm.** 4-step process with POI classification,
    category fold derivation, next-task fold derivation. (Round 3)
12. **Relaxation as explicit opt-in.** Strict mode is default. (Round 4)
13. **Seed regeneration policy.** Deterministic sequence, max 5 attempts,
    all k folds must be valid. (Round 4)
14. **User isolation as hard invariant.** When user isolation and bidirectional
    POI isolation conflict, user isolation wins. Cross-task POI overlap is
    quantified residual, not protocol violation. (Round 5)
15. **Relaxation = move, not duplicate.** POIs moved from category training
    to category validation, not duplicated in both. (Round 6)
16. **Two-phase search order.** All strict seeds first, then relaxed seeds
    for eligible failures only. (Round 7)
17. **Raw checkins in provenance.** FoldCreator uses them for overlap
    diagnostics. Added to dataset_signatures. (Round 7)
18. **Feasibility report as first-class artifact.** Hashed in RunManifest.
    Advisory, not blocking. Manual governance is intentional. (Round 8)
19. **Overlap threshold: first valid seed, not worst-case.** Report runs same
    deterministic search, can predict selected seed. (Round 9)
20. **Threshold scope separated from report validity.** Threshold is
    seed-independent (dataset property). Report measurements include seed. (Round 10)
21. **FeasibilityReport dataclass.** Structured schema with per-seed
    diagnostics, governance flags, decision reasons. (Round 10)
22. **Sequence-to-POI mapping.** Promoted to Phase 2 subtask to decouple
    FoldCreator from raw checkins. (Round 11)
23. **Calibration inputs vs outputs.** ExperimentConfig thresholds are inputs;
    Phase 0 recommendations are outputs. (Round 12)

## What Was Kept Throughout

- Folder tree as target (one-time cost for long-term benefit)
- Dataset SHA-256 hashes in RunManifest
- Rejecting Hydra, Lightning, package rename
- Data contracts before training refactoring
- Model registry
- Experiment hierarchy (configs/baselines/ablations/archive)
