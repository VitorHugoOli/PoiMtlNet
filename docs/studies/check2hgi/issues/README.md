# Check2HGI Study — Issues & Flaws Log

Deep-dive audits of flaws, bugs, and methodological issues discovered during the study. Complements `CONCERNS.md` (lightweight index of open concerns) with full investigations.

Convention (mirrors `docs/studies/fusion/issues/`):
- `<ISSUE_NAME>_AUDIT.md` — investigation + fix + verification
- `<ISSUE_NAME>_EXPLAINED.md` — optional deeper explanation for reviewers

## Index

| Issue | Severity | Detected | Status | Fix |
|-------|----------|----------|--------|-----|
| [FOLD_LEAKAGE_AUDIT](FOLD_LEAKAGE_AUDIT.md) | HIGH | 2026-04-17 (P2 critical review) | FIXED + VERIFIED | `src/data/folds.py::_create_single_task_folds` uses `StratifiedGroupKFold` for NEXT task |
| [REGION_HEAD_MISMATCH](REGION_HEAD_MISMATCH.md) | HIGH | 2026-04-16 (P1) | FIX PENDING | Swap `next_mtl` Transformer → `next_gru` in MTL region slot |
| [GRADNORM_EXPERT_GATING](GRADNORM_EXPERT_GATING.md) | MEDIUM | 2026-04-17 (P2 screen) | WORKAROUND | Skip gradnorm × expert-gating combos until root-caused |

## Related logs

- `../CONCERNS.md` — lightweight concerns index (C01–C11)
- `../CLAIMS_AND_HYPOTHESES.md` — every claim with its evidence + status
- `../HANDOFF.md` — session-level state
