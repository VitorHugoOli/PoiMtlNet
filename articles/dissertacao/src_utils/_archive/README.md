# `src_utils/_archive/` — superseded working files

Nothing here is live. These files were the working record of rounds 1 to 5 and are kept because they
are the provenance of decisions the document now carries. Archived 2026-07-27 at the author's request,
so `src_utils/` holds only what is still in use.

**If you are looking for the current state**, it is in `src_utils/PENDENCIAS.md` (what still needs the
author) and `src_utils/codex_reviewer.md` (the external review, annotated with per-finding verdicts).

## `reviews_v1/` — superseded review documents

| File | What it was | Why it is here |
|---|---|---|
| `dissertation_review_codex_v1_89pp.md` | The first external review, against an 89/84-page build (was `test.md` at the repo root) | Superseded by `codex_reviewer.md`; every finding was dispositioned |
| `dissertation_review_v1.md` | The first internal review pass | Superseded by v2, then by the round-4 and round-5 work |
| `dissertation_review_v2.md` | The re-audited review, 39 findings with per-finding disposition | Its open items moved to `PENDENCIAS.md`; kept as the audit trail |
| `DECISOES_PENDENTES_ptBR.md` | The first author-decision register | Every item carried into `PENDENCIAS.md` or resolved; verified item by item 2026-07-27 |
| `noth_star_consideration.md` | Three author notes on arc, fundamentals and scope | Absorbed into `NORTH_STAR.md` §4b, which records which points landed and which is still open |

## `reports_2026-07/` — one-off reports

| File | What it was |
|---|---|
| `FRAME_INTEGRATION_REPORT.md` | How the three papers were stitched into one document |
| `APPENDIX_D_EXPLAINED.md` | A working explanation of the benchmark appendix, written before the appendix was rewritten |
| `cbic_recompute_result.md` | NOT archived. Still cited as provenance from `3_cbic.tex`, `apx_b_errata.tex` and `NORTH_STAR.md` |

## `handoffs/` — machine handoff payloads

`HANDOFF_v1.md`, `cbic_recompute_handoff.json`, `handoff_tooling.json`,
`item4_licence_evidence.json`. Structured evidence produced by sub-agents. The claims they support are
in the document; these are the raw payloads behind them.

## What deliberately stayed in `src_utils/`

| File | Why it is still live |
|---|---|
| `PENDENCIAS.md` | The register of what needs the author |
| `codex_reviewer.md`, `CODEX_AUDIT.md`, `CODEX_VS_PERSONAS.md` | The external review, its per-finding audit, and the seam against the persona suite |
| `DATASET_LICENSING_FINDINGS.md` | Every licence claim in Appendix E traces here |
| `etl_tooling_contribution_evidence.md` | The evidence behind Appendix A's software-contribution claim |
| `cbic_recompute_result.md` | Cited as provenance from three source files |
| `BIB_MERGE_REPORT.md` | Cited from `apx_b_errata.tex` and `references.bib` |
| `build.sh`, `check.sh`, `check_trapped_prose.py`, `test_trapped_prose.py` | The build and gate toolchain |
| `README_SRC.md` | The build recipe and TeX-tree notes |
| `_review_v1/`, `_review_v2/`, `_specialists_v1/`, `_specialists_v2/`, `_gates/` | Reviewer reports, referenced by both registers |
| `adaptation_ledgers/` | Per-chapter records of what changed against the published text |
