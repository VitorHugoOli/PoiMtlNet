# docs/ — Project Documentation

**Project:** MTLnet — multi-task learning for POI prediction (category + next-POI).
**Primary study:** **check2hgi** — check-in-level Check2HGI substrate (paper at BRACIS 2026).

> **Single source of truth for paper numbers**: [`results/RESULTS_TABLE.md §0`](results/RESULTS_TABLE.md).
> **Single source of truth for paper prose**: [`../articles/[BRACIS]_Beyond_Cross_Task/`](../articles/[BRACIS]_Beyond_Cross_Task/) — read `AGENT.md` first.
> **Operational documentation** (Colab, RunPod, Lightning, H100, local, Drive): [`infra/`](infra/).

---

## Navigation by question

### "Where are we now?" (check2hgi study briefing)

- [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) — long-form study briefing
- [`NORTH_STAR.md`](NORTH_STAR.md) — committed champion config (B9 MTL recipe; H3-alt small-state)
- [`CHANGELOG.md`](CHANGELOG.md) — timeline of findings + lessons learned
- [`CLAIMS_AND_HYPOTHESES.md`](CLAIMS_AND_HYPOTHESES.md) — claim catalog (paper-facing whitelist banner inside)
- [`CONCERNS.md`](CONCERNS.md) — risk audit log
- [`FINAL_SURVEY.md`](FINAL_SURVEY.md) — substrate-axis 5-state matrix
- [`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md) — supplementary narrative (F-trail B3 → F45 → F48-H3-alt → F49 → paper closure)
- [`PAPER_BASELINES_STRATEGY.md`](PAPER_BASELINES_STRATEGY.md) — which baselines appear in which paper table

### "What are the canonical numbers?"

- [`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md) ⭐ canonical paper-facing numbers (v11)
- [`results/paired_tests/`](results/paired_tests/) — Wilcoxon JSONs
- [`results/`](results/) — raw artefacts by phase (P0, P1, P1_5b, P1_5b_post_f27, P2, P5_bugfix, P8_sota, B3_baselines, B3_validation, B5, F2_fl_diagnostic, F27_*, F41_preencoder, baselines/, hgi/, perf_compare/, phase1_perfold/, phase2_logs/, probe/, figs/)

### "What previous experiments led us here?"

- [`findings/`](findings/) — paper-supporting per-experiment findings (the F-trail) — closed read-only history
- [`findings/archive/F50/`](findings/archive/F50/) — further-archived findings

### "What active follow-up studies are running?"

- [`studies/`](studies/) — active research tracks layered on check2hgi
  - [`studies/canonical_improvement/`](studies/canonical_improvement/) — canonical Check2HGI improvement (18-experiment slate, 5 tiers)
  - [`studies/merge_design/`](studies/merge_design/) — Designs A-M / Levers 1-6 / Phase 11 audit trail
  - [`studies/hgi_category_injection/`](studies/hgi_category_injection/) — HGI POI2Vec category-injection on AZ (CLOSED, falsified, kept pending FL/CA/TX revisit)

### "I'm running on machine X — where do I look?"

- [`infra/`](infra/) — operational documentation
  - [`infra/local/`](infra/local/) — M4 Pro / MPS / .venv
  - [`infra/runpod/`](infra/runpod/) — RunPod (CUDA, RTX 4090)
  - [`infra/colab/`](infra/colab/) — Colab T4 (notebooks, study runner, drive)
  - [`infra/lightning/`](infra/lightning/) — Lightning.ai pods (A100/H100)
  - [`infra/h100/`](infra/h100/) — H100 SSH bare-metal
  - [`infra/data/`](infra/data/) — Drive download / data movement

### "Where's the paper?"

- [`../articles/[BRACIS]_Beyond_Cross_Task/`](../articles/[BRACIS]_Beyond_Cross_Task/) — BRACIS 2026 submission working folder
- [`paper/`](paper/) — section drafts (methods, results, limitations, appendix_methodology)
- [`thesis/`](thesis/) — thesis options (A / B)
- [`BRACIS_GUIDE.md`](BRACIS_GUIDE.md) — conference submission guide
- [`check2hgi_overview.tex`](check2hgi_overview.tex) — paper LaTeX figure asset

### "What does the framework support? (background reading)"

- [`context/`](context/) — task / embedding / architecture / optimizer / head background
- [`datasets/`](datasets/) — dataset reference
- [`baselines/`](baselines/) — external baselines (overview + per-task audits)
- [`plans/`](plans/) — non-archive ablation plans
- [`reports/`](reports/) — status reports
- [`scope/`](scope/) — scoping decisions
- [`review/`](review/) — dated critical reviews
- [`launch_plans/`](launch_plans/) — historical launch plans (durable ops recipes are in `infra/`)

### "Where are open issues?"

- [`issues/`](issues/) — generic project issues
- [`issues/check2hgi/`](issues/check2hgi/) — check2hgi study-specific issues

### "Where's old / closed / archived work?"

- [`archive/`](archive/) — archived studies and snapshots
  - [`archive/fusion-study/`](archive/fusion-study/) — predecessor fusion study (closed 2026-05-14)
  - [`archive/check2hgi-post-paper-closure-2026-05-01/`](archive/check2hgi-post-paper-closure-2026-05-01/) — paper-closure snapshot
  - [`archive/check2hgi-pre-b3-framing/`](archive/check2hgi-pre-b3-framing/), [`...research-pre-b3/`](archive/check2hgi-research-pre-b3/), [`...research-pre-b5/`](archive/check2hgi-research-pre-b5/), [`...phases-original/`](archive/check2hgi-phases-original/), [`...v1-wip-mixed-scope/`](archive/check2hgi-v1-wip-mixed-scope/), [`...2026-04-20-status-reports/`](archive/check2hgi-2026-04-20-status-reports/)
  - [`archive/check2hgi-README-pre-promotion.md`](archive/check2hgi-README-pre-promotion.md) — original check2hgi study landing page (pre-2026-05-14 promotion)
  - [`archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md`](archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md), [`HGI_PERFORMANCE_IMPROVEMENT_PLAN.md`](archive/HGI_PERFORMANCE_IMPROVEMENT_PLAN.md), [`KNOWLEDGE_BASE_2026-04-13.md`](archive/KNOWLEDGE_BASE_2026-04-13.md)
  - [`archive/MERGE_REORG_PLAN_2026-05-14.md`](archive/MERGE_REORG_PLAN_2026-05-14.md) — full record of the 2026-05-14 docs reorganization (this folder)
- [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md) — legacy findings (pre-bugfix; revalidate, don't trust)

---

## Folder semantics — what each subdir means

| Subdir | Purpose | Reader question it answers |
|---|---|---|
| `docs/` root files | What the project IS right now (check2hgi study briefing, claims, north star, change log) | "Where are we now?" |
| `docs/results/` | Canonical paper-facing numbers + raw run artefacts by phase | "What are the canonical numbers?" |
| `docs/findings/` | Per-experiment findings (the F-trail) supporting the BRACIS paper — closed read-only history | "What previous experiments led us here?" |
| `docs/studies/` | **ACTIVE follow-up studies** layered on check2hgi (each is its own track) | "What's still being worked on?" |
| `docs/archive/` | Closed studies / superseded snapshots (fusion-study, paper-closure, pre-b3, etc.) | "Where's the historical knowledge?" |
| `docs/infra/` | Operational docs (RunPod, Colab, Lightning, H100, local, Drive) | "I'm on machine X — where do I look?" |
| `docs/baselines/` | External baselines (overview + per-task audits) | "What are we comparing against?" |
| `docs/paper/` | Paper section drafts | "Where are the writing pieces?" |
| `docs/scope/`, `docs/review/`, `docs/launch_plans/` | Scoping decisions, dated reviews, historical launch plans | "Why did we make these decisions?" |
| `docs/context/`, `docs/datasets/`, `docs/thesis/`, `docs/plans/`, `docs/reports/` | Project-wide background | "Background reading" |
| `docs/issues/` | Generic + check2hgi-nested subdir | "Open / known issues" |

## Reorg history

This `docs/` layout was established 2026-05-14 by the merge & reorg pass. Full record: [`archive/MERGE_REORG_PLAN_2026-05-14.md`](archive/MERGE_REORG_PLAN_2026-05-14.md).
