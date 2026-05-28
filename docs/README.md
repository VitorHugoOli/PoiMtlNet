# docs/ — Project Documentation

**Project:** MTLnet — multi-task learning for POI prediction (category + next-POI).
**Primary study:** **check2hgi** — check-in-level Check2HGI substrate (paper at BRACIS 2026).

> **Single source of truth for paper numbers**: [`results/RESULTS_TABLE.md §0`](results/RESULTS_TABLE.md).
> **Single source of truth for paper prose**: [`../articles/[BRACIS]_Beyond_Cross_Task/`](../articles/[BRACIS]_Beyond_Cross_Task/) — read `AGENT.md` first.
> **Operational documentation** (Colab, RunPod, Lightning, H100, local, Drive): [`infra/`](infra/).
> **Cross-study outcomes log**: [`studies/log.md`](studies/log.md) (one line per closure / direction shift).

> 🔬 **2026-05-28 — Active studies caveat (read before citing v11 numbers).** Two studies launched after v11 paper closure are expected to produce better numbers: [`studies/mtl_improvement/`](studies/mtl_improvement/) (architectural axis, branch `mtl-improve`) and [`studies/substrate-protocol-cleanup/`](studies/substrate-protocol-cleanup/) (substrate + protocol axis, main). Both are **ACTIVE / OPEN**. When they land champions, the plan is to **re-run the full §0 pipeline from scratch** at all five states. Already-known caveats on v11: F1 selector fix recovers +5.6 pp FL multi-seed (C21 RESOLVED); composite STL c2hgi-cat + STL HGI-reg ceiling +7-12 pp vs MTL@disjoint (Phase 3 §4.2 ESTABLISHED). See [`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md) top banner for the full list.

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

- [`studies/`](studies/) — research tracks layered on check2hgi (active + closed). Cross-study outcomes log at [`studies/log.md`](studies/log.md).
  - **ACTIVE** [`studies/mtl_improvement/`](studies/mtl_improvement/) — architectural axis (T0-T8 chain: backbones, loss, batch, LR, α, heads, multi-seed champion). LAUNCHED 2026-05-16, branch `mtl-improve`.
  - **ACTIVE** [`studies/substrate-protocol-cleanup/`](studies/substrate-protocol-cleanup/) — substrate + protocol cleanup (log_T-KD multi-seed promotion, Designs B/J/Lever 4/Lever 5 MTL+F1 re-eval, §4.1 per-task 3-snapshot routing, §4.4 freeze-reg-after-peak, P4 K/V capacity-stealing pilot, window/mask audit). LAUNCHED 2026-05-28, main. Small states only.
  - **CLOSED 2026-05-24 v6 final** [`studies/mtl-protocol-fix/`](studies/mtl-protocol-fix/) — F1 selector fix (+5.6 pp FL multi-seed deployable; C21 RESOLVED); P4 frozen-cat identifies residual MTL-vs-STL reg gap as architectural; Phase 3 outcomes: §4.5 log_T-KD PROMOTED, §4.6 sampler FALSIFIED, §4.2 composite ESTABLISHED (+7-12 pp project headline on reg). Closure verdict: [`results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md). Deferred-work map: [`studies/mtl-protocol-fix/DEFERRED_WORK.md`](studies/mtl-protocol-fix/DEFERRED_WORK.md).
  - **CLOSED 2026-05-19** [`studies/canonical_improvement/`](studies/canonical_improvement/) — 6 tiers, 26 mechanism families, substrate axis exhausted ±0.8 pp; surfaced C21. ⚠ Tier 6 FL-MTL artefacts carry stale-log_T caveat — see [`CONCERNS.md` C22](CONCERNS.md#c22).
  - [`studies/mtl-exploration/`](studies/mtl-exploration/) — support / scaffold study; F1/F2/F3 memo superseded by `mtl-protocol-fix`.
  - [`studies/merge_design/`](studies/merge_design/) — Designs A-M / Levers 1-6 / Phase 11 audit trail. Lever 5 orphan rescued by `substrate-protocol-cleanup` Tier B4.
  - [`studies/hgi_category_injection/`](studies/hgi_category_injection/) — HGI POI2Vec category-injection on AZ (CLOSED 2026-05-04, falsified, kept pending FL/CA/TX revisit).

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
- [`thesis/`](thesis/) — thesis options (A / B)
- [`BRACIS_GUIDE.md`](BRACIS_GUIDE.md) — conference submission guide
- [`context/check2hgi_overview.tex`](context/check2hgi_overview.tex) — paper LaTeX figure asset (Check2HGI architecture)

### "What's being deferred to future work?"

- [`future_works/`](future_works/) — forward-looking memos (work to-do-later that's not yet a study)

### "What does the framework support? (background reading)"

- [`context/`](context/) — task, datasets, splits, metrics, embeddings, fusion, MTL architectures, optimizers, task heads, and the Check2HGI architecture figure (`check2hgi_overview.tex`)
- [`baselines/`](baselines/) — external baselines (overview + per-task audits)
- [`reports/`](reports/) — status reports

### "Where are open issues?"

- [`issues/check2hgi/`](issues/check2hgi/) — check2hgi study bug-audit log (some still partial-open)

### "Where's old / closed / archived work?"

- [`archive/`](archive/) — archived studies, paper drafts, plans, reviews, scoping memos, snapshots
  - **Predecessor study**: [`archive/fusion-study/`](archive/fusion-study/) (closed 2026-05-14)
  - **Pre-promotion check2hgi snapshots**: [`archive/check2hgi-post-paper-closure-2026-05-01/`](archive/check2hgi-post-paper-closure-2026-05-01/), [`...pre-b3-framing/`](archive/check2hgi-pre-b3-framing/), [`...research-pre-b3/`](archive/check2hgi-research-pre-b3/), [`...research-pre-b5/`](archive/check2hgi-research-pre-b5/), [`...phases-original/`](archive/check2hgi-phases-original/), [`...v1-wip-mixed-scope/`](archive/check2hgi-v1-wip-mixed-scope/), [`...2026-04-20-status-reports/`](archive/check2hgi-2026-04-20-status-reports/)
  - **Pre-promotion landing page**: [`archive/check2hgi-README-pre-promotion.md`](archive/check2hgi-README-pre-promotion.md)
  - **Paper drafts (pre-BRACIS LaTeX port)**: [`archive/check2hgi-paper-drafts-pre-bracis/`](archive/check2hgi-paper-drafts-pre-bracis/) (methods, results, limitations, appendix_methodology — work landed in `articles/[BRACIS]_*/src/sections/`)
  - **Dated reviews 2026-04**: [`archive/check2hgi-reviews-2026-04/`](archive/check2hgi-reviews-2026-04/) (7 critical-review snapshots)
  - **Scope memos 2026-04**: [`archive/check2hgi-scope-memos-2026-04/`](archive/check2hgi-scope-memos-2026-04/) (CH14/CH10/P0.2 + CH15-rename memos)
  - **Launch plans 2026-04**: [`archive/check2hgi-launch-plans-2026-04/`](archive/check2hgi-launch-plans-2026-04/) (CA+TX upstream + F33/F36 staging plans)
  - **Fusion-era ablation plans**: [`archive/check2hgi-fusion-era-ablation-plans/`](archive/check2hgi-fusion-era-ablation-plans/) (HEAD/HYPERPARAM ablation plans superseded by canonical_improvement)
  - **Generic issues pre-promotion**: [`archive/check2hgi-generic-issues-pre-promotion/`](archive/check2hgi-generic-issues-pre-promotion/) (8 early proposals/literature surveys; superseded by current docs)
  - **Other**: [`archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md`](archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md), [`HGI_PERFORMANCE_IMPROVEMENT_PLAN.md`](archive/HGI_PERFORMANCE_IMPROVEMENT_PLAN.md), [`KNOWLEDGE_BASE_2026-04-13.md`](archive/KNOWLEDGE_BASE_2026-04-13.md), [`CHECK2HGI_MTL_OVERVIEW-pre-promotion.md`](archive/CHECK2HGI_MTL_OVERVIEW-pre-promotion.md), [`CHECK2HGI_MTL_BRANCH_PLAN-pre-promotion.md`](archive/CHECK2HGI_MTL_BRANCH_PLAN-pre-promotion.md)
  - **Reorg records**: [`archive/MERGE_REORG_PLAN_2026-05-14.md`](archive/MERGE_REORG_PLAN_2026-05-14.md), [`archive/REORG_HANDOFF_2026-05-14.md`](archive/REORG_HANDOFF_2026-05-14.md), [`archive/REORG_REFSWEEP_CATALOG_2026-05-14.md`](archive/REORG_REFSWEEP_CATALOG_2026-05-14.md)
- [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md) — legacy findings (pre-bugfix; revalidate, don't trust)

---

## Folder semantics — what each subdir means

| Subdir | Purpose | Reader question it answers |
|---|---|---|
| `docs/` root files | What the project IS right now (check2hgi study briefing, claims, north star, change log) | "Where are we now?" |
| `docs/results/` | Canonical paper-facing numbers + raw run artefacts by phase | "What are the canonical numbers?" |
| `docs/findings/` | Per-experiment findings (the F-trail) supporting the BRACIS paper — closed read-only history | "What previous experiments led us here?" |
| `docs/studies/` | **ACTIVE follow-up studies** layered on check2hgi (each is its own track) | "What's still being worked on?" |
| `docs/future_works/` | Forward-looking memos (work to-do-later that's not yet a study) | "What's deferred?" |
| `docs/archive/` | Closed studies, paper drafts, plans, reviews, scoping memos, snapshots | "Where's the historical knowledge?" |
| `docs/infra/` | Operational docs (RunPod, Colab, Lightning, H100, local, Drive) | "I'm on machine X — where do I look?" |
| `docs/baselines/` | External baselines (overview + per-task audits) | "What are we comparing against?" |
| `docs/context/`, `docs/thesis/`, `docs/reports/` | Project-wide background (incl. tasks, datasets, splits, metrics, embeddings, MTL archs, etc.) | "Background reading" |
| `docs/issues/check2hgi/` | Bug-audit log — partial-open + fixed | "What bugs did we find / are tracking?" |

## Reorg history

This `docs/` layout was established 2026-05-14 by the merge & reorg pass. Full record: [`archive/MERGE_REORG_PLAN_2026-05-14.md`](archive/MERGE_REORG_PLAN_2026-05-14.md).
