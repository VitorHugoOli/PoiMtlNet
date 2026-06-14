# Merge & Reorganization Plan — `worktree-check2hgi-mtl` → `main`

**Date drafted:** 2026-05-14
**Status:** DRAFT — pending user approval (after advisor review)
**Author:** Claude (Opus 4.7, 1M)

This is the consolidated Phase 2 plan covering merge strategy (2a), code conflicts (2b), fusion archival (2c), check2hgi promotion (2d), ops/infra consolidation (2e), branch cleanup (2f), and CLAUDE.md updates (2g).

A central finding from Phase 1 reframes the work: **`worktree-check2hgi-mtl` is a content-superset of `main` (372 commits ahead, 1 commit behind), and main's only unique commit is a stale revert of `docs/studies/archive/fusion/results/P0/folds/frozen.json`.** No fusion-study knowledge sits on main that isn't already on this branch. The merge itself is therefore mechanical; the high-value/high-risk work is the **docs reorganization** that lands check2hgi as the project's primary study and tucks fusion into archive.

---

## 2a · Merge strategy

**Recommendation: `git merge --no-ff worktree-check2hgi-mtl` from `main`, accepting branch's `frozen.json` on conflict.**

Why:
- **Preserves the full 372-commit history** of the check2hgi study on main — every BRACIS doc-pass, every F-trail experiment, every Phase-11 substrate probe is reachable from `main` after merge.
- **Single readable merge commit** as the integration point — bisection-friendly, reviewable as one diff.
- **No force-push** required to main (avoids destroying the `bump: froxen file` commit, even though it's effectively a stale revert).
- The one expected conflict (`docs/studies/archive/fusion/results/P0/folds/frozen.json`) is resolved trivially: keep `worktree-check2hgi-mtl`'s newer (2026-04-18) version with the cleaned-up entries.

Alternatives considered:
- **Fast-forward (impossible)**: main has +1 commit, so FF is blocked unless we force-update.
- **Squash merge**: discards 372 commits of context; would lose the BRACIS audit-trail and the per-experiment finding history. Rejected.
- **Rebase main onto branch + FF**: would require force-push to main and discard the stale-revert commit. Cleaner history but more destructive than necessary; the merge commit is fine as a once-only event.
- **Cherry-pick branch onto main**: 372 cherry-picks is impractical and loses merge structure.

The reorg work (fusion archive, check2hgi promotion, ops/infra consolidation, CLAUDE.md update) will land as **commits on this branch first**, then the single merge brings everything to main atomically.

Order on this branch (revised after advisor B2/B3 + suggestion to commit per-step):
1. **Pre-flight** (Phase 3-0):
   - `git tag pre-reorg-2026-05-14 worktree-check2hgi-mtl` (recovery anchor).
   - Verify `h100/pervisit-fl-ca-tx-results` integration. Concrete check: grep `docs/studies/check2hgi/results/` for FL/CA/TX × 5 folds × 3 cells (= 45 per-fold CH19 JSONs). If absent → `git cherry-pick a858177 d714ce0` first.
   - Repo-wide grep for cross-refs that the reorg will break: especially `articles/[BRACIS]_Beyond_Cross_Task/**` and `notebooks/**` for `docs/studies/check2hgi/...` paths. Record the list — sed sweep happens in steps 3 and 5 below.
2. Fusion: in-place consolidation (ARCHIVE_NOTE only, no drops) → `git mv docs/studies/archive/fusion docs/archive/fusion-study`. **Single commit.**
3. Ops/infra consolidation into `docs/infra/` + breadcrumbs at old locations + ref-sweep pass for moved RUNPOD/COLAB/H100 references. **Single commit.**
4. Promote check2hgi docs to `docs/` (2d) — **after explicit user approval**. Includes the `research/` → `findings/` rename + ref-sweep pass updating every `docs/studies/check2hgi/...` path across `articles/`, `notebooks/`, `CLAUDE.md`, `README.md`, anywhere else found in step 1's grep. **Single commit (or split move-commit + sed-commit if cleaner).**
5. Update `CLAUDE.md` (2g) + delete `CLAUDE.local.md`. **Single commit.**
6. Optional: `pytest -x` sanity (tests don't reference docs/ paths; should pass unchanged).
7. Single merge commit on `main`: `git merge --no-ff worktree-check2hgi-mtl -m "Merge check2hgi study into main; archive fusion; reorganize docs"`. Resolve `frozen.json` in favor of branch.
8. Branch cleanup (2f) after merge lands. Tag each deleted branch as `archived/<branch-name>` first (insurance — keeps SHA reachable by name).

---

## 2b · Code conflict resolution

There is **one** code/asset conflict in the merge:

| Path | Stake | Resolution | Preserved | Dropped |
|---|---|---|---|---|
| `docs/studies/archive/fusion/results/P0/folds/frozen.json` | Fusion fold registry — main has 2 stale entries (`alabama/fusion/category`, `alabama/fusion/next`) re-added on 2026-05-13; branch removed them on 2026-04-18 with newer `updated_at` | **Keep branch version (newer cleanup)** | Cleaned, current registry state | The stale main entries (their fold artifacts may not exist anymore) |

**No source-code conflicts.** Every `src/`, `scripts/`, `pipelines/`, `experiments/` change on main is already on this branch.

**Forward compatibility with fusion (no action required, but flagged so we don't regress):**
- `src/configs/paths.py` — `EmbeddingEngine.FUSION` enum + FUSION routing in `IoPaths`: KEEP intact.
- `src/data/inputs/fusion.py` — fusion input builder: KEEP intact.
- `experiments/full_fusion_ablation.py` — fusion ablation harness: KEEP intact.
- `pipelines/fusion.pipe.py` — fusion pipeline: KEEP intact.
- `output/fusion/` symlink convention (per memory `fusion_embedding_design`): KEEP supported.

These ride along with the merge unchanged.

---

## 2c · Fusion study consolidation plan

Goal: make `docs/studies/archive/fusion/` coherent as a closed body of work, then move under `docs/archive/`.

### Step 1 — In-place consolidation

**Add (1 file):**
- `docs/studies/archive/fusion/ARCHIVE_NOTE.md` — closure-status banner: when archived, what the final status was, what concepts/results survived (point to `KNOWLEDGE_SNAPSHOT.md`, `STUDY_OVERVIEW.md`, `CLAIMS_AND_HYPOTHESES.md`, results), what work was superseded by check2hgi, and how to reach back into the archive if needed.

**Update (1 file):**
- `docs/studies/archive/fusion/README.md` — add a CLOSURE banner at top pointing to `ARCHIVE_NOTE.md`. The existing body content (entry point, master plan reference, claim catalog, phase navigation) stays.

**Drop: NOTHING.** (Revised after advisor B1.) Earlier draft proposed dropping `state.json`, `machines.yaml`, `assignments/P1_2026-04-16_screen/`. Keeping them — they are the lab-notebook record of how the multi-machine grid was conducted (who ran what, where, in what order, with what outcome). Combined size is a few KB; cost of keeping is nothing; preserving them honors the "concepts/results/findings survive" principle.

**Keep all of the following (concepts/results/findings + reusable infra):**
- All top-level docs: `AGENT_CONTEXT.md`, `MASTER_PLAN.md`, `COORDINATOR.md`, `HANDOFF.md`, `KNOWLEDGE_SNAPSHOT.md`, `QUICK_REFERENCE.md`, `STUDY_OVERVIEW.md`, `CLAIMS_AND_HYPOTHESES.md`.
- `archive/{ablation_studies, full_ablation_study}/` — pre-fusion-study archive (further-archived findings).
- `coordinator/` (integrity_checks.md, state_schema.md) — design specs.
- `issues/{HGI_LEAKAGE_AUDIT, HGI_LEAKAGE_EXPLAINED, P1_METHODOLOGY_FLAWS}.md` — durable diagnoses.
- `phases/P0..P6` — study design.
- `plans/CRITICAL_REVIEW_ATTACK_PLAN.md`.
- `results/P0/...` — all run summaries, integrity files, leakage-ablation outputs (the actual scientific record).

### Step 2 — Move to `docs/archive/`

**Move:** `docs/studies/archive/fusion/` → `docs/archive/fusion-study/` (preserves git history via `git mv`).

After move, `docs/studies/` contains only `check2hgi/` (until 2d empties it further).

### Justification (everything preserved; framing only)

- **Concepts preserved**: AGENT_CONTEXT.md, KNOWLEDGE_SNAPSHOT.md, STUDY_OVERVIEW.md, MASTER_PLAN.md.
- **Results preserved**: full `results/P0/` tree (CSVs, JSONs, integrity, leakage-ablation).
- **Findings preserved**: CLAIMS_AND_HYPOTHESES.md, issues/ (leakage diagnoses), plans/CRITICAL_REVIEW_ATTACK_PLAN.md.
- **Lab-notebook records preserved**: state.json (run timeline), machines.yaml (state-pinning contract), assignments/ (who-ran-what-where forensic record).
- **Closure framing added**: ARCHIVE_NOTE.md + README banner so the archive reads as closed, not abandoned.

---

## 2d · check2hgi promotion + docs reorganization (REVISED) ⚠ NEEDS EXPLICIT APPROVAL

**Revised goal (per user feedback 2026-05-14):**
1. check2hgi's core docs sit at `docs/` level (it IS the project's main study).
2. The three open research workstreams (`canonical_improvement`, `merge_design`, `hgi_category_injection`) get promoted to `docs/studies/` as standalone follow-up studies (still being worked on).
3. The whole `docs/` tree is reorganized for future-readability — clear answers to "where are we now / canonical numbers / previous experiments / active follow-up studies / ops / paper / archive".

### Mental model — what each docs/ subdir means

| Subdir | Purpose | Reader question it answers |
|---|---|---|
| `docs/` root files | What the project IS right now (check2hgi study briefing, claims, north star, change log) | "Where are we now?" |
| `docs/results/` | Canonical paper-facing numbers + raw run artefacts by phase | "What are the canonical numbers?" |
| `docs/findings/` | Per-experiment findings (the F-trail) that support the BRACIS paper — closed read-only history | "What previous experiments led us here?" |
| `docs/studies/` | **ACTIVE follow-up studies** layered on check2hgi (each is its own track) | "What's still being worked on?" |
| `docs/archive/` | Closed studies / superseded snapshots (fusion-study, paper-closure, pre-b3, etc.) | "Where's the historical knowledge?" |
| `docs/infra/` | Operational docs (RunPod, Colab, Lightning, H100, local, Drive) | "I'm on machine X — where do I look?" |
| `docs/baselines/` | External baselines (overview + per-task audits) | "What are we comparing against?" |
| `docs/paper/` | Paper section drafts (methods, results, limitations) | "Where are the writing pieces?" |
| `docs/scope/`, `docs/review/`, `docs/launch_plans/` | Scoping decisions, dated reviews, historical launch plans | "Why did we make these decisions?" |
| `docs/context/`, `docs/datasets/`, `docs/thesis/`, `docs/plans/`, `docs/reports/` | Project-wide background (unchanged from current location) | "Background reading" |
| `docs/issues/` | Generic issues + check2hgi-nested subdir | "Open / known issues" |

### Top-level docs (PROMOTE to `docs/`)

| From | To | Notes |
|---|---|---|
| `docs/studies/check2hgi/README.md` | `docs/README.md` (rewritten as navigation landing — see below) | NEW navigation map; merges old README's content into a navigable structure |
| `docs/studies/check2hgi/AGENT_CONTEXT.md` | `docs/AGENT_CONTEXT.md` | Long-form briefing |
| `docs/studies/check2hgi/NORTH_STAR.md` | `docs/NORTH_STAR.md` | Champion config |
| `docs/studies/check2hgi/CHANGELOG.md` | `docs/CHANGELOG.md` | Timeline of findings |
| `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md` | `docs/CLAIMS_AND_HYPOTHESES.md` | Claim catalog |
| `docs/studies/check2hgi/CONCERNS.md` | `docs/CONCERNS.md` | Risk audit |
| `docs/studies/check2hgi/FINAL_SURVEY.md` | `docs/FINAL_SURVEY.md` | 5-state matrix |
| `docs/studies/check2hgi/MTL_ARCHITECTURE_JOURNEY.md` | `docs/MTL_ARCHITECTURE_JOURNEY.md` | Supplementary narrative |
| `docs/studies/check2hgi/PAPER_BASELINES_STRATEGY.md` | `docs/PAPER_BASELINES_STRATEGY.md` | Baseline-table mapping |

### Subdirectories (PROMOTE to `docs/`)

| From | To | Conflict / Notes |
|---|---|---|
| `docs/studies/check2hgi/results/` | `docs/results/` | No conflict |
| `docs/studies/check2hgi/baselines/` | `docs/baselines/` (merge with existing) | Existing `docs/baselines/BASELINE.md` becomes overview alongside next_category/+next_region/ audits + new `docs/baselines/README.md` (study-side audit map) |
| `docs/studies/check2hgi/paper/` | `docs/paper/` | No conflict |
| `docs/studies/check2hgi/issues/` | `docs/issues/check2hgi/` | Nested to avoid clash with existing 7 generic issues |
| `docs/studies/check2hgi/scope/` | `docs/scope/` | No conflict |
| `docs/studies/check2hgi/review/` | `docs/review/` | No conflict |
| `docs/studies/check2hgi/launch_plans/` | `docs/launch_plans/` | No conflict (durable infra recipes already extracted to `docs/infra/` in 2e) |

### NEW: Carve out the three open research workstreams to `docs/studies/`

The current `docs/studies/check2hgi/research/` contains:
- **3 open workstreams** (each its own subdir): `canonical_improvement/`, `merge_design/`, `hgi_category_injection/`.
- **68 paper-supporting finding files** (F-trail) at the root of `research/`.
- `research/archive/F50/` — further-archived material.
- `research/figs/` — 2 finding figures.

Plan:

| From | To | Why |
|---|---|---|
| `docs/studies/check2hgi/research/canonical_improvement/` | `docs/studies/archive/canonical_improvement/` | Active 18-experiment track on its own branch (`check2hgi-canonical-improve`) — promoted to first-class study |
| `docs/studies/check2hgi/research/merge_design/` | `docs/studies/merge_design/` | Audit trail with open levers (Lever 6 active), Phase 11 plan in flight — promoted to first-class study |
| `docs/studies/check2hgi/research/hgi_category_injection/` | `docs/studies/archive/hgi_category_injection/` | Completed sub-study (AZ falsified) but per user request kept in `studies/` (not archived) — may be re-opened. **Add `STATUS.md` (or banner in `INDEX.md`) reading: "CLOSED — AZ falsified 2026-05-04. Kept under `studies/` pending decision to revisit on FL/CA/TX. Do NOT treat as active without an explicit re-open commit."** |
| `docs/studies/check2hgi/research/*.md` and `*.json` (68 files) | `docs/findings/` | Paper-supporting F-trail (closed evidence) |
| `docs/studies/check2hgi/research/archive/F50/` | `docs/findings/archive/F50/` | Further-archived findings stay alongside their parent F-trail |
| `docs/studies/check2hgi/research/figs/` | `docs/findings/figs/` | Finding figures stay alongside the F-trail |

### Subdirectory: archive

| From | To | Notes |
|---|---|---|
| `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/` | `docs/archive/check2hgi-post-paper-closure-2026-05-01/` | |
| `docs/studies/check2hgi/archive/{pre_b3_framing, research_pre_b3, research_pre_b5, phases_original, v1_wip_mixed_scope, 2026-04-20_status_reports}/` | `docs/archive/check2hgi-{name}/` | One promoted sibling each |
| Existing `docs/archive/{HGI_HYPERPARAMETER_TUNING_2026-04-13.md, HGI_PERFORMANCE_IMPROVEMENT_PLAN.md}` | unchanged | Pre-existing archived items stay |

### NEW: `docs/README.md` — navigation landing (REWRITE)

```markdown
# docs/ — Project Documentation

**Project:** MTLnet — multi-task learning for POI prediction (category + next-POI).
**Primary study:** **check2hgi** — check-in-level Check2HGI substrate (paper at BRACIS 2026).

> Single source of truth for paper numbers: [`results/RESULTS_TABLE.md §0`](results/RESULTS_TABLE.md).
> Single source of truth for paper prose: [`../articles/[BRACIS]_Beyond_Cross_Task/`](../articles/[BRACIS]_Beyond_Cross_Task/) — read `AGENT.md` first.

## Navigation by question

### "Where are we now?" (check2hgi study briefing)
- [`AGENT_CONTEXT.md`](AGENT_CONTEXT.md) — long-form study briefing
- [`NORTH_STAR.md`](NORTH_STAR.md) — committed champion config (B9 MTL recipe; H3-alt small-state)
- [`CHANGELOG.md`](CHANGELOG.md) — timeline of findings + lessons
- [`CLAIMS_AND_HYPOTHESES.md`](CLAIMS_AND_HYPOTHESES.md) — claim catalog (paper-facing whitelist banner inside)
- [`CONCERNS.md`](CONCERNS.md) — risk audit log
- [`FINAL_SURVEY.md`](FINAL_SURVEY.md) — substrate-axis 5-state matrix
- [`MTL_ARCHITECTURE_JOURNEY.md`](MTL_ARCHITECTURE_JOURNEY.md) — supplementary narrative (F-trail B3 → F45 → F48-H3-alt → F49 → paper closure)
- [`PAPER_BASELINES_STRATEGY.md`](PAPER_BASELINES_STRATEGY.md) — which baselines appear in which paper table

### "What are the canonical numbers?"
- [`results/RESULTS_TABLE.md`](results/RESULTS_TABLE.md) ⭐ canonical paper-facing numbers (v11)
- [`results/paired_tests/`](results/paired_tests/) — Wilcoxon JSONs
- [`results/{P0,P1,P1_5b,...}/`](results/) — raw run artefacts by phase

### "What previous experiments led us here?"
- [`findings/`](findings/) — per-experiment findings (the F-trail) supporting the BRACIS paper
- [`findings/archive/F50/`](findings/archive/F50/) — further-archived findings

### "What active follow-up studies are running?"
- [`studies/`](studies/) — active research tracks layered on check2hgi
  - [`studies/archive/canonical_improvement/`](studies/archive/canonical_improvement/) — 18-experiment slate to improve canonical Check2HGI (branch `check2hgi-canonical-improve`)
  - [`studies/merge_design/`](studies/merge_design/) — Designs A-M / Levers 1-6 / Phase 11 audit trail
  - [`studies/archive/hgi_category_injection/`](studies/archive/hgi_category_injection/) — HGI POI2Vec category-injection on AZ (falsified, archived in studies/ pending revisit)

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
- [`paper/`](paper/) — paper section drafts (methods, results, limitations, appendix)
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
- [`issues/`](issues/) — generic issues + `issues/check2hgi/` (study-specific)

### "Where's old / closed / archived work?"
- [`archive/`](archive/) — archived studies and snapshots
  - [`archive/fusion-study/`](archive/fusion-study/) — predecessor fusion study (closed)
  - [`archive/check2hgi-post-paper-closure-2026-05-01/`](archive/check2hgi-post-paper-closure-2026-05-01/) — paper-closure snapshot
  - [`archive/check2hgi-pre-b3-framing/`](archive/check2hgi-pre-b3-framing/), [`...research-pre-b3/`](archive/check2hgi-research-pre-b3/), [`...research-pre-b5/`](archive/check2hgi-research-pre-b5/), [`...phases-original/`](archive/check2hgi-phases-original/), [`...v1-wip-mixed-scope/`](archive/check2hgi-v1-wip-mixed-scope/), [`...2026-04-20-status-reports/`](archive/check2hgi-2026-04-20-status-reports/)
  - [`archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md`](archive/HGI_HYPERPARAMETER_TUNING_2026-04-13.md), [`HGI_PERFORMANCE_IMPROVEMENT_PLAN.md`](archive/HGI_PERFORMANCE_IMPROVEMENT_PLAN.md), [`KNOWLEDGE_BASE_2026-04-13.md`](archive/KNOWLEDGE_BASE_2026-04-13.md)
- [`PAPER_FINDINGS.md`](PAPER_FINDINGS.md) — legacy findings (pre-bugfix; revalidate, don't trust)
```

### NEW: `docs/studies/README.md` — studies landing

```markdown
# docs/studies/ — Active follow-up studies on check2hgi

This folder hosts ongoing research tracks layered on top of the primary check2hgi study (`docs/` root). Each subdir is a self-contained study with its own briefing, design, log, and findings.

## Active studies

| Study | Status | Branch (post-merge) | Read first |
|---|---|---|---|
| [`canonical_improvement/`](canonical_improvement/) | ACTIVE — 18 experiments across 5 tiers | branch from `main` (`check2hgi-canonical-improve` not yet created) | `AGENT_PROMPT.md` |
| [`merge_design/`](merge_design/) | ACTIVE-CLOSING — Designs A-M and Levers 1-6 falsified or saturated; Phase 11 plan in flight | (no dedicated branch) | `STATE.md` |
| [`hgi_category_injection/`](hgi_category_injection/) | **CLOSED** (AZ falsified 2026-05-04) — kept here pending decision to revisit on FL/CA/TX. **Do NOT treat as active without an explicit re-open commit.** | (no dedicated branch) | `INDEX.md` + `STATUS.md` |

## `studies/` vs `findings/` vs `archive/`

> **`studies/`** = active or pending-decision research tracks (still being worked on, or recently closed but kept here pending re-open).
> **`findings/`** = closed per-experiment findings supporting the BRACIS paper (read-only history).
> **`archive/`** = fully closed studies and snapshots, unlikely to be re-opened.

## Archive policy

Studies in this folder are *active or pending decision*. When a study is fully closed and unlikely to be re-opened, `git mv` it to `docs/archive/<study>-closed-YYYY-MM-DD/`.

## Where do paper-supporting findings go?

Closed per-experiment findings that support the BRACIS paper (the "F-trail") live at [`docs/findings/`](../findings/), not here.
```

### NEW: `docs/findings/README.md` — findings landing

```markdown
# docs/findings/ — Paper-supporting per-experiment findings (F-trail)

Closed evidence supporting the check2hgi BRACIS 2026 paper. Each file documents a single experiment's findings (B-series early experiments, F-series ablations, F50-tier decompositions, B5/B7/B9/F21c/F27/F37/F40/F41/F44-F48/F49/F50/F51/...).

This is **read-only history**. Active research tracks live at [`docs/studies/`](../studies/).

## `findings/` vs `results/`

> **`findings/`** = narrative + analysis with conclusions (.md). What we learned.
> **`results/`** = canonical numerical artifacts (CSV/JSON tables, RESULTS_TABLE.md, paired-test JSONs). What the numbers were.

Some files in `findings/` have "RESULTS" in the name (e.g., `F50_RESULTS_TABLE.md`, `F49_LAMBDA0_DECOMPOSITION_RESULTS.md`) — they are still narrative findings *about* results, not the canonical numerical source. The canonical source is `results/RESULTS_TABLE.md §0`.

## Index

(see [`figs/`](figs/) for finding figures, [`archive/F50/`](archive/F50/) for further-archived F50-tier material)

[Auto-generated index of *.md and *.json files would go here, or hand-curated by topic.]
```

### What's left under `docs/studies/check2hgi/` after promotion

Nothing. Delete the empty directory. After this:
- `docs/studies/` contains exactly **3 active studies** + a `README.md` landing.
- `docs/` root has the 9 promoted check2hgi study docs + the new `README.md` navigation.
- `docs/findings/` holds the 68-file F-trail + archive/figs.
- `docs/archive/` holds fusion-study + 7 promoted check2hgi-archive subdirs + KNOWLEDGE_BASE-2026-04-13 + 2 pre-existing files.

### Per-file decisions for what's at `docs/` root pre-promotion

These already-at-`docs/`-root files need decisions:

| File | Decision | Why |
|---|---|---|
| `docs/BRACIS_GUIDE.md` | KEEP at `docs/` | Conference submission guide, study-agnostic |
| `docs/PAPER_FINDINGS.md` | KEEP at `docs/` (with stale-warning banner) | Legacy findings, already flagged |
| `docs/KNOWLEDGE_BASE_2026-04-13.md` | MOVE to `docs/archive/KNOWLEDGE_BASE_2026-04-13.md` | Date-stamped snapshot |
| `docs/RUNPOD_GUIDE.md` | MOVE to `docs/infra/runpod/` (handled in 2e) | Ops doc |
| `docs/COLAB_GUIDE.md` | MOVE to `docs/infra/colab/` (handled in 2e) | Ops doc |
| `docs/check2hgi_overview.tex` | KEEP at `docs/` | LaTeX paper asset, project-wide |
| `docs/baselines/BASELINE.md` | KEEP (becomes overview alongside check2hgi audit details) | Merged in promotion |
| `docs/context/`, `docs/datasets/`, `docs/issues/`, `docs/plans/`, `docs/reports/`, `docs/thesis/` | KEEP at current paths | Project-wide background |

### Resulting `docs/` structure (target end-state — REVISED)

```
docs/
├── README.md                          ← navigation landing (NEW, rewritten)
├── AGENT_CONTEXT.md                   ← check2hgi briefing (promoted)
├── NORTH_STAR.md                      ← champion config (promoted)
├── CHANGELOG.md                       ← timeline (promoted)
├── CLAIMS_AND_HYPOTHESES.md           ← claim catalog (promoted)
├── CONCERNS.md                        ← risk audit (promoted)
├── FINAL_SURVEY.md                    ← 5-state matrix (promoted)
├── MTL_ARCHITECTURE_JOURNEY.md        ← supplementary narrative (promoted)
├── PAPER_BASELINES_STRATEGY.md        ← baseline-table mapping (promoted)
├── BRACIS_GUIDE.md                    ← (kept, study-agnostic)
├── PAPER_FINDINGS.md                  ← (kept, legacy)
├── check2hgi_overview.tex             ← (kept, paper asset)
│
├── results/                           ← canonical numbers + raw artefacts (promoted)
│   ├── RESULTS_TABLE.md ⭐
│   ├── paired_tests/
│   └── P0/, P1/, P1_5b/, P1_5b_post_f27/, P2/, P5_bugfix/, P8_sota/, B3_baselines/, B3_validation/, B5/, F2_fl_diagnostic/, F27_cathead_sweep/, F27_validation/, F41_preencoder/, baselines/, hgi/, perf_compare/, phase1_perfold/, phase2_logs/, probe/, figs/, BASELINES_AND_BEST_MTL.md, CH19_PERVISIT_5STATE_SUMMARY.md, SCALE_CURVE.md
│
├── findings/                          ← paper-supporting F-trail (promoted from research/)
│   ├── README.md                      ← findings landing (NEW)
│   ├── F##_*.md and F##_*.json        ← 68 finding files
│   ├── B##_*.md                       ← B-series findings
│   ├── ATTRIBUTION_*, AZ_PERVISIT_WILCOXON.json, ARCH_DELTA_WILCOXON.json, etc.
│   ├── archive/F50/                   ← further-archived
│   └── figs/                          ← finding figures
│
├── studies/                           ← ⭐ ACTIVE follow-up studies (NEW container)
│   ├── README.md                      ← studies landing (NEW)
│   ├── canonical_improvement/         ← (PROMOTED) 18-experiment track
│   ├── merge_design/                  ← (PROMOTED) Designs A-M, Levers, Phase 11
│   └── hgi_category_injection/        ← (PROMOTED) AZ falsified sub-study
│
├── baselines/                         ← (MERGED) overview + per-task audits
│   ├── BASELINE.md                    ← (kept, overview)
│   ├── README.md                      ← (promoted from check2hgi/baselines/)
│   ├── next_category/                 ← (promoted)
│   └── next_region/                   ← (promoted)
│
├── paper/                             ← section drafts (promoted)
│   ├── methods.md, results.md, limitations.md, appendix_methodology.md
├── scope/                             ← (promoted)
├── review/                            ← (promoted)
├── launch_plans/                      ← (promoted, durable ops recipes already in infra/)
│
├── infra/                             ← ⭐ NEW: ops/infra (see 2e)
│   ├── README.md                      ← decision tree (NEW)
│   ├── local/, runpod/, colab/, lightning/, h100/, data/
│
├── archive/                           ← archived knowledge
│   ├── HGI_HYPERPARAMETER_TUNING_2026-04-13.md       (kept)
│   ├── HGI_PERFORMANCE_IMPROVEMENT_PLAN.md           (kept)
│   ├── KNOWLEDGE_BASE_2026-04-13.md                  (moved from docs/)
│   ├── fusion-study/                                 ← from docs/studies/archive/fusion/ (2c)
│   ├── check2hgi-post-paper-closure-2026-05-01/      ← from check2hgi/archive/
│   ├── check2hgi-pre-b3-framing/
│   ├── check2hgi-research-pre-b3/
│   ├── check2hgi-research-pre-b5/
│   ├── check2hgi-phases-original/
│   ├── check2hgi-v1-wip-mixed-scope/
│   └── check2hgi-2026-04-20-status-reports/
│
├── context/                           ← (kept, project-wide background)
├── datasets/                          ← (kept)
├── issues/                            ← generic + check2hgi/ subdir (merged)
├── plans/                             ← (kept)
├── reports/                           ← (kept)
└── thesis/                            ← (kept)
```

### What's preserved vs. dropped

- **Preserved**: every check2hgi finding, claim, result, baseline audit, paper draft, archive snapshot, open-research workstream — all moved with `git mv` (history intact).
- **Dropped**: nothing.

### Explicit Phase 3-3 sub-steps (added per advisor C2)

After all `git mv` operations and the standard ref-sweep, the following **explicit edits** must happen in the same Phase 3-3 commit:

1. **Update `docs/studies/archive/canonical_improvement/AGENT_PROMPT.md`**:
   - Branching instruction: `check2hgi-up` → `main`. Verify the example `git worktree add ../worktree-check2hgi-canonical-improve -b check2hgi-canonical-improve check2hgi-up` becomes `... -b check2hgi-canonical-improve main`.
   - All 10 path references in the "Required reading" table:
     - `docs/studies/check2hgi/research/canonical_improvement/log.md` → `docs/studies/archive/canonical_improvement/log.md`
     - `docs/studies/check2hgi/research/canonical_improvement/INDEX.html` → `docs/studies/archive/canonical_improvement/INDEX.html`
     - `docs/studies/check2hgi/research/merge_design/STUDY_BRIEFING.html` → `docs/studies/merge_design/STUDY_BRIEFING.html`
     - `docs/studies/check2hgi/research/merge_design/STATE.md` → `docs/studies/merge_design/STATE.md`
     - `docs/studies/check2hgi/research/merge_design/AUDIT_HGI_GAP.md` → `docs/studies/merge_design/AUDIT_HGI_GAP.md`
     - `docs/studies/check2hgi/research/canonical_improvement/considerations.md` → `docs/studies/archive/canonical_improvement/considerations.md`
     - `docs/studies/check2hgi/AGENT_CONTEXT.md` → `docs/AGENT_CONTEXT.md`
     - `docs/studies/check2hgi/NORTH_STAR.md` → `docs/NORTH_STAR.md`
     - `research/embeddings/check2hgi/CLAUDE.md` (no change — repo-root `research/`)
     - `research/embeddings/check2hgi/model/variants.py` (no change)
   - Any other internal `docs/studies/check2hgi/...` references in this file.

2. **Author the three new READMEs**:
   - `docs/README.md` — navigation landing (drafted above).
   - `docs/studies/README.md` — studies landing (drafted above).
   - `docs/findings/README.md` — findings landing (drafted above).

3. **Author the closure banner**:
   - `docs/studies/archive/hgi_category_injection/STATUS.md` (or banner in `INDEX.md`) — closure status text per the table above.

4. **Repo-wide ref-sweep** (sed pass + manual review):
   - All `docs/studies/check2hgi/...` references → new path:
     - `docs/studies/check2hgi/{README, AGENT_CONTEXT, NORTH_STAR, CHANGELOG, CLAIMS_AND_HYPOTHESES, CONCERNS, FINAL_SURVEY, MTL_ARCHITECTURE_JOURNEY, PAPER_BASELINES_STRATEGY}.md` → `docs/{name}.md`
     - `docs/studies/check2hgi/results/...` → `docs/results/...`
     - `docs/studies/check2hgi/research/{canonical_improvement,merge_design,hgi_category_injection}/...` → `docs/studies/{name}/...`
     - `docs/studies/check2hgi/research/...` (non-workstream) → `docs/findings/...`
     - `docs/studies/check2hgi/baselines/...` → `docs/baselines/...`
     - `docs/studies/check2hgi/{paper, scope, review, launch_plans}/...` → `docs/{name}/...`
     - `docs/studies/check2hgi/issues/...` → `docs/issues/check2hgi/...`
     - `docs/studies/check2hgi/archive/...` → `docs/archive/check2hgi-...`
   - Files known to need a sweep: `articles/[BRACIS]_Beyond_Cross_Task/**/*.md`, `notebooks/**/*.md`, `notebooks/**/*.ipynb` (text cells), `CLAUDE.md`, repo-root `README.md`, `docs/CLAUDE.md`-equivalents, the three new READMEs themselves.

### Risk flags for this section

- The list above is exhaustive *to the best of my Phase-1 grep*. The Phase 3-0 pre-flight grep (per 2a) is the verification gate — any path the grep finds that's not covered above gets added before commit.
- `git mv` preserves file history, but the canonical_improvement/AGENT_PROMPT.md edit is content-changing — that's an additional small commit on top of the move (or a single compound commit, both fine).

### Top-level docs (PROMOTE to `docs/`)

| From | To | Notes |
|---|---|---|
| `docs/studies/check2hgi/README.md` | `docs/README.md` | Becomes the docs landing. (No existing `docs/README.md`.) Rewrite intro to drop "study folder" framing; point to per-area subdirs. |
| `docs/studies/check2hgi/AGENT_CONTEXT.md` | `docs/AGENT_CONTEXT.md` | Long-form briefing |
| `docs/studies/check2hgi/NORTH_STAR.md` | `docs/NORTH_STAR.md` | Champion config |
| `docs/studies/check2hgi/CHANGELOG.md` | `docs/CHANGELOG.md` | Timeline of findings (NB: there is no other `docs/CHANGELOG.md`) |
| `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md` | `docs/CLAIMS_AND_HYPOTHESES.md` | Claim catalog |
| `docs/studies/check2hgi/CONCERNS.md` | `docs/CONCERNS.md` | Risk audit |
| `docs/studies/check2hgi/FINAL_SURVEY.md` | `docs/FINAL_SURVEY.md` | 5-state matrix |
| `docs/studies/check2hgi/MTL_ARCHITECTURE_JOURNEY.md` | `docs/MTL_ARCHITECTURE_JOURNEY.md` | Supplementary narrative |
| `docs/studies/check2hgi/PAPER_BASELINES_STRATEGY.md` | `docs/PAPER_BASELINES_STRATEGY.md` | Baseline-table mapping |

### Subdirectories (PROMOTE to `docs/`)

| From | To | Conflict / Notes |
|---|---|---|
| `docs/studies/check2hgi/results/` | `docs/results/` | No conflict (no existing `docs/results/`). Includes RESULTS_TABLE.md (canonical), per-phase artefacts, paired-tests. |
| `docs/studies/check2hgi/research/` (68 files: F-trail findings) | `docs/findings/` | Renamed to avoid collision with repo-root `research/` (embedding code). "Findings" is more accurate (these are paper-supporting per-experiment findings, not separate research projects). |
| `docs/studies/check2hgi/baselines/` (next_category/, next_region/, README) | `docs/baselines/` (merge with existing) | **Conflict to resolve**: `docs/baselines/BASELINE.md` is generic external-baseline overview; check2hgi's `baselines/README.md` is detailed audit. Plan: keep both as `docs/baselines/BASELINE.md` (overview, top-level) + `docs/baselines/{next_category,next_region}/` (detailed audits) + `docs/baselines/README.md` (study-side audit map, demoted from current top). |
| `docs/studies/check2hgi/paper/` (4 files: methods, results, limitations, appendix_methodology) | `docs/paper/` | Paper section drafts. No existing `docs/paper/`. |
| `docs/studies/check2hgi/issues/` | `docs/issues/check2hgi/` | **Conflict to resolve**: `docs/issues/` already has 7 generic issues. Nest check2hgi's 8 issues under `docs/issues/check2hgi/` to avoid collision. |
| `docs/studies/check2hgi/scope/` (3 files) | `docs/scope/` | No conflict. |
| `docs/studies/check2hgi/review/` (7 dated reviews) | `docs/review/` | No conflict. |
| `docs/studies/check2hgi/launch_plans/` (2 files: ca_tx_upstream.md, f33_f36_colab.md) | **mixed**: keep historical launch-plan content; durable Colab/Lightning recipes get extracted to `docs/infra/` (2e). | These are dated tactical plans — keep at `docs/launch_plans/` for historical reference. |

### Subdirectory: archive

| From | To | Notes |
|---|---|---|
| `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/` | `docs/archive/check2hgi-post-paper-closure-2026-05-01/` | Historical paper-closure snapshot |
| `docs/studies/check2hgi/archive/{pre_b3_framing, research_pre_b3, research_pre_b5, phases_original, v1_wip_mixed_scope, 2026-04-20_status_reports}/` | `docs/archive/check2hgi-{name}/` | Each pre-existing archive subdir becomes a top-level archived sibling under `docs/archive/`. |
| Existing `docs/archive/{HGI_HYPERPARAMETER_TUNING_2026-04-13.md, HGI_PERFORMANCE_IMPROVEMENT_PLAN.md}` | unchanged in place | Pre-existing archived items stay |

### What's left under `docs/studies/check2hgi/` after promotion

Nothing. Delete the empty directory. `docs/studies/` then contains exactly **zero** active studies — it's a clean container ready for the next layer of work (e.g., a future "ablation-on-check2hgi" study would live at `docs/studies/<name>/`).

### Per-file decisions for what's at `docs/` root pre-promotion

These already-at-`docs/`-root files need decisions:

| File | Decision | Why |
|---|---|---|
| `docs/BRACIS_GUIDE.md` | KEEP at `docs/` | Conference submission guide, study-agnostic. |
| `docs/PAPER_FINDINGS.md` | KEEP at `docs/` (with stale-warning banner) | Legacy findings (pre-bugfix), already flagged. The promoted `docs/CHANGELOG.md` + `docs/results/RESULTS_TABLE.md` supersede it but it's listed in CLAUDE.md as "revalidate, don't trust pre-bugfix" — keep as historical reference. |
| `docs/KNOWLEDGE_BASE_2026-04-13.md` | MOVE to `docs/archive/KNOWLEDGE_BASE_2026-04-13.md` | Date-stamped snapshot. |
| `docs/RUNPOD_GUIDE.md` | MOVE to `docs/infra/runpod/` (handled in 2e) | Ops doc. |
| `docs/COLAB_GUIDE.md` | MOVE to `docs/infra/colab/` (handled in 2e) | Ops doc. |
| `docs/check2hgi_overview.tex` | KEEP at `docs/` | LaTeX figure asset for paper; project-wide. |
| `docs/baselines/BASELINE.md` | KEEP (becomes overview alongside check2hgi audit details) | See subdirectory table above. |
| `docs/context/` (6 files) | KEEP at `docs/context/` | Task/embedding/architecture background — project-wide. |
| `docs/datasets/DATASETS.md` | KEEP at `docs/datasets/` | Dataset reference — project-wide. |
| `docs/issues/` (7 generic files) | KEEP at `docs/issues/` (with check2hgi's nested under `docs/issues/check2hgi/`) | Generic + study-specific. |
| `docs/plans/{HEAD_ABLATION_PLAN, HYPERPARAM_ABLATION_PLAN}.md` | KEEP at `docs/plans/` | Ablation plans, project-wide. |
| `docs/reports/{README, report_v1_20260415}.md` | KEEP at `docs/reports/` | Status report repository. |
| `docs/thesis/` | KEEP at `docs/thesis/` | Paper thesis options. |

### Resulting `docs/` structure (target end-state)

```
docs/
├── README.md                          ← landing (promoted from check2hgi/README.md, rewritten)
├── AGENT_CONTEXT.md                   ← check2hgi briefing (promoted)
├── NORTH_STAR.md                      ← champion config (promoted)
├── CHANGELOG.md                       ← timeline (promoted)
├── CLAIMS_AND_HYPOTHESES.md           ← claim catalog (promoted)
├── CONCERNS.md                        ← risk audit (promoted)
├── FINAL_SURVEY.md                    ← 5-state matrix (promoted)
├── MTL_ARCHITECTURE_JOURNEY.md        ← supplementary narrative (promoted)
├── PAPER_BASELINES_STRATEGY.md        ← baseline-table mapping (promoted)
├── BRACIS_GUIDE.md                    ← (kept)
├── PAPER_FINDINGS.md                  ← (kept, legacy)
├── check2hgi_overview.tex             ← (kept, paper asset)
├── results/                           ← canonical numerical source (promoted)
├── findings/                          ← per-experiment F-trail (promoted from research/)
├── baselines/                         ← BASELINE.md + next_category/ + next_region/ + README.md (merged)
├── paper/                             ← section drafts (promoted)
├── scope/                             ← (promoted)
├── review/                            ← (promoted)
├── launch_plans/                      ← (promoted)
├── infra/                             ← ⭐ NEW: ops/infra docs (see 2e)
├── archive/
│   ├── HGI_HYPERPARAMETER_TUNING_2026-04-13.md       (kept)
│   ├── HGI_PERFORMANCE_IMPROVEMENT_PLAN.md           (kept)
│   ├── KNOWLEDGE_BASE_2026-04-13.md                  (moved)
│   ├── fusion-study/                                 ← from docs/studies/archive/fusion/ (2c)
│   ├── check2hgi-post-paper-closure-2026-05-01/      ← from check2hgi/archive/
│   ├── check2hgi-pre-b3-framing/                     ← from check2hgi/archive/
│   ├── check2hgi-research-pre-b3/                    ← from check2hgi/archive/
│   ├── check2hgi-research-pre-b5/                    ← from check2hgi/archive/
│   ├── check2hgi-phases-original/                    ← from check2hgi/archive/
│   ├── check2hgi-v1-wip-mixed-scope/                 ← from check2hgi/archive/
│   └── check2hgi-2026-04-20-status-reports/          ← from check2hgi/archive/
├── studies/                           ← clean container, EMPTY (ready for future studies on check2hgi)
├── context/                           ← (kept)
├── datasets/                          ← (kept)
├── issues/                            ← generic + check2hgi/ subdir (merged)
├── plans/                             ← (kept)
├── reports/                           ← (kept)
└── thesis/                            ← (kept)
```

### What's preserved vs. dropped

- **Preserved**: every check2hgi finding, claim, result, baseline audit, paper draft, archive snapshot — all moved with `git mv` (history intact).
- **Dropped**: nothing from check2hgi.

### Risk flags for this section

- `docs/studies/check2hgi/research/` → `docs/findings/` is a **rename**. Internal cross-references in markdown that say `studies/check2hgi/research/...` will break — must run a repo-wide grep + update.
- `docs/issues/` already exists; nesting check2hgi's issues under `docs/issues/check2hgi/` keeps both visible.
- `articles/[BRACIS]_Beyond_Cross_Task/` references `docs/studies/check2hgi/...` paths heavily (per Phase-1 grep). Needs cross-reference sweep.

---

## 2e · Ops/infra documentation consolidation plan

**Proposed location: `docs/infra/`** (chosen over `docs/operations/` for brevity and convention; matches the "infrastructure-as-context" framing — runtime environments, not application logic).

### Target structure

```
docs/infra/
├── README.md                       ← landing: "I'm on machine X, where do I look?" decision tree
├── local/
│   └── README.md                   ← M4 Pro / MPS notes, .venv, dev workflow
├── runpod/
│   ├── README.md                   ← from docs/RUNPOD_GUIDE.md (canonical)
│   └── scripts.md                  ← index of scripts/runpod_*.sh + what each does
├── colab/
│   ├── README.md                   ← from docs/COLAB_GUIDE.md (canonical)
│   ├── notebooks.md                ← index of notebooks/colab_*.ipynb
│   └── study_runner.md             ← scripts/study/colab_runner.py usage (extracted from docstring)
├── lightning/
│   └── README.md                   ← consolidated from PHASE3_LIGHTNING_HANDOFF.md durable parts + scripts/setup_lightning_pod.sh usage
├── h100/
│   └── README.md                   ← consolidated from H100_FLCATX_PERVISIT_PROMPT.md + scripts/run_h100_*.sh + GAP_A_RUNPOD_HANDOFF_PROMPT durable parts (note: gap_a was H100-on-RunPod hybrid)
└── data/
    └── drive_download.md           ← scripts/phase3_download_drive.py + gdown patterns + Drive layout conventions
```

### File-by-file migration

| Source | Destination | Action |
|---|---|---|
| `docs/RUNPOD_GUIDE.md` | `docs/infra/runpod/README.md` | `git mv` + edit title |
| `docs/COLAB_GUIDE.md` | `docs/infra/colab/README.md` | `git mv` + edit title |
| `scripts/H100_FLCATX_PERVISIT_PROMPT.md` | `docs/infra/h100/README.md` | `git mv` + extract durable infra recipes; the experiment-specific prompt content can become a "Historical: H100 FL+CA+TX per-visit run" appendix or be archived. |
| `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/PHASE3_LIGHTNING_HANDOFF.md` | extract durable parts → `docs/infra/lightning/README.md`; original stays in archive | Cherry-pick (manual). The PHASE3-specific tracking content stays in archive. |
| `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/GAP_A_RUNPOD_HANDOFF_PROMPT.md` | cherry-pick durable RunPod facts into `docs/infra/runpod/README.md`; original stays in archive | Manual extraction; the gap_a-specific prompt stays archived. |
| `docs/studies/check2hgi/archive/post_paper_closure_2026-05-01/H100_CAMERA_READY_GAPS_PROMPT.md` | cherry-pick durable H100 facts into `docs/infra/h100/README.md`; original stays in archive | Manual extraction. |
| Index of `scripts/runpod_*.sh` (8 files) | `docs/infra/runpod/scripts.md` (NEW) | One-paragraph summary per script. |
| Index of `scripts/run_h100_*.sh` (4 files) | `docs/infra/h100/README.md` "Scripts" section | One-paragraph summary per script. |
| Index of `scripts/setup_lightning_pod.sh` + `scripts/run_phase{2,3}*lightning.sh` | `docs/infra/lightning/README.md` "Scripts" section | One-paragraph summary per script. |
| Index of `notebooks/colab_*.ipynb` (6 notebooks) | `docs/infra/colab/notebooks.md` (NEW) | One-paragraph summary per notebook. |
| `scripts/study/colab_runner.py` docstring content | `docs/infra/colab/study_runner.md` (NEW) | Extract module docstring + usage examples. |
| `scripts/phase3_download_drive.py` usage | `docs/infra/data/drive_download.md` (NEW) | Extract docstring + the Drive folder convention. |

**Local environment doc (NEW):**
- `docs/infra/local/README.md` — synthesize from CLAUDE.md "DEVICE config" notes + plans/mtlnet_speed_optimization.md + project memory (MPS/PyTorch versions, .venv conventions).

**What stays in `scripts/` (not under `docs/infra/`):**
Per advisor review, the ~30 study-specific experiment launchers (`scripts/run_b3_*.sh`, `run_b5_*.sh`, `run_b9_*.sh`, `run_f21c_*`, `run_f27_*`, `run_f37_*`, `run_f40_*`, `run_f48_*`, etc.) are **experiment recipes, not ops**. They stay in `scripts/`. `docs/infra/` does NOT index them. The line is: a script that *sets up a machine or moves data* is ops (and gets indexed); a script that *runs an experiment configuration* is a recipe (and stays in `scripts/` without an infra index entry).

**Index page (NEW):**
- `docs/infra/README.md` — decision tree:
  - "Are you on local M4 Pro / Apple Silicon?" → `local/`
  - "Are you on RunPod (CUDA, RTX 4090)?" → `runpod/`
  - "Are you on Colab T4 (in a notebook)?" → `colab/`
  - "Are you on Lightning.ai (A100/H100 pods)?" → `lightning/`
  - "Are you on H100 SSH bare-metal?" → `h100/`
  - "Need to fetch data from Google Drive?" → `data/drive_download.md`

### Breadcrumbs at old locations

For documents that **moved** (not just cherry-picked), leave a 1-line stub at the original path so external links / agent memory still works:

- `docs/RUNPOD_GUIDE.md` → 1-line file: `Moved to [docs/infra/runpod/README.md](infra/runpod/README.md).`
- `docs/COLAB_GUIDE.md` → 1-line stub.
- `scripts/H100_FLCATX_PERVISIT_PROMPT.md` → 1-line stub pointing to `docs/infra/h100/`.

For cherry-picked archive docs (PHASE3_LIGHTNING_HANDOFF, GAP_A_RUNPOD_HANDOFF, H100_CAMERA_READY_GAPS): NO breadcrumbs — they stay in archive AND their durable content is in `docs/infra/`. The archive context is historical and self-contained.

For the bare scripts (`scripts/runpod_*.sh` etc.): NO file moves — only documented from `docs/infra/`. The scripts themselves stay in `scripts/` (canonical executable location).

### CLAUDE.md hook

CLAUDE.md gets a single pointer line in the docs/-architecture section:
> `docs/infra/` — operational documentation (RunPod, Colab, Lightning, H100, local). Start at `docs/infra/README.md`.

The current "Running on Google Colab (T4)" section in CLAUDE.md collapses from 12 lines to 2-3 lines pointing at `docs/infra/colab/`.

---

## 2f · Branch cleanup plan

### Local branches

| Branch | Status vs `worktree-check2hgi-mtl` | Action | Reason |
|---|---|---|---|
| `worktree-check2hgi-mtl` | self | KEEP | Active branch about to merge |
| `main` | -1 / +1 (only stale frozen.json) | KEEP | Target |
| `check2hgi-up` | +1 / 0 (= the PR #20 round-2 fix already merged) | **DELETE** (auto) | Dead — content fully merged via PR #20 |
| `feat/colab-gpu-perf` | +243 / 0 | **DELETE** (auto, after verifying head is in worktree's history) | Dead — content fully integrated |
| `bracis` | +92 / +4 | **KEEP** | Active for BRACIS 2026 anonymous review (anonymization commits unique to this branch) |
| `create-improvement-plan-for-mtlnet` | +500 / 0 | **DELETE** (auto) | Old copilot exploration, no unique value |
| `review/mtl-ablation-rebase` | +500 / 0 (same tip as above) | **DELETE** (auto) | Same |

### Remote branches

| Branch | Status | Action | Reason |
|---|---|---|---|
| `origin/worktree-check2hgi-mtl` | mirrors local | KEEP | Source-of-truth for this branch |
| `origin/main` | mirrors local | KEEP | Target |
| `origin/check2hgi-up` | dead (PR merged) | **DELETE** (auto) | |
| `origin/feat/colab-gpu-perf` | dead | **DELETE** (auto) | |
| `origin/bracis` | active (review branch) | **KEEP** | |
| `origin/perf/training-optimizations` | dead — `git log worktree..perf/training-optimizations` empty (already merged into worktree) | **DELETE** (auto) | Verified no unique commits |
| `origin/copilot/create-improvement-plan-for-mtlnet` | dead | **DELETE** (auto) | |
| `origin/h100/pervisit-fl-ca-tx-results` | +65/+2 — has `a858177` and `d714ce0` not in our branch | **VERIFY IN PHASE 3-0 (PRE-MERGE), then DELETE** | The 2 commits added per-fold JSONs and the 5-state §6.1 figure. Per `4b20085 D2 closed — 5-state per-visit figure` on this branch, the *figure and §6.1 update* were integrated. Per advisor B2: integration check happens BEFORE merge, not after. Concrete check: `find docs/studies/check2hgi/results/ -name "*ch19*pervisit*" -o -name "*CH19*"` should return 45 per-fold JSONs (FL/CA/TX × 5 folds × 3 cells). If yes → DELETE. If raw JSONs are missing → `git cherry-pick a858177 d714ce0` first, then DELETE. |
| `origin/worktree-metrics` | merged into main per `git branch -r --merged main` | **DELETE** (auto) | |
| `origin/add-claude-github-actions-1775967051138` | merged | **DELETE** (auto) | |
| `origin/feat/hgi-paper-alignment` | merged | **DELETE** (auto) | |
| `origin/feat/sphere2vec-paper-variant` | merged | **DELETE** (auto) | |
| `origin/perf/mtl-speed-batch1` | merged | **DELETE** (auto) | |
| `origin/copilot/research-prediction-classification-pois` | merged | **DELETE** (auto) | |
| `origin/fix/nashmtl-ecos-solver` | merged | **DELETE** (auto) | |
| `origin/chore/post-refactoring-cleanup` | merged | **DELETE** (auto) | |
| `origin/docs/refactoring-plan` | merged | **DELETE** (auto) | |
| `origin/copilot/deep-analysis-of-merge-request` | merged | **DELETE** (auto) | |
| `origin/copilot/deep-analysis-of-mr` | merged | **DELETE** (auto) | |

### Worktree cleanup

Per `git worktree list`:
- `ingred-perf-opt` (perf/training-optimizations) — marked **prunable**; safe to `git worktree prune`.
- `mtlnet-improve` (copilot/create-improvement-plan-for-mtlnet) — marked **prunable**; safe to prune.
- `worktree-check2hgi-mtl` — KEEP (current).
- `worktree-check2hgi-up` (check2hgi-up) — REMOVE after deleting branch.
- `bracis` — KEEP (branch is active).
- `feat-colab-gpu-perf` — REMOVE after deleting branch.

### Categorized summary

- **Auto-delete (dead, no confirmation needed)**: 17 branches (5 local + 12 remote) + 2 prunable worktrees.
- **Confirm individually (1)**: `origin/h100/pervisit-fl-ca-tx-results` — verify per-fold JSONs are integrated, then DELETE (or cherry-pick first).
- **Keep (active)**: `main`, `worktree-check2hgi-mtl`, `bracis` (local + remote each).

### Execution

Per advisor B3, **before deleting any branch**, tag its tip as `archived/<branch-name>` so the SHA stays reachable by name (insurance, near-zero cost). Example:

```bash
for b in check2hgi-up feat/colab-gpu-perf perf/training-optimizations \
         copilot/create-improvement-plan-for-mtlnet review/mtl-ablation-rebase \
         worktree-metrics add-claude-github-actions-1775967051138 \
         feat/hgi-paper-alignment feat/sphere2vec-paper-variant \
         perf/mtl-speed-batch1 copilot/research-prediction-classification-pois \
         fix/nashmtl-ecos-solver chore/post-refactoring-cleanup \
         docs/refactoring-plan copilot/deep-analysis-of-merge-request \
         copilot/deep-analysis-of-mr h100/pervisit-fl-ca-tx-results; do
  # use origin/$b for remote-only; both for dual-existing branches
  ...
done
```

After Phase 3-8 (branch cleanup), only these branches remain:
- Local: `main`, `worktree-check2hgi-mtl`, `bracis`.
- Remote: `origin/main`, `origin/worktree-check2hgi-mtl`, `origin/bracis`.
- Worktrees: main repo, `bracis`, `worktree-check2hgi-mtl`.
- Tags: `pre-reorg-2026-05-14` (recovery anchor) + `archived/<branch-name>` (one per deleted branch).

---

## 2g · CLAUDE.md update plan

The current CLAUDE.md (240 lines) is already a good map. The reorg requires targeted edits, not a rewrite.

### Edits

1. **Add "Project focus" section near top** (between "Project Overview" and "File Architecture"):
   ```markdown
   ## Project focus
   
   The project's primary study is **check2hgi** — a check-in-level Check2HGI substrate
   for joint POI prediction (paper at BRACIS 2026).
   
   **Two source-of-truth folders:**
   - **Science**: `docs/` root — `README.md`, `AGENT_CONTEXT.md`, `NORTH_STAR.md`,
     `CHANGELOG.md`, `CLAIMS_AND_HYPOTHESES.md`, `CONCERNS.md`, `FINAL_SURVEY.md`,
     `MTL_ARCHITECTURE_JOURNEY.md`, `PAPER_BASELINES_STRATEGY.md`. Canonical numbers:
     `docs/results/RESULTS_TABLE.md §0`.
   - **Paper**: `articles/[BRACIS]_Beyond_Cross_Task/` — the BRACIS 2026 submission
     working folder (`AGENT.md`, `PAPER_DRAFT.md`, `PAPER_STRUCTURE.md`,
     `STATISTICAL_AUDIT.md`, `TABLES_FIGURES.md`, `samplepaper.tex`,
     `references.bib`, `AUDIT_LOG.md`). Read `AGENT.md` first if you are writing prose.
   
   The earlier **fusion** study has been archived under `docs/archive/fusion-study/`
   (concepts, results, claim catalog, leakage diagnoses, lab-notebook records all
   preserved intact).
   
   **Important: fusion remains a first-class engine in the codebase even though the
   study is archived.** Code paths (`src/configs/paths.py` FUSION enum,
   `src/data/inputs/fusion.py`, `experiments/full_fusion_ablation.py`,
   `pipelines/fusion.pipe.py`) are intact and supported. "Archived" means the
   study is closed, NOT that the engine is deprecated.
   
   `docs/studies/` is a clean container for future studies layered on check2hgi
   (e.g., new ablation tracks). It is currently empty.
   ```

2. **Update "File Architecture" → docs/ subtree** to the new layout (showing promoted check2hgi docs at `docs/` root, `docs/findings/`, `docs/studies/{canonical_improvement, merge_design, hgi_category_injection}/`, `docs/infra/`, `docs/archive/fusion-study/`, etc.).

3. **Replace "Running on Google Colab (T4)" section** with:
   ```markdown
   ## Running on remote/cloud machines
   
   All operational documentation (Colab T4, RunPod 4090, Lightning A100/H100, H100 SSH,
   local M4 Pro, Drive download) lives under [`docs/infra/`](docs/infra/). Start at
   [`docs/infra/README.md`](docs/infra/README.md) — it has a decision tree by machine type.
   ```

4. **Update "Branch-scoped study context" section** to reflect the new role:
   ```markdown
   ## Branch-scoped study context

   The primary study (check2hgi) lives at `docs/` root and is loaded automatically
   on every branch.

   Active follow-up studies live under [`docs/studies/`](docs/studies/) — currently
   `canonical_improvement/`, `merge_design/`, `hgi_category_injection/`. Each has its
   own onboarding doc (e.g., `AGENT_PROMPT.md`, `STATE.md`, `INDEX.md`).

   When a branch is dedicated to one of those follow-up studies (or a new one), create
   a `CLAUDE.local.md` at the repo root pointing to the study's onboarding doc. The
   file is gitignored and branch-local. Example:

   ```markdown
   # Branch-active study
   This branch is the **canonical_improvement** study. Read first:
   - `docs/studies/archive/canonical_improvement/AGENT_PROMPT.md`
   - `docs/studies/archive/canonical_improvement/log.md`
   ```
   ```

5. **Add a "What changed (2026-05-14 reorg)" appendix at the end** — short, dated, points to this `docs/MERGE_REORG_PLAN_2026-05-14.md` for the full record:
   ```markdown
   ## What changed (2026-05-14 reorg)
   
   - check2hgi promoted from `docs/studies/check2hgi/` to `docs/` root.
   - fusion study archived to `docs/archive/fusion-study/` (intact, with closure note).
   - Ops/infra docs (RunPod, Colab, Lightning, H100, local, Drive) consolidated under `docs/infra/`.
   - Old `docs/RUNPOD_GUIDE.md`, `docs/COLAB_GUIDE.md`, `scripts/H100_FLCATX_PERVISIT_PROMPT.md` left as 1-line breadcrumbs.
   - Dead branches pruned; `bracis` (BRACIS review) and `worktree-check2hgi-mtl` retained.
   - Full record: `docs/MERGE_REORG_PLAN_2026-05-14.md`.
   ```

### `CLAUDE.local.md` update

Currently points to fusion. Per advisor P2, the right call depends on what active work the user picks up next on this branch:

- If the next active work is the primary check2hgi paper closure / followups → **delete** `CLAUDE.local.md` (primary study is now visible at `docs/` root, no routing needed).
- If the next active work is one of the follow-up studies (e.g., `canonical_improvement`) → **replace** with a 3-line pointer to `docs/studies/<name>/AGENT_PROMPT.md`.

Decision: **default to delete in Phase 3-4**, but mention this in the Phase 3 commit message so the user can replace at execution time if they're picking up a `studies/<name>/` track. (The file is gitignored either way; this is just a courtesy.)

### `notebooks/CLAUDE.md`, `research/embeddings/check2hgi/CLAUDE.md`, etc.

Per-area CLAUDE.md files at deeper paths stay in place — they're context-local and unaffected by the reorg.

---

## 2h · Mandatory advisor review

I will run an advisor review on this drafted plan before presenting to you. Findings get incorporated, then the final plan goes to you for explicit approval.

The advisor's specific lens (per your brief):
1. Is any fusion-study knowledge at risk?
2. Are conflict resolutions sound or hiding regressions?
3. Is the docs reorg coherent and intuitive?
4. Is the ops/infra consolidation complete (nothing scattered remains)?
5. Are any branches flagged for deletion potentially valuable?
6. Is the merge strategy appropriate given the history?
7. Does the CLAUDE.md update capture what an agent really needs?

---

## What this plan deliberately does NOT do

- **Does not move `articles/[BRACIS]_Beyond_Cross_Task/`** — it stays as a sibling of `docs/`, the article working folder is conceptually distinct from project docs.
- **Does not edit `src/`** — no source changes, only docs/branch reorg.
- **Does not rerun any experiments or regenerate results.**
- **Does not touch `data/`, `output/`, `results/` (gitignored runtime artefacts).**
- **Does not delete the `bracis` branch** — kept active for review.
- **Does not force-push main** — uses a clean merge commit instead.
- **Does not strip fusion code support** — fusion remains a first-class engine in the framework.

## Plan-doc destination (per advisor C3)

This plan doc itself (`docs/MERGE_REORG_PLAN_2026-05-14.md`) is currently at `docs/` root. Post-reorg it should NOT sit alongside `AGENT_CONTEXT.md` / `NORTH_STAR.md` (those are study-defining; this is a one-time-event document).

**Action**: as part of Phase 3-4 (CLAUDE.md update commit) or Phase 3-8 (final verification), `git mv docs/MERGE_REORG_PLAN_2026-05-14.md docs/archive/MERGE_REORG_PLAN_2026-05-14.md`. Add a one-line pointer in the new "What changed (2026-05-14 reorg)" appendix in CLAUDE.md.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Markdown cross-references break after path renames | Repo-wide grep + sed pass after each move; verify with link-check before merge |
| `articles/[BRACIS]_Beyond_Cross_Task/` references old `docs/studies/check2hgi/...` paths | Same sweep |
| `h100/pervisit-fl-ca-tx-results` raw JSONs may not be integrated | Inspect `docs/studies/check2hgi/results/` for per-fold JSONs before deleting that branch; cherry-pick if missing |
| Forgotten ops content stays scattered | Repo-wide grep for `runpod\|colab\|lightning\|h100\|drive` after consolidation; list any orphans |
| User changes mind on promotion specifics | Promotion plan (2d) requires explicit approval before any move — gate is in place |
| Tests break after config/path moves | Tests don't reference docs/ paths; should be unaffected. Verify with `pytest` after Phase 3-3 |
| `CLAUDE.local.md` deletion confuses future branch agents | New CLAUDE.md "Branch-scoped study context" wording is explicit about when CLAUDE.local.md is needed |
