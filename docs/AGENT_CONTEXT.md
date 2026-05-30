# Check2HGI Study — Agent Context

Read this before any scientific work on the `worktree-check2hgi-mtl` branch. It is the study-specific briefing extracted from the repo-wide `CLAUDE.md` — the check2HGI track's counterpart to `docs/studies/fusion/AGENT_CONTEXT.md`.

---

## Current status

This study runs **alongside** the fusion study — they coexist under `docs/studies/`. Fusion investigates POI-category classification on fused POI-level embeddings; this study investigates **joint next_category + next_region prediction on check-in-level contextual embeddings** (Check2HGI). Do not cross-reference or mix artefacts between the two studies.

**Post-paper-closure + cat-Δ Wilcoxon + TX recipe multi-seed + CA/TX arch-Δ n=20 era (2026-05-02 v11).** Paper closure complete; the headline narrative is **substrate task-asymmetry first, classic MTL tradeoff second**. The article-side BRACIS submission lives in `articles/[BRACIS]_Beyond_Cross_Task/`. **Single canonical numerical source: `results/RESULTS_TABLE.md §0` (v11).** All other numbers in this folder reference it; numbers that contradict v11 are either stale (mark, fix, or archive) or audit-historical.

> 🔬 **2026-05-28 — Post-v11 study progress (read this before citing v11).**
> Three closures + two active studies have layered on top of v11:
> - **`canonical_improvement` CLOSED 2026-05-19** — substrate axis exhausted ±0.8 pp (Tier 1-6, 26 mechanism families).
> - **`mtl-protocol-fix` CLOSED 2026-05-24 v6 final** — F1 selector fix (`joint_geom_simple`) recovers +5.6 pp at FL multi-seed deployable axis. C21 RESOLVED. P4 frozen-cat identifies residual MTL-vs-STL reg gap as ARCHITECTURAL.
> - **Phase 3 of `mtl-protocol-fix`** — §4.5 log_T-KD PROMOTED (+2.4/+5.1/+2.3 pp AL/AZ/FL Wilcoxon-strict); §4.6 class-balanced sampler FALSIFIED; §4.2 composite STL c2hgi-cat + STL HGI-reg ESTABLISHED as **current project headline on the reg axis (+7-12 pp vs MTL@disjoint at every state)**.
> - **`mtl_improvement` ACTIVE since 2026-05-16** (branch `mtl-improve`) — T0-T8 chain: backbones, loss, batch, LR, α, heads, multi-seed champion.
> - **`substrate-protocol-cleanup` CLOSED 2026-05-29** (main) — Tier A-D substrate + protocol cleanup. log_T-KD PROMOTED (now **v12 default**); substrate axis NULL in MTL at AL/AZ/FL (regime-limited, even HGI ≈ canonical in MTL); ResLN encoder = STL-best (v12 default, no MTL benefit). Default flip v11→v12: see [`results/CANONICAL_VERSIONS.md`](results/CANONICAL_VERSIONS.md). BRACIS paper cites v11 (reproduce: `--log-t-kd-weight 0.0 --encoder gcn`).
>
> When both active studies land champions, the §0 pipeline will be **re-run from scratch** and this briefing refreshed. Until then, v11 below remains the paper-citable canon with the F1 / composite / log_T-KD caveats above. Cross-study outcomes log at [`studies/log.md`](studies/log.md).

**Before doing scientific work, read in this order:**
1. `CHANGELOG.md` — chronological timeline of findings + lessons (the single source for "what was found when, why").
2. `results/RESULTS_TABLE.md §0` — canonical paper numbers (v11, 2026-05-02).
3. `articles/[BRACIS]_Beyond_Cross_Task/AGENT.md` — article-side operational rules + voice + statistics + page budget.
4. `CLAIMS_AND_HYPOTHESES.md` (with whitelist banner) — paper-facing safe entries are CH16 / CH18-cat / CH15 reframing / CH19 / CH22; CH23 (Tier-6 substrate-protocol mismatch) is §Discussion-only; everything else needs cross-checking against v11.

> ⚠ **Data-pipeline preflight (mtl-protocol-fix study 2026-05-20 — MANDATORY for ALL agents):** Before ANY MTL/STL training run that uses `--per-fold-transition-dir`:
>
> 1. **Check stale log_T**: `region_transition_log_seed{S}_fold{N}.pt` mtime MUST be ≥ `output/check2hgi/{state}/input/next_region.parquet` mtime. If older, **rebuild**: `python scripts/compute_region_transition.py --state {state} --per-fold --seed {S}`. `regen_emb_t3.py` does NOT auto-rebuild this in older versions (Tier 6 FL-MTL sweeps silently consumed a May-6 stale log_T inflating reg-Acc@10 by +8 to +12 pp). See [`CONCERNS.md` C22](CONCERNS.md#c22).
> 2. **Paper-grade reporting uses seeds {0, 1, 7, 100}, NOT seed=42.** Seed=42 is the development seed (recipe was tuned on it). At small states (AL/AZ) and FL (post-stale-log_T fix), seed=42 matches multi-seed. At large states (CA/TX), seed=42 overshoots §0.1 v11 by +3 to +7 pp. See [`CONCERNS.md` C23](CONCERNS.md#c23).
> 3. **scripts/train.py preflight raises on stale log_T** as of mtl-protocol-fix branch — heed the error message.
>
> **⚠ MTL protocol blocker (2026-05-19, also MANDATORY):** The production B9 joint selector (`mtl_cv.py:679`, `joint_score = 0.5 * (cat_macro_f1 + reg_macro_f1)`) is **structurally broken on the canonical shipping recipe itself** — not just on substrate variants. `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise and is blind to `reg_top10_acc_indist`'s peak-and-collapse trajectory.
>
> **Concrete fingerprint (canonical shipping FL ep=50 single-seed=42 n=5, NO substrate changes):**
>
> | Selector | cat F1 | reg top10 |
> |---|---:|---:|
> | Per-task disjoint best | 70.49 ± 0.86 | **76.12 ± 0.33** ← capacity |
> | `joint_canonical_b9` (production) | 69.99 ± 1.13 | **65.38 ± 9.10** ← shipped |
>
> Production selector throws away **~10.7 pp of reg-top10 capacity** that the canonical Check2HGI substrate already produces. §0.1's published reg top10 = 63.27 ± 0.10 is consistent with the matched-protocol single-seed `joint_canonical_b9` value (65.38 ± 9.10) within single-seed variance.
>
> **Implications:**
> - The canonical_improvement Tier-6 T6.4 substrate hypothesis (POI-level supervision lifts reg) is **falsified at matched protocol** — T6.4 variants add Δ_reg = +0.08-0.17 pp over shipping under per-task disjoint, well within σ. The "+13 pp" originally claimed was a cross-selector artefact, not a substrate effect. See `CLAIMS_AND_HYPOTHESES.md` CH23-A.
> - The selector bug affects **every Check2HGI MTL run in the published canon**. Future substrate comparisons under the production selector are comparing checkpoints where reg has destabilised; substrate-axis orderings may flip once the selector is fixed.
> - See `CONCERNS.md` C21 (full diagnosis) and the **CLOSED follow-on study at [`docs/studies/mtl-protocol-fix/`](studies/mtl-protocol-fix/) (closed 2026-05-20 v6 final)** which validated the F1 fix at multi-seed (+5.6 pp at FL deployable selector axis), discovered the C22 stale log_T bug, and characterised the residual MTL-vs-STL gap as **architectural** (P4 frozen-cat test falsified the cat-task-interference hypothesis). Next-tier highest-EV is [`future_works/mtl_architecture_revisit.md`](future_works/mtl_architecture_revisit.md). Full closure verdict at [`docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md).

**Headline (the classic MTL tradeoff, sign-consistent across 5 states):**
- Cat: MTL ≥ STL at four of five states (AZ +1.20 p < 1e-4 / FL +1.40 p = 2e-06 / CA +1.68 p = 2e-06 / TX +1.89 p = 2e-06); AL is small-significantly negative (Δ = −0.78 pp, p = 0.036, n = 20 multi-seed; magnitude ~1.9% relative). **All five states (AL/AZ/CA/TX/FL) are now pooled multi-seed (n = 20) on the headline §0.1 axis — the last remaining single-seed asymmetry (FL) closed in v11.**
- Reg: MTL < STL at every state by 7–17 pp (sign-consistent).
- Substrate (matched-head STL): cat Δ = +14.5 to +29 pp at every state (paired Wilcoxon p = 0.0312 each); reg HGI nominally ahead by 1.6–3.1 pp (TOST tied at CA/TX).
- Mechanism (CH19, AL-only): per-visit context = ~72% of cat substrate gap.
- Methodological side-finding (CH3 / Layer 2): cross-attn `task_weight = 0` co-adapts via K/V — encoder-frozen isolation is required.

**The leak-free reframe.** Two of the most prominent earlier findings (F49 "AL +6.48 pp MTL > STL on reg architecture-dominant"; CH18-reg "MTL substrate-specific"`) were leak artefacts of pre-F50 measurements (full-data `region_transition_log.pt` leaks ~13–27 pp; substrate-asymmetric, hurting C2HGI more than HGI). Under leak-free measurement, MTL trails STL on reg at every state. **The leak-free narrative is the paper-facing one**; the leak-era narrative is preserved in `archive/post_paper_closure_2026-05-01/` for audit. See `CHANGELOG.md` 2026-04-29 to 2026-05-01 entries for the timeline of how the reframe happened.

## Thesis (post-leak-free, 2026-05-01)

The bidirectional thesis ("MTL must lift both heads over STL") is **not** what we land. The honest finding is:

1. **Substrate-task-asymmetry (C1).** Check-in-level Check2HGI lifts cat by +14.5 to +29 pp at every state; on reg it ties or marginally trails per-place HGI. Mechanism: per-visit variance is what cat needs; per-POI pooling smooths it away for reg.
2. **Classic MTL tradeoff (C2).** With Check2HGI fixed, joint MTL adds a small cat lift at four of five states and pays a sign-consistent reg cost. Drop-in fixes (FAMO, Aligned-MTL, HSM) do not recover the reg gap.
3. **Methodological note (C3).** Cross-attn `task_weight = 0` ablations are unsound; encoder-frozen isolation is required.

Why "MTL lifts both heads" doesn't survive: the bidirectional thesis was a pre-leak-free framing. Under leak-free measurement, the paper-honest claim is that the substrate carries cat (paper-grade significant at every state) while the architecture pays reg (sign-consistent at every state) — the textbook tradeoff. This is documented and accepted.

---

## Study navigation (post-cleanup 2026-05-01)

**Active (read these first):**

| File | Purpose |
|------|---------|
| `docs/README.md` | Entry point + canonical-source-aware navigation |
| `docs/CHANGELOG.md` ⭐ | Chronological timeline of findings + lessons learned (single source for "what was found when") |
| `docs/results/RESULTS_TABLE.md §0` ⭐ | **Canonical numerical source** for all paper tables (v11, 2026-05-02) |
| `articles/[BRACIS]_Beyond_Cross_Task/` ⭐ | Article-side BRACIS submission folder (AGENT.md / PAPER_DRAFT.md / PAPER_STRUCTURE.md / STATISTICAL_AUDIT.md / TABLES_FIGURES.md / samplepaper.tex / references.bib / AUDIT_LOG.md) |
| `docs/CLAIMS_AND_HYPOTHESES.md` | Claim catalogue with **paper-facing whitelist banner** (CH16 / CH18-cat / CH15 reframing / CH19 / CH22 are safe; others need cross-checking against v11) |
| `docs/NORTH_STAR.md` | Committed champion config (B9 + H3-alt scale-conditional; v11-aligned) |
| `docs/FINAL_SURVEY.md` | Substrate-axis 5-state matrix (cat + reg panels) |
| `docs/CONCERNS.md` | Acknowledged-risks audit log |
| `docs/MTL_ARCHITECTURE_JOURNEY.md` | Supplementary material narrative (F-trail through B3 → F21c → F45 → F48-H3-alt → F49 → paper closure). **Do not narrate the F-trail in main paper text.** |
| `docs/PAPER_BASELINES_STRATEGY.md` | Which baselines appear in which paper table; what is deliberately scoped out |
| `docs/findings/GAP_FILL_WILCOXON.json` | v9 Wilcoxon JSON (cat-Δ AL/AZ/FL + CA/TX recipe n=20 multi-seed) |
| `docs/findings/ARCH_DELTA_WILCOXON.json` | v10 Wilcoxon JSON (CA+TX §0.1 arch-Δ n=20; all four axes p=2e-06) |
| `docs/findings/FL_CAT_DELTA_WILCOXON.json` | v11 Wilcoxon JSON (FL §0.1 arch-Δ n=20; cat p=2e-06, reg p=1.9e-06) |
| `docs/findings/PAPER_CLOSURE_WILCOXON.json` | Paper-closure Wilcoxon artefact |
| `docs/findings/PAPER_CLOSURE_RECIPE_WILCOXON.json` | Recipe-selection Wilcoxon artefact |
| `docs/findings/F49_LAMBDA0_DECOMPOSITION_GAP.md` | Cross-attn `task_weight=0` methodology contribution (the survival of F49) |
| `docs/findings/F50_DELTA_M_FINDINGS_LEAKFREE.md` | Δm leak-free reframe (CH22) |
| `docs/findings/F50_T1_RESULTS_SYNTHESIS.md` | Drop-in MTL ablation (FAMO, Aligned-MTL, HSM) |
| `docs/findings/F51_MULTI_SEED_FINDINGS.md` | Multi-seed B9 vs H3-alt validation |
| `docs/findings/SUBSTRATE_COMPARISON_FINDINGS.md` | Phase-1 substrate-comparison verdict (cat side survives leak-free; reg side does not) |
| `docs/baselines/` | Faithful baseline ports (POI-RGNN, MHA+PE, STAN, ReHDM) + audits |
| `docs/future_works/` | Forward-looking memos (deferred work) |
| `docs/archive/check2hgi-paper-drafts-pre-bracis/` | Pre-BRACIS paper section drafts (methods.md, results.md, limitations.md, appendix_methodology.md) — work landed in `articles/[BRACIS]_*/src/sections/` |
| `docs/archive/check2hgi-reviews-2026-04/` | Dated critical reviews (2026-04-22 → 2026-04-28) |

**Archived (historical reference only):**

| Location | What's there | See also |
|---|---|---|
| `archive/post_paper_closure_2026-05-01/` | Stale paper-prep docs moved during the 2026-05-01 cleanup: study-side `PAPER_DRAFT.md`, `PAPER_STRUCTURE.md` (now under `articles/`), `PAPER_CLOSURE_RESULTS_2026-05-01.md` (background provenance), `OBJECTIVES_STATUS_TABLE.md`, `PAPER_PREP_TRACKER.md`, `PAPER_CLOSURE_PHASES.md`, `FOLLOWUPS_TRACKER.md`, `HANDOVER.md`, all `GAP_A_*` / `H100_CAMERA_READY_GAPS_PROMPT` / `F50_NORTH_STAR_DEEP_EXPLORATION_PROMPT` / `PHASE2_*` / `PHASE3_*` / `SESSION_HANDOFF_*` files. | `archive/post_paper_closure_2026-05-01/README.md` for what's there + why |
| `archive/pre_b3_framing/` | Pre-B3 framing docs (MASTER_PLAN, etc.). | NORTH_STAR.md |
| `archive/phases_original/` | Original P0..P7 phase plans. | CHANGELOG.md |
| `archive/research_pre_b3/`, `archive/research_pre_b5/` | Pre-B3 / pre-B5 research notes. | `research/` (post-B5) |
| `archive/2026-04-20_status_reports/` | Early status reports. | CHANGELOG.md |
| `archive/v1_wip_mixed_scope/` | Pre-scope-split WIP. | — |

Before any scientific work, read `CHANGELOG.md`, `results/RESULTS_TABLE.md §0` (v11), and `articles/[BRACIS]_Beyond_Cross_Task/AGENT.md`.

---

## Skills

Study-agnostic commands from the `94c4dda` refactor. The check2hgi study lives at `docs/` root post-2026-05-14 reorg, so commands targeting it use `STUDY_DIR=docs`:

```bash
STUDY_DIR=docs /coordinator P0   # analyze P0, check gate, recommend next
STUDY_DIR=docs /worker P1        # run the next planned test in P1
STUDY_DIR=docs /study status     # show phase + test statuses
```

To target an active follow-up study under `docs/studies/<name>/` (e.g., canonical_improvement), use `STUDY_DIR=docs/studies/<name>`.

If `STUDY_DIR` is unset, commands default to `docs/archive/fusion-study/` — do NOT run them without the override (the fusion study is archived and read-only).

---

## Task pair

**Slot A (`task_a`):** `next_category` — predict the next check-in's category from a 9-window. Cardinality: 7 classes. Primary metric: **macro-F1** (tail-class sensitive). Top-K is reported but not primary because K=5 over 7 classes is near-trivial.

**Slot B (`task_b`):** `next_region` — predict the region of the next POI. Cardinality: 1,109 (AL), 4,703 (FL), 1,547 (AZ) classes. Primary metric: **Acc@{1, 5, 10}, MRR**. Macro-F1 reported for completeness (tiny, given tail classes have ≤10 samples).

Both heads are **sequential**; under the P3 champion they will consume **different input modalities** (see CH03 / per-task input modality) routed through their task-specific encoders, not a shared X tensor. Joint monitor:

```
val_joint_geom_lift = sqrt(
    max(f1_category      / majority_category,   1e-8) *
    max(acc1_region      / majority_region,     1e-8)
)
```

**Geometric** mean of per-head lifts-over-majority, NOT arithmetic. The arithmetic version is scale-incoherent when head cardinalities span orders of magnitude (F1 on 7 classes vs Acc@1 on 1,109 classes — arithmetic mean is dominated by F1). Geometric mean forces both heads to contribute multiplicatively so the monitor penalises either collapsing.

**Reported alongside:** `val_joint_arith_lift` (old formula, for continuity) + per-head raw metrics (F1 for category; Acc@K, MRR for region) + OOD-restricted Acc@K on the region head.

---

## Baselines

**For `next_category` (classification, macro-F1):**

| Paper | Venue | Relevance |
|---|---|---|
| POI-RGNN (Capanema et al.) | IJCNN '19 | Next-category on Gowalla FL/CA/TX: 31.8–34.5% macro-F1 |
| MHA+PE (Zeng et al.) | 2019 | Next-category on Gowalla global: 26.9% F1 |

Our STL Check2HGI on AL (matched-head `next_mtl`, 5f × 50ep fair folds): **38.58% ± 1.23 F1** (already above POI-RGNN's 31.8–34.5% range). MTL-B3 post-F27 lifts to **42.71% ± 0.0137 F1** on AL (F31), **45.81% on AZ** (Wilcoxon p=0.0312 over STL).

**For `next_region` (ranking, Acc@K/MRR) — concept-aligned, different datasets:**

| Paper | Venue | Relevance |
|---|---|---|
| HMT-GRN (Lim et al.) | SIGIR '22 | Hierarchical GRU on region-seq + POI-seq, MTL — closest architectural match |
| MGCL (Zhu et al.) | Frontiers '24 | Multi-granularity contrastive with region + category auxiliary heads |
| Bi-Level GSL | arXiv '24 | Explicit region-POI bi-level graph |

Our STL Check2HGI on AL (region-emb input, `next_gru`, 5f × 50ep): **56.94% ± 4.01 Acc@10**. STL STAN (ceiling): **59.20% ± 3.62 Acc@10**. **STL `next_getnext_hard` matched-head (F21c)**: **68.37% ± 2.66 Acc@10** — the new reg ceiling at AL/AZ scale (CH18, 2026-04-24). MTL-B3 post-F27 on AL: **59.60 ± 4.09 Acc@10** (first MTL to cross STL STAN on AL; trails matched-head STL by −8.77 pp).

**Simple-baseline floor (updated 2026-04-16):**
- AL next_category: majority 34.2%, Markov 31.7%
- FL next_category: majority 24.7%, Markov 37.2%
- AL next_region: **Markov-1-region 47.01%** Acc@10 (the old POI-level `markov_1step` reporting 21.3% was degenerate — ~50% of val rows fell back to top-k-popular)
- FL next_region: **Markov-1-region 65.05%** Acc@10

**Direct-numeric-comparison honesty:** our state-level Gowalla runs are not directly comparable to the NYC/TKY/Gowalla-global numbers HMT-GRN / MGCL / GETNext report. Declared limitation (CH10).

---

## Key environmental facts

- **Python env:** `/Volumes/Vitor's SSD/ingred/.venv/bin/python` (Python 3.12 + venv).
- **Data root:** set `DATA_ROOT=/Volumes/Vitor's SSD/ingred/data` when invoking scripts from this worktree (the worktree's own `data/` is empty; the real data lives in the main repo root).
- **Output dir:** set `OUTPUT_DIR=/Volumes/Vitor's SSD/ingred/output` so the embedding + input parquets are visible from both main and this branch.
- **MPS runs:** before long training, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` and `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- **scikit-learn 1.8.0 pinned** (same as fusion; the `StratifiedGroupKFold(shuffle=True)` bugfix).

---

## CLI entry point

```bash
# Smoke the end-to-end path on Alabama (uses task_set=CHECK2HGI_NEXT_REGION)
PYTHONPATH=src DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/output \
  python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_region \
    --folds 1 --epochs 2 --gradient-accumulation-steps 1 \
    --no-folds-cache

# P1 region-head ablation (single-task next_region)
PYTHONPATH=src DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/output \
  python scripts/p1_region_head_ablation.py \
    --state alabama --heads next_gru next_temporal_cnn \
    --folds 1 --epochs 30 --input-type region --tag E_region_only
```

The `CHECK2HGI_NEXT_REGION` preset resolves `task_b.num_classes` (n_regions) from the actual next_region label tensor at runtime (see `src/tasks/presets.py::resolve_task_set`). `task_a` (`next_category`) has a fixed num_classes=7.

Per-task input modality flags `--task-a-input-type` / `--task-b-input-type` (values `{checkin, region, concat}`) are wired in `scripts/train.py` and `src/data/folds.py`. **NORTH_STAR B9 requires `--task-a-input-type checkin --task-b-input-type region`.** The trainer default for both is `checkin` — running task_b on `checkin` instead of `region` is a silent failure mode (cat F1 stays correct, reg Acc@10 drops ~22 pp at AL). See `docs/NORTH_STAR.md §Champion` for the full canonical CLI.

---

## Issue tracker (opened as of 2026-04-15)

None yet — this study is freshly split from the fusion track. Issues raised during P0 integrity checks will land here.

---

## What lives in the sibling fusion study (don't touch from this branch)

- `docs/studies/fusion/` — any artefact, any state.
- The POI-category classification task + HAVANA / PGC baselines.
- The 5×20 arch×optim grid (fusion's P1).
- The fusion engine + Sphere2Vec + Time2Vec wrangling.
