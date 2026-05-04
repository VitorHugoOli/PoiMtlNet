# Check2HGI Study — Agent Context

Read this before any scientific work on the `worktree-check2hgi-mtl` branch. It is the study-specific briefing extracted from the repo-wide `CLAUDE.md` — the check2HGI track's counterpart to `docs/studies/fusion/AGENT_CONTEXT.md`.

---

## Current status

This study runs **alongside** the fusion study — they coexist under `docs/studies/`. Fusion investigates POI-category classification on fused POI-level embeddings; this study investigates **joint next_category + next_region prediction on check-in-level contextual embeddings** (Check2HGI). Do not cross-reference or mix artefacts between the two studies.

P0 complete, P1 in progress (2026-04-16). Embeddings + next_region labels exist for AL / FL / AZ; Markov floor corrected to region granularity; single-task baselines established; MTL variant code (CGC/MMoE/DSelectK/PLE) ported to the TaskSet preset system.

## Thesis (bidirectional, 2026-04-16)

MTL `{next_category, next_region}` must improve **both** heads over their respective single-task baselines on AL and FL. A one-sided lift fails the thesis. The architectural hypothesis for P3/P4 is **per-task input modality** (check-in emb → `category_encoder`, region emb → `next_encoder`) rather than shared or concatenated inputs. See CH01/CH02/CH03 in the claim catalog.

---

## Study navigation

| File | Purpose |
|------|---------|
| `docs/studies/check2hgi/README.md` | Entry point + scope statement |
| `docs/studies/check2hgi/QUICK_REFERENCE.md` | One-page overview |
| `docs/studies/check2hgi/MASTER_PLAN.md` | 6-phase strategy (P0 prep → P1 single-task + input-modality ablation → P2 MTL arch×optim grid → P3 MTL headline bidirectional → P4 per-task input modality → P5 cross-attention (gated) → P6 encoder enrichment) |
| `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md` | Authoritative claim catalog (CH01..CHnn) |
| `docs/studies/check2hgi/COORDINATOR.md` | Orchestrator agent spec |
| `docs/studies/check2hgi/phases/` | Per-phase execution plans |
| `docs/studies/check2hgi/state.json` | Runtime state (current phase + test statuses) |
| `docs/studies/check2hgi/KNOWLEDGE_SNAPSHOT.md` | Current baseline-knowledge snapshot |
| `docs/studies/check2hgi/HANDOFF.md` | Session handoff notes |

Before any scientific work, read `QUICK_REFERENCE.md` and `KNOWLEDGE_SNAPSHOT.md`.

---

## Skills

Study-agnostic commands from the `94c4dda` refactor. Always set `STUDY_DIR` explicitly when operating on this branch:

```bash
STUDY_DIR=docs/studies/check2hgi /coordinator P0   # analyze P0, check gate, recommend next
STUDY_DIR=docs/studies/check2hgi /worker P1        # run the next planned test in P1
STUDY_DIR=docs/studies/check2hgi /study status     # show phase + test statuses
```

If `STUDY_DIR` is unset, commands default to `docs/studies/fusion/` — do NOT run them on this branch without the override.

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

Our single-task Check2HGI on AL: **38.67% F1** (already above POI-RGNN's 31.8–34.5% range).

**For `next_region` (ranking, Acc@K/MRR) — concept-aligned, different datasets:**

| Paper | Venue | Relevance |
|---|---|---|
| HMT-GRN (Lim et al.) | SIGIR '22 | Hierarchical GRU on region-seq + POI-seq, MTL — closest architectural match |
| MGCL (Zhu et al.) | Frontiers '24 | Multi-granularity contrastive with region + category auxiliary heads |
| Bi-Level GSL | arXiv '24 | Explicit region-POI bi-level graph |

Our single-task Check2HGI on AL (region-emb input, `next_gru` default, **5f × 50ep**): **56.94% ± 4.01 Acc@10**.

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

For P4 (per-task input modality), new flags `--task-a-input-type` / `--task-b-input-type` with values `{checkin, region, concat}` are planned but not yet wired (require FoldCreator extension, ~80 LOC).

---

## Issue tracker (opened as of 2026-04-15)

None yet — this study is freshly split from the fusion track. Issues raised during P0 integrity checks will land here.

---

## What lives in the sibling fusion study (don't touch from this branch)

- `docs/studies/fusion/` — any artefact, any state.
- The POI-category classification task + HAVANA / PGC baselines.
- The 5×20 arch×optim grid (fusion's P1).
- The fusion engine + Sphere2Vec + Time2Vec wrangling.
