# Check2HGI Study — Agent Context

Read this before any scientific work on the `worktree-check2hgi-mtl` branch. It is the study-specific briefing extracted from the repo-wide `CLAUDE.md` — the check2HGI track's counterpart to `docs/studies/fusion/AGENT_CONTEXT.md`.

---

## Current status

This study runs **alongside** the fusion study — they coexist under `docs/studies/`. Fusion investigates POI-category classification on fused POI-level embeddings; this study investigates **next-POI + next-region prediction on check-in-level contextual embeddings** (Check2HGI). Do not cross-reference or mix artefacts between the two studies.

Phase P0 is active: embeddings have been generated for Alabama + Florida (+ Arizona as triangulation); next_region labels are derived; integrity checks are outstanding.

---

## Study navigation

| File | Purpose |
|------|---------|
| `docs/studies/check2hgi/README.md` | Entry point + scope statement |
| `docs/studies/check2hgi/QUICK_REFERENCE.md` | One-page overview |
| `docs/studies/check2hgi/MASTER_PLAN.md` | 5-phase strategy (P0 prep → P1 single-task baselines → P2 MTL headline → P3 dual-stream → P4 cross-attention → P5 ablations) |
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

**Slot A (`task_a`):** `next_poi` — predict the exact next POI id from a 9-window of check-in embeddings. Cardinality: ~10K (AL) to ~80K (FL) classes. Primary metric: **Acc@{1, 5, 10}, MRR, NDCG@{5, 10}**. Macro-F1 reported for completeness only.

**Slot B (`task_b`):** `next_region` — predict the region of the next POI. Cardinality: ~1.1K (AL), ~4.7K (FL), ~1.5K (AZ) classes. Primary metric: **Acc@{1, 5, 10}, MRR**. Macro-F1 reported for completeness only.

Both heads are sequential (`next_mtl` transformer) and consume the same X tensor `[B, 9, 64]` of check-in embeddings. Labels differ. Joint monitor:

```
val_joint_geom_lift = sqrt(
    max(acc1_poi    / majority_poi,    1e-8) *
    max(acc1_region / majority_region, 1e-8)
)
```

**Geometric** mean of per-head lifts-over-majority, NOT arithmetic mean. The arithmetic version (used in v1 and since fixed on 2026-04-15 per the review-agent finding) is scale-incoherent when head cardinalities span orders of magnitude (FL next_poi majority ~0.001% vs FL next_region majority 22.5% — arithmetic mean is dominated by the POI term). The geometric mean forces both heads to contribute multiplicatively so the monitor penalises either head collapsing.

**Reported alongside:** `val_joint_arith_lift` (the old formula) + per-head raw Acc@K and OOD-restricted Acc@K. The paper uses `val_joint_geom_lift` as the checkpoint monitor but reports the full metric suite.

---

## Baselines

**Next-POI literature (ranking metrics).** All report Acc@K / MRR / NDCG on Foursquare-NYC/TKY or Gowalla-global.

| Paper | Venue | Relevance |
|---|---|---|
| HMT-GRN (Lim et al.) | SIGIR '22 | Closest: hierarchical GRU on region-seq + POI-seq, MTL |
| MGCL (Zhu et al.) | Frontiers '24 | Multi-granularity contrastive + region + category auxiliary |
| Bi-Level GSL | arXiv '24 | Explicit region-POI bi-level graph |
| LSTPM (Sun et al.) | AAAI '20 | Long/short-term user prefs |
| STAN (Luo et al.) | WWW '21 | Spatio-temporal attention |
| GETNext (Yang et al.) | SIGIR '22 | Trajectory flow + transformer |
| Graph-Flashback | KDD '22 | Sequential + graph |
| ImNext (He et al.) | KBS '24 | Irregular interval MTL |

**What this study does NOT benchmark against:** HAVANA / PGC / POI-RGNN. Those are POI-category classification baselines on Gowalla state-level; different task, different metric family, covered by the fusion study.

**Direct-numeric-comparison honesty:** our state-level Gowalla runs are not directly comparable to the NYC/TKY/Gowalla-global numbers those papers report. Declared limitation (CH12).

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
# Smoke the end-to-end path on Alabama (uses task_set=check2hgi_next_poi_region)
PYTHONPATH=src DATA_ROOT=/path/to/data OUTPUT_DIR=/path/to/output \
  python scripts/train.py \
    --state alabama --engine check2hgi --task mtl \
    --task-set check2hgi_next_poi_region \
    --folds 1 --epochs 2 --gradient-accumulation-steps 1 \
    --no-folds-cache
```

The `check2hgi_next_poi_region` preset resolves `task_b.num_classes` from the actual next_region label tensor at runtime (see `src/tasks/presets.py::resolve_task_set`).

---

## Issue tracker (opened as of 2026-04-15)

None yet — this study is freshly split from the fusion track. Issues raised during P0 integrity checks will land here.

---

## What lives in the sibling fusion study (don't touch from this branch)

- `docs/studies/fusion/` — any artefact, any state.
- The POI-category classification task + HAVANA / PGC baselines.
- The 5×20 arch×optim grid (fusion's P1).
- The fusion engine + Sphere2Vec + Time2Vec wrangling.
