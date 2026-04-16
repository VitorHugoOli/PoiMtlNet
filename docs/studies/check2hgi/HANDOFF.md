# Check2HGI Study — Session Handoff (2026-04-15)

> **Sibling to `docs/studies/fusion/HANDOFF.md`.** This document captures the check2HGI study's own running state, separate from the fusion track.

---

## Status at a glance

| Phase | Status | Notes |
|---|---|---|
| **Study setup** | ✅ done | Rewrote docs/studies/check2hgi/ to mirror fusion structure; archived v1 scope-mixed WIP |
| **P0: preparation** | 🟡 partial | Embeddings + next_region labels ready (AL, FL, AZ); integrity checks + fclass-shortcut audit (CH14) outstanding |
| **P1: single-task baselines** | ❌ blocked | Needs next_poi loader + preset code additions (~2–3h) |
| **P2: MTL headline** | ❌ blocked | Needs P1 |
| **P3: dual-stream** | ❌ blocked | Needs P2 |
| **P4: cross-attention** | ❌ blocked (gated) | Needs P3 with ≥ 2pp Acc@10 lift on FL |
| **P5: ablations** | ❌ blocked | Needs P2 |

---

## Commits on this branch (since diverged from main)

After rebase onto `origin/main` (commit `06066998…`), the branch holds 17 commits:

```
<TBD>   feat(study): rewrite docs/studies/check2hgi with correct scope
20c22e9 feat(check2hgi): address CRITICAL_REVIEW agent findings
62d8975 feat(check2hgi): fix [BLOCK] items from CRITICAL_REVIEW.md §1
2194a07 docs(studies): port legacy P1/P2 ablations to check2HGI + AZ probe
b1537c2 docs(critical-review): probe results + Option C spec
2e8c1f5 docs(critical-review): self-audit of code + strategy + SOTA gap
882505a docs(handoff): surface three BRACIS-critical concerns
c058ec0 docs(studies): add HANDOFF snapshot
fa5dbaa feat(training): per-task num_classes + CPU fallback for high-cardinality metrics
f729200 feat(data): next-region label pipeline for check2HGI
44fae97 feat(mtl_cv): parameterise task names with TaskSet
5bfa6a8 feat(mtlnet): parameterise MTLnet with TaskSet
f9f1998 docs(studies): scaffold docs/studies/check2hgi/ with claims + ablations
79456f0 feat(tasks): add TaskConfig + TaskSet presets
0c92f33 plan(check2hgi): finalize branch plan + enable FL in embedding pipeline
c45b01d docs(plans): add check2HGI MTL research overview and decisions
8c52d6c docs(plans): add check2HGI MTL branch plan (next-POI + next-region)
```

---

## Where the data lives

- **Embeddings (all 3 states):** `/Volumes/Vitor's SSD/ingred/output/check2hgi/{alabama,florida,arizona}/embeddings.parquet` (check-in-level, 64-dim) + `poi_embeddings.parquet` + `region_embeddings.parquet`.
- **Next-POI sequences:** `output/check2hgi/{state}/input/next.parquet` (576 emb cols + `next_category` (will be dropped) + `userid`).
- **Next-region labels:** `output/check2hgi/{state}/input/next_region.parquet` (X + `region_idx` + `userid`).
- **Next-POI labels (to generate):** `output/check2hgi/{state}/input/next_poi.parquet` — same X + `poi_idx` column derived via `placeid_to_idx[target_poi]`.
- **Preprocessing artefact:** `output/check2hgi/{state}/temp/checkin_graph.pt` (pickle: `placeid_to_idx`, `poi_to_region`, `num_pois`, `num_regions`).

---

## Key dataset numbers

| State | Check-ins | POIs | Regions | Sequence rows | `next_poi` majority | `next_region` majority |
|---|---|---|---|---|---|---|
| Alabama | 113,846 | 11,848 | 1,109 | 12,709 | (to compute) | 2.33% |
| Florida | 1,407,034 | 76,544 | 4,703 | 159,175 | (to compute) | 22.51% |
| Arizona | ~120K | ~10K | 1,547 | 26,396 | (to compute) | 6.35% |

Florida's 22.5% next_region majority is the single biggest experimental risk — it probably requires `--use-class-weights` (which is now wired) to avoid NashMTL over-weighting the region head and starving next_POI's gradient under MTL.

---

## Critical-path code work before P1 can run

**Estimated total: 2–3h.**

1. **`src/data/inputs/next_poi.py`** — new loader (mirror `next_region.py`): join `sequences_next.parquet::target_poi` → `placeid_to_idx` → `poi_idx`. Output a parquet with emb cols + `poi_idx` + `userid`.
2. **`IoPaths.get_next_poi(state, engine)`** — path helper (mirror `get_next_region`).
3. **`CHECK2HGI_NEXT_POI_REGION` preset** in `src/tasks/presets.py` — `task_a=next_poi` (sequential, ACCURACY primary metric), `task_b=next_region` (sequential, ACCURACY primary metric). Both at `num_classes=0` placeholder; both resolved at runtime via `resolve_task_set`.
4. **`pipelines/create_inputs_check2hgi_next_poi.pipe.py`** — pipeline wrapper using the new loader (mirror the next_region pipe).
5. **`FoldCreator._create_check2hgi_mtl_folds`** — extend to route `next_poi` labels into slot A when `task_set.task_a.name == 'next_poi'`.
6. **`scripts/train.py::_run_mtl_check2hgi`** — accept both presets; resolve `task_a.num_classes` too (not just task_b). This is a new change — the existing code only resolved task_b.
7. **Unit tests:** `tests/test_data/test_next_poi_loader.py` (mirror next_region's 3 tests) + extend `tests/test_tasks/test_presets.py` with the new preset.

After this, the P1 single-task runs can start.

---

## Carry-over advisor & reviewer concerns (from v1 WIP)

These are preserved in `archive/v1_wip_mixed_scope/HANDOFF.md` and `archive/v1_wip_mixed_scope/CRITICAL_REVIEW.md`. Summary:

- **Training-budget matching for CH01** — check2HGI loss still decreasing at ep 500; HGI's training budget unknown. Decision options in `archive/v1_wip_mixed_scope/TRAINING_BUDGET_DECISION.md`. **Open.**
- **Florida class imbalance** — `--use-class-weights` is wired; enable for FL runs. **Resolved (infrastructure) / pending (per-run decision).**
- **FL probe reliability** — linear probe was divergent on FL. If CH11 depends on the probe narrative, the probe should be rerun with class-weighted LR or MLP. **Open.**
- **joint_acc1 scale incoherence** — resolved via `val_joint_lift` normalisation. **Done.**
- **Slot-naming rename** — resolved via `task_a_*`/`task_b_*` across `mtl_cv.py`. **Done.**
- **HANDOFF freshness** — original WIP HANDOFF had stale [BLOCK] items; this document replaces it. **Done.**

---

## Open strategic decisions (needs user call)

1. **Training-budget matching for CH01** (see `archive/v1_wip_mixed_scope/TRAINING_BUDGET_DECISION.md`). Three options: (a) match FLOPs between HGI and Check2HGI, (b) extend Check2HGI only, (c) document 500-epoch cap as scope caveat.
2. **Seed plan.** Single seed (42) → n=5 paired samples, or 3 seeds {42,123,2024} → n=15? Multi-seed triples P5 budget.
3. **Thesis framing.** Narrow ("check-in embeddings + aux task helps next-POI") or broader ("hierarchical input representation matters")? The broader framing commits to Option C cross-attention in P4.
4. **FL probe re-run**. Re-run the linear-probe experiment with class-weighted LR / MLP to salvage CH11's state-dependent-gain narrative, or drop CH11 from the headline?

---

## Test command to reproduce current state

```bash
cd /Volumes/Vitor\'s\ SSD/ingred/.claude/worktrees/check2hgi-mtl
git log --oneline -10

# Legacy + check2HGI tests (must stay green):
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q --ignore=tests/test_integration

# End-to-end CLI smoke on the OLD preset (next_category + next_region) — confirms infra:
PYTHONPATH=src DATA_ROOT=/Volumes/Vitor\'s\ SSD/ingred/data OUTPUT_DIR=/Volumes/Vitor\'s\ SSD/ingred/output \
  python scripts/train.py \
    --state alabama --engine check2hgi --task mtl --task-set check2hgi_next_region \
    --folds 1 --epochs 2 --gradient-accumulation-steps 1 --batch-size 1024 \
    --no-folds-cache
```

The new `check2hgi_next_poi_region` preset + CLI path will be smoked once the code deltas above land.
