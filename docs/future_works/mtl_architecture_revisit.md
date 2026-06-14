# Future Work — MTL architecture revisit (rigorous re-implementation)

**Date drafted:** 2026-05-20
**Updated:** 2026-05-28 — execution superseded by [`docs/studies/archive/mtl_improvement/`](../studies/archive/mtl_improvement/) (T0-T8 chain, active on branch `mtl-improve`). This memo retained as the *forward-looking design rationale*; §4.1 per-task best-epoch shipping was **moved out** of this scope into [`docs/studies/archive/substrate-protocol-cleanup/`](../studies/archive/substrate-protocol-cleanup/) (see Tier C there) because it is a protocol/serving concern, not an architectural one.
**Source:** [`docs/studies/mtl-exploration/considerations.md`](../studies/mtl-exploration/considerations.md) + [`docs/studies/mtl-exploration/EXPERIMENT_NO_ENCODERS.md`](../studies/mtl-exploration/EXPERIMENT_NO_ENCODERS.md) findings + Tier 6 closure (2026-05-19)
**Sequencing:** **next-tier successor to `mtl-protocol-fix`**. Once F1 has fixed the selector and characterised the residual MTL-vs-STL reg gap, this study takes over. **The mechanism motivation is P4 frozen-cat horizon test (Phase 2 of `mtl-protocol-fix`)**: MTL reg peaks at ep 2 even with cat frozen and `cat_weight=0` from epoch 0 — meaning the residual is NOT cat-vs-reg interference but the shared backbone itself. The upper-bound benchmark for any architectural winner is the §4.2 composite (STL c2hgi cat + STL HGI reg) deploy ceiling: +7 to +12 pp vs MTL@disjoint at AL/AZ/FL/CA/TX (Phase 3 Rank 4, `phase3_rank4_composite_analysis.md`).

## What's deferred

A rigorous, faithfully-implemented MTL architecture ablation. The previous ablation (F-trail F30s) tested *"very simple variants"* of the proposed architectures (`mtlnet_crossstitch`, `cgc`, `mmoe`, `dselect_k`) that did not match the literature implementations and produced unreliable per-task numbers because the comparison only looked at joint metrics.

The user's framing:

> "the original ablation study on the arch of the MTL was with a very simple archs variants of the proposes apporachs like the cross-stitch and other apporachs that you can find in the codebase; I belive worth to revise this study in a more rigorous way, where we evaluate the original paper and code for the approach and do a real and strong implementation of the approaches for our usecase, and maybe even propose some change to better fit to our usecases and inputs, as we had to do with the crossattn"

## Scope

| Family | Concrete architectures | Why |
|---|---|---|
| **Soft-shared** | MMoE (faithful), CGC (faithful), DSelect-K (faithful), PLE | Soft gating between task-shared and task-specific representations |
| **Hard-shared** | Hard parameter sharing with task heads only; ablate shared-backbone depth | Lower-bound baseline |
| **Dynamic-shared** | Cross-stitch (faithful), sluice networks, NDDR-CNN-style fusion | Per-layer task interaction |
| **Hybrid** | Cross-stitch + cross-attention; MMoE + cross-attention; soft-PLE + per-task-modality input | User-suggested combinations |
| **Per-task evaluation** | Every variant reports cat F1 and reg Acc@10 (and reg MRR) separately, not just a joint metric | Closes the "joint metric blind spot" the user flagged |

## Pre-requisites (gates)

1. ✅ `mtl-protocol-fix` lands the F1 selector fix.
2. ✅ Three-frontier evaluation protocol (best joint + best disjoint + STL ceiling) is operational.
3. The residual MTL-vs-STL reg gap that survives F1 is documented per-state.
4. **Paper canon re-evaluation under whatever arch wins** (see [`paper_canon_reevaluation.md`](paper_canon_reevaluation.md)) — sequenced AFTER this study's winner is locked.

## Acceptance criterion

When picked up:

1. Faithful implementations of MMoE, CGC, DSelect-K, cross-stitch (mirroring published papers; not legacy simplified variants).
2. Each variant runs under per-fold seed-tagged log_T (leak-free post-2026-05-15).
3. Per-task results table: cat F1 + reg Acc@10 + reg MRR + the three frontiers per state.
4. 5 seeds × 5 folds minimum at FL; AL/AZ for cross-state validation.
5. Winner promotion criterion: paired Wilcoxon p ≤ 0.05 on AT LEAST ONE head at FL n=20, no state regression > σ_fold on the other head.
6. If a winner is promoted: paper §0.1 needs revision (see `paper_canon_reevaluation.md`).

## Cost (estimated)

- Faithful re-implementation of MMoE/CGC/DSelect-K: ~2-3 days code + unit tests.
- Cross-stitch / sluice / NDDR-CNN: ~2 days code + unit tests.
- Hybrid combinations: ~1-2 days code each.
- Multi-seed sweep at FL × 4-6 variants × 5 seeds × 5 folds = ~30-50 GPU-h.
- Cross-state AL/AZ validation: ~10-15 GPU-h.
- **Total: ~3 weeks calendar / ~50-80 GPU-h.**

## Risks / caveats

1. **The current MTL ablation table (`docs/context/MTL_ARCHITECTURES.md`) cites legacy fusion-study CGC/MMoE/DSelect-K numbers from 1-fold × 10-epoch on the legacy 7+7 task pair** (per `mtl-exploration/README.md` §Gaps). Numbers there will need full migration or supersession.
2. **Per-task balance:** the current static_weight w_cat=0.75 is hand-tuned to the cross-attn backbone. New architectures may need re-tuning; treat as a co-axis with loss-balancing (`substrate_adaptive_mtl_balancing.md`).
3. **Head re-design interplay:** see `head_window_batch_audit.md` — if heads are also revised, sequence head work BEFORE arch (or co-design).

## Live docs the work would touch

- `src/models/mtl/` — new architecture modules
- `src/configs/experiment.py` — new model presets
- `docs/context/MTL_ARCHITECTURES.md` — full rewrite
- `docs/MTL_ARCHITECTURE_JOURNEY.md` — append new chapter
- `docs/results/RESULTS_TABLE.md` — new §0.5 section for arch-axis ablation
- `docs/NORTH_STAR.md` — possible champion update
- `articles/[BRACIS]_Beyond_Cross_Task/PAPER_DRAFT.md` §Architecture — revision

## Includes paper-canon multi-seed re-evaluation

This study's winner triggers the deferred [`paper_canon_reevaluation.md`](paper_canon_reevaluation.md) (Rank 2 from the canonical_improvement post-closure memo). Sequenced together to avoid double-revisions.

## Pointers

- User's framing: [`docs/studies/mtl-exploration/considerations.md`](../studies/mtl-exploration/considerations.md) (bullets on arch revisit, soft/hard/dynamic shared, hybrid combinations)
- Existing scaffolded backbone: `src/models/mtl/mtlnet_crossstitch/` (never tested head-to-head with cross-attn)
- Cross-attn champion reference: `src/models/mtl/mtlnet_crossattn/`
- Legacy ablation (to supersede): `docs/context/MTL_ARCHITECTURES.md` (1-fold × 10-epoch on legacy 7+7 task pair)
