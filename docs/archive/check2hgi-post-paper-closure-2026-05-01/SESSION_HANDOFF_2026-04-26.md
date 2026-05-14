# Session Handoff — 2026-04-26 (F48-H3-alt champion candidate + F40 / F48-H2 negative controls + docs sync)

**For the next agent picking up this branch.** Supersedes `SESSION_HANDOFF_2026-04-24_PM.md`. Read this first, then `MTL_ARCHITECTURE_JOURNEY.md` for the end-to-end derivation.

## 0 · One-minute summary

- **F48-H3-alt is the champion candidate.** Per-head LR (`cat_lr=1e-3, reg_lr=3e-3, shared_lr=1e-3, --scheduler constant`) closes the F21c gap on AL (MTL exceeds STL by +6.25 pp on reg Acc@10), 75% on AZ, validates at scale on FL (cat preserved + reg +6.7 pp over predecessor B3). **CH18 promoted Tier B → A.**
- **Three negative controls landed** bracketing H3-alt as the unique design in this space:
  - **F40** (loss-side cat_weight ramp 0.75 → 0.25) — cat preserved, reg only +1 pp (Pareto fails)
  - **F48-H1** (constant LR=1e-3, single-LR) — cat preserved, reg flat (α can't grow at gentle LR)
  - **F48-H2** (warmup_constant 50ep ramp + 100ep plateau, single-LR) — cat preserved BUT reg WORSE than B3 by 1.8-4.9 pp (cat-vs-reg compete for shared cross-attn at plateau LR)
- **Docs swept** to reflect H3-alt: NORTH_STAR (champion candidate added), CLAIMS §CH18 (Tier B → A), PAPER_STRUCTURE (champion + validation table), OBJECTIVES_STATUS_TABLE (v3 → v4), AGENT_CONTEXT (thesis reformulated), README (entry point reordered), RESULTS_TABLE (rows for H3-alt + F40 + F48-H1/H2 + F45 + F21c populated), CONCERNS §C15 (resolved). Plus new `MTL_ARCHITECTURE_JOURNEY.md` for the narrative.

## 1 · Recipe (paper-strength MTL-over-STL, all 3 states)

```bash
python scripts/train.py --task mtl --task-set check2hgi_next_region \
    --state <al|az|fl> --engine check2hgi \
    --model mtlnet_crossattn --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_getnext_hard \
    --reg-head-param d_model=256 --reg-head-param num_heads=8 \
    --reg-head-param "transition_path=$OUTPUT_DIR/check2hgi/$STATE/region_transition_log.pt" \
    --task-a-input-type checkin --task-b-input-type region \
    --folds 5 --epochs 50 --seed 42 \
    --batch-size 2048 \                # 1024 for FL (MPS OOM at 2048)
    --scheduler constant \              # ← H3-alt requires constant
    --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3   # ← per-head LR
```

Launchers: `scripts/run_f48_h3alt_per_head_lr.sh` (AL+AZ), `scripts/run_f48_h3alt_fl.sh` (FL).

## 2 · Headline numbers (5-fold × 50 ep, seed 42)

| State | cat F1 (B3 → H3-alt) | reg Acc@10 (B3 → H3-alt) | vs STL F21c GETNext-hard | Cat preserved? |
|---|---|---|---|:-:|
| AL | 42.71 → **42.22 ± 1.00** (-0.49) | 59.60 → **74.62 ± 3.11** (+15.02) | **+6.25 EXCEEDS** ✓ | ✓ |
| AZ | 45.81 → **45.11 ± 0.32** (-0.70) | 53.82 → **63.45 ± 2.49** (+9.63) | -3.29 (75% closed) | ✓ |
| FL | 65.72† → **67.92 ± 0.72** (+2.20) | 65.26† → **71.96 ± 0.68** (+6.70) | TBD F37 | ✓ |

†FL B3 = F32 1-fold n=1.

## 3 · What ran this session (chronological)

| Order | Experiment | State | Status | Key finding |
|---:|---|---|---|---|
| 1 | F43 (B3 cat_weight=0.01) | AL | done | reg=60.91 ≈ B3 — refutes joint-loss-signal; LR is the bottleneck |
| 2 | F48-H3 (cat=1e-3, reg=3e-3, **shared=3e-3**, const) | AL+AZ | done | reproduces F45 — cat collapses; shared cross-attn LR is the cat-destabilizer |
| 3 | F48-H3-alt (cat=1e-3, reg=3e-3, **shared=1e-3**, const) | AL+AZ | 🎯 **WIN** | cat preserved, AL exceeds STL by +6.25 pp |
| 4 | F48-H3-alt FL | FL | done (batch=1024, ~4.3h) | scales: cat 67.92, reg 71.96 |
| 5 | F40 (scheduled cat_weight 0.75→0.25 ramp) | AL+AZ | done | refuted Pareto: cat OK, reg only +1 pp |
| 6 | F48-H2 (warmup_constant single-LR 150ep) | AL+AZ | done | refuted: cat OK, reg WORSE than B3 |

## 4 · Mechanism (single sentence)

α (graph-prior weight in `next_getnext_hard.head`) needs sustained 3e-3 LR to grow → reg lift; `shared_lr=1e-3` keeps cross-attn gentle so cat path stays stable; `cat_lr=1e-3` keeps cat encoder/head from diverging. The earlier monolithic-LR family (F44-F48-H2) couldn't satisfy both because it forced α and the cat path to share an LR.

The F45 reg-lift mechanism was actually two coupled effects: (a) α growth + (b) cat collapse → uncontested reg gradient through shared cross-attn. F48-H3 (shared=3e-3) replicates F45 because shared at 3e-3 destabilises cat regardless of cat-encoder LR. F48-H2 (warmup-then-plateau, single LR) preserves cat through the ramp, but at the plateau the surviving cat competes with reg for shared cross-attn capacity, starving α — reg drops below B3. H3-alt decouples the regimes by giving α its own LR group while keeping shared gentle.

## 5 · Pending work (priority-ranked)

| Priority | Item | Notes |
|---|---|---|
| P1 | **F37 STL `next_gru` cat 5f** — 4050-assigned, user runs | Once landed, gives the FL "MTL exceeds STL" headline analogous to AL |
| P1 | **Seed sweep on H3-alt** — {0, 7, 100} on AL+AZ | Hardens σ confidence interval for the AL surplus claim |
| P2 | **Wilcoxon paired test H3-alt vs B3** across 5 folds | Formal statistical strength claim, similar to F27 cat-head Wilcoxon p=0.0312 |
| P2 | **OneCycleLR per-head variant** — cat max=1e-3, reg max=3e-3, shared max=1e-3 | Might tighten reg σ on AZ via late annealing |
| P3 | **CA + TX upstream pipelines** (F22/F23) → CA/TX H3-alt 5f (F24/F25) | Headline paper expansion |
| P3 | **Combined recipe** — H3-alt + scheduled cat_weight | Unlikely to beat H3-alt alone but cheap to test |
| P4 | **Mechanism instrumentation** — log α value per epoch per fold | Confirm the α-growth claim quantitatively |

See `MTL_ARCHITECTURE_JOURNEY.md §9` for the longer-form follow-up list.

## 6 · Code changes landed this session

| File | Change |
|---|---|
| `src/models/mtl/mtlnet_crossattn/model.py` | Added `cat_specific_parameters()` and `reg_specific_parameters()` methods; `shared_parameters()` unchanged. |
| `src/training/helpers.py` | New `setup_per_head_optimizer` (3-group AdamW). Guarded `constant`/`cosine` scheduler from overwriting per-group LRs. New `warmup_constant` scheduler (LinearLR → ConstantLR via SequentialLR). |
| `src/configs/experiment.py` | Added `cat_lr / reg_lr / shared_lr: Optional[float]` fields. |
| `src/training/runners/mtl_cv.py` | Per-head detection branches `setup_per_head_optimizer` vs `setup_optimizer`. Smoke print of optimizer groups on fold 0. New epoch hook calling `mtl_criterion.set_epoch(epoch_idx)` (no-op for losses without it). Auto-default `total_epochs=config.epochs` for `scheduled_static`. |
| `scripts/train.py` | New flags `--cat-lr / --reg-lr / --shared-lr` (all-or-nothing validation). New `--scheduler warmup_constant` choice. |
| `src/losses/scheduled_static/{__init__.py,loss.py,metadata.yaml}` | New `ScheduledStaticWeightLoss` — interpolates cat_weight linearly from start → end across epochs with optional warmup. |
| `src/losses/registry.py` | Registered `scheduled_static`. |

All back-compat (defaults preserve legacy behaviour).

## 7 · Scripts added this session

| Script | Purpose |
|---|---|
| `scripts/run_f48_h3_per_head_lr.sh` | F48-H3 (sh=3e-3) — refuted; cat collapses |
| `scripts/run_f48_h3alt_per_head_lr.sh` | F48-H3-alt (sh=1e-3) — winner, AL+AZ |
| `scripts/run_f48_h3alt_fl.sh` | F48-H3-alt FL (batch=1024) |
| `scripts/run_f40_scheduled_handover.sh` | F40 scheduled cat_weight 0.75→0.25 — refuted |
| `scripts/run_f48_h2_warmup_constant.sh` | F48-H2 warmup_constant 150ep — refuted |

## 8 · Research notes added this session

- `research/F48_H3_PER_HEAD_LR_FINDINGS.md` — H3 + H3-alt + FL scale validation
- `research/F40_SCHEDULED_HANDOVER_FINDINGS.md` — loss-side negative control
- `research/F48_H2_WARMUP_CONSTANT_FINDINGS.md` — warmup-plateau negative control
- `MTL_ARCHITECTURE_JOURNEY.md` — end-to-end narrative

## 9 · Files NOT to commit (transient)

- `/tmp/check2hgi_logs/*.log` — verbose run logs
- `results/check2hgi/*/mtlnet_*_2026042*/summary/full_summary.json` — per-run summaries (live under `results/`, gitignored, content is referenced not committed)

## 10 · Commits this session

| SHA | Subject |
|---|---|
| `565c478` | F48-H3 + H3-alt infra + AL+AZ |
| `439e6b2` | H3-alt FL scale validation |
| `87e216e` | F40 scheduled handover (refuted) |
| `664706a` | F48-H2 warmup_constant (refuted) |
| `035afc1` | MTL_ARCHITECTURE_JOURNEY.md + NORTH_STAR/CLAIMS/README updates |
| (this) | RESULTS_TABLE / PAPER_STRUCTURE / OBJECTIVES / AGENT_CONTEXT / CONCERNS sweep |

## 11 · How to resume

1. Read `MTL_ARCHITECTURE_JOURNEY.md` (15 min) — gives the full causal narrative from initial design to H3-alt
2. Read this handoff (5 min) — current state + pending work
3. Read `NORTH_STAR.md` (3 min) — recipe + numbers
4. If running F37 on 4050: see `scripts/run_stl_next_gru_cat.sh` — also has STL FL
5. If running seed sweep: copy `scripts/run_f48_h3alt_per_head_lr.sh` and parameterize `--seed`
6. If running CA/TX upstream: see existing FL pipeline (`pipelines/embedding/check2hgi.pipe.py` plus `scripts/regenerate_next_region.py` plus `scripts/compute_region_transition.py`)
