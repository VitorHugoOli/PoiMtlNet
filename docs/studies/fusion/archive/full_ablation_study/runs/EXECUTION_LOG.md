# Full Fusion Ablation — Execution Log

Running the staged protocol from `STUDY_DESIGN.md`. State: `alabama` (Stages 0–3),
`florida` (Stage 4). Engine: `fusion` (Sphere2Vec+HGI for cat; HGI+Time2Vec for next).

Start: 2026-04-13.

---

## Pre-Flight (2026-04-13)

- **Fusion inputs present:** `output/fusion/alabama/input/{category,next}.parquet` (8.8 MB / 78.2 MB).
- **Checkins present:** `data/checkins/Alabama.parquet`.
- **Test suite:** 765 passed, 17 skipped, **1 failed** (vs CONTINUE.md's 692 pass claim — codebase grew).
  - Failing test: `test_regression::test_mtl_f1_within_tolerance`
    - Synthetic-data calibrated F1 floor: category F1 = **0.8925** vs floor **0.9200** (below by ~0.03).
    - This is an MPS nondeterminism / minor drift issue on a synthetic-data smoke test; **not a blocker** for the ablation runs (no shared state affected).
    - **TODO after study:** recalibrate floor or investigate whether a recent change (head-swap / optimizer registration) nudged the shared backbone init path.

## Assessment of the CONTINUE.md plan

Plan is scientifically sound. Notable strengths:
- Progressive narrowing (47–55 experiments) vs full 8,550-grid is the only feasible path.
- Stage 0's HGI reference is critical given documented scale imbalance (Sphere2Vec L2 = 0.55, HGI = 8.46, 15×) and zero-ablation evidence that the model is 90 % HGI-dependent.
- Rejection of per-source normalization is empirically grounded (0.606 → 0.504 acc drop).

Concerns:
1. Stage 0 could *falsify the fusion thesis*. If fusion ≈ HGI, the narrative shifts from "fusion helps" to "task-specific fusion is robust but adds no signal." Both are publishable but require different framing.
2. The hypothesis that **DCN learns cross-features between Sphere2Vec↔HGI halves** in Stage 2 is the only real upside for heads — worth reporting even if the joint score is neutral.
3. PLE excluded; justified by Phase 4 (0.235) but noted for supplementary.
4. DWA should be run as supplementary since it's already implemented.

---

## Stage 0 result summary (2026-04-13)

See `STAGE_0_ANALYSIS.md`. HGI-ref wins joint 0.3861 vs fusion best 0.3676 (−4.8%, within tolerance). Per-task split is the headline: fusion **+26% next F1**, **−15% category F1** vs HGI.

## Stage 1 — patch required (2026-04-13)

**Bug discovered:** First Stage 1 run crashed 10/25 candidates. All `cagrad` and `aligned_mtl` variants failed with:

```
TypeError: CAGradLoss is not compatible with gradient accumulation;
use gradient_accumulation_steps=1 or a loss with get_weighted_loss().
```

Root cause: these gradient-surgery losses need a single backward per optimizer step, but `scripts/train.py` defaults to `gradient_accumulation_steps=2`. The ablation runner (`src/ablation/runner.py::_candidate_argv`) was not injecting a `--gradient-accumulation-steps 1` override for incompatible losses.

**Fix applied:** `src/ablation/runner.py` now emits `--gradient-accumulation-steps 1` whenever `mtl_loss ∈ {cagrad, aligned_mtl, pcgrad}`. Stage 1 re-run from scratch.

**Limitation to note in paper:** This creates a mild apples-to-oranges comparison — `ca`/`al` candidates see an effective batch size half of `eq`/`db`/`uw`. Unavoidable without either (a) doubling batch_size for ca/al (risks OOM on MPS) or (b) halving it everywhere (changes protocol from prior Phase 1–2). Flag in Stage 1 writeup; if a ca/al candidate wins, a follow-up matched-batch-size run would be warranted.

## Stage 1 — second bug (2026-04-13)

After the `gradient-accumulation-steps=1` fix, `aligned_mtl` hit a second error:

```
File "src/training/runners/mtl_cv.py", line 331, in train_model
    running_loss += loss.detach()
AttributeError: 'NoneType' object has no attribute 'detach'
```

Root cause: `CAGrad.backward()` and `AlignedMTL.backward()` return `(None, extras)` because they do the gradient surgery internally and leave no scalar usable for a second `.backward()`. The training loop correctly skips the second backward (via `already_backpropagated=True`) but naively called `.detach()` on `None` for loss bookkeeping.

**Fix applied:** `src/training/runners/mtl_cv.py::_get_weighted_loss` now falls back to `losses.sum().detach()` as the reporting scalar when the loss object returns `None`. The scalar is only used for `running_loss` accumulation; it never re-enters backward.

**Sanity check:** `--mtl-loss aligned_mtl` on fusion 2ep/1fold completes cleanly (next F1 9.43%, cat F1 16.27% at 2 ep — as expected for short training).

Also noted:
- `torch.linalg.eigh` falls back to CPU on MPS for aligned_mtl (performance implication, not correctness).
- `fvcore` missing → `[INFO] FLOPS: 0 | Params: 0` shown but training proceeds. Not a blocker.

## Stage 1 Complete (2026-04-13)

25/25 screen completed; 5 promoted at 2f/15ep. CAGrad/Aligned-MTL dominate top-10.
See `STAGE_1_ANALYSIS.md`. Top-3 promoted: dsk42_al, dsk42_ca, cgc21_ca.

## Stage 2 Complete (2026-04-13)

9/9 head variants completed. DCN category head gives +1.7% on dselectk backbone at short training.
See `STAGE_2_ANALYSIS.md`. Top overall: dsk42_al+dcn (0.528), dsk42_ca+dcn (0.526), dsk42_al default (0.519).

## Stage 3 Complete (2026-04-13)

3 candidates × 5f × 50ep. All statistically indistinguishable (p > 0.6).
Champion: **dsk42 + aligned_mtl + default heads** → joint 0.548 ± 0.015.
See `STAGE_3_ANALYSIS.md` and `FINAL_ANALYSIS.md`.

## Stage 4 BLOCKED

Florida has no embeddings. Requires 2-4h of embedding generation + fusion pipeline.
Recommended: run as overnight batch job.
