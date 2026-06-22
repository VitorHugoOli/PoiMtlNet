# Going-forward defaults, guards & the anti-stumble contract

> A40, `study/pre-freeze-a40`, 2026-06-19. Audited (workflow `wf_aa8dcada`, 3 audits + adversarial verify):
> are the session's settled decisions reflected as code defaults / guards so a future agent/dev who just
> "runs the project" does NOT hit the wrong flow? **Answer: the recipe IS defaulted; the 4 board-values are
> deliberately NOT global defaults** (flipping them silently breaks reproduction of the frozen §0.1 numbers),
> and three silent traps are now caught by WARN guards.

## The key distinction
- **Recipe identity (champion-G / v16)** → **enforced as the default.** `DEFAULT_CANON="v16"` (`canon.py:26`)
  auto-injects the full bundle on a bare `train.py --task mtl` (model/heads/loss/selector/LRs/KD/input-modality/
  engine). Verified complete + test-locked (`tests/test_configs/test_canon.py`). A bare run reproduces the
  champion **recipe** exactly. The old CLAUDE.md "6 silently-wrong flags" trap is handled by canon.
- **Board execution / windowing values** → **NOT global defaults, by design.** The frozen v11/v14 substrates +
  the §0.1 numbers were built at MIN_SEQ=5 / stride-9 / no-knobs. A silent global flip would desync new builds
  from the frozen artifacts and break reproduction. These live in the **P3 board recipe/driver**, not in
  `core.py`/`canon.py`.

## Decision table

| param | adopted | current default | mechanism | status |
|---|---|---|---|---|
| recipe (v16 heads/loss/selector/LRs/KD/modality/engine) | v16 | `DEFAULT_CANON=v16` | CODE-DEFAULT | ✅ enforced + test-locked |
| dataset-on-GPU auto-fit | auto | `folds._dataset_device` | CODE-DEFAULT | ✅ enforced (byte-identical) |
| num_workers | 0 | hard-0 (`folds.py`) | CODE-DEFAULT | ✅ enforced (workers rejected) |
| batch-size / n_splits / window | 2048 / 5 / 9 | bundle + `InputsConfig` | CODE-DEFAULT | ✅ enforced |
| loss-scale-norm | OFF | `False` | CODE-DEFAULT | ✅ enforced (excluded) |
| **MIN_SEQUENCE_LENGTH** | **10** (P3 rebuild) | **5** (`core.py:17`) | **P3-BOARD-RECIPE** | ⚠ NOT global — flip only at the P3 rebuild |
| **stride / overlap** | **1** (P3 board) | **9 / non-overlap** (`core.py:26,56`) | **P3-BOARD-RECIPE** (or `check2hgi_dk_ovl` engine) | ⚠ NOT global — Lane-2-gated |
| **compile + tf32** | **ON** (P3 board) | **OFF** (`train.py`), not in canon | **P3-BOARD-RECIPE** | ⚠ board driver passes uniformly; NEVER in canon |
| **seed (reporting)** | {0,1,7,100} | None → **42** (dev seed) | **FAIL-LOUD-GUARD** | ✅ guard added (WARN) |
| **engine ↔ substrate** | v16 ⇒ v14 substrate | user `--engine` wins | **FAIL-LOUD-GUARD** | ✅ guard added (WARN) |
| **torch 2.11+cu128** | pinned | only lane1 shell guard | **FAIL-LOUD-GUARD** | ✅ guard added to `train.py` (WARN) |
| checkpoint-selector | geom_simple | argparse default (v12/v15/v16 omit it) | pin into bundles | ✅ pinned (v12/v15/v16; identity-preserving) |
| build provenance (stride, min_seq) | recorded | not recorded | ExperimentConfig fields + sidecar | ✅ added (config fields + `<task>_build_provenance.json`) |
| stride/min_seq plumbing | reachable | not threaded | parameterize build path | ✅ threaded `--stride`/`--min-seq` (DEFAULT-PRESERVING; no-flag path byte-identical) |

## Guards added (`train.py:_preflight_canon_guards`, WARN-only → numerically inert; `MTL_STRICT=1` hard-fails)
1. **dev-seed 42** — canon-active MTL with no `--seed` → WARN (paper-grade needs {0,1,7,100}).
2. **wrong substrate** — canon-active MTL where `--engine` ≠ the bundle's pinned substrate → WARN.
3. **torch build** — `torch != 2.11.0+cu128` → WARN (2.12 rewrote TopK → reg Acc@10 tie-break shift).

## The biggest anti-stumble lever — BUILT (safety-stopped pending 3 launch-blockers)
**`scripts/closing_data/p3_board.sh`** is the ONLY sanctioned build+run path: (1) rebuild the base at
`--stride 1 --min-seq 10` reusing the frozen v14 embeddings; (2) `train.py --canon v16 --compile --tf32
--seed <s>` per cell, uniformly; (3) `--canon` pinned per cell + PID-suffix rundir capture (no `ls -dt|head`);
(4) torch + log_T-freshness preflight. The four board values live HERE and nowhere else. **`--dry-run` works
(prints the 24-cell plan); a real run currently REFUSES (`exit 1`)** behind a LAUNCH SAFETY-STOP because the
adversarial review found 3 launch-blockers that would corrupt the frozen substrate / produce wrong numbers:

> **P3 board-build launch-blockers (fix before `P3_BOARD_FORCE=1`):**
> 1. `build_inputs` overwrites the frozen v14 `next.parquet`/`sequences_next.parquet` IN PLACE then aborts →
>    stage stride-1 inputs to a separate dir (or back-up/restore); never clobber the frozen substrate.
> 2. `build_design_next_region.py` joins against the CANONICAL stride-9 sequences → hard-fails at stride-1 →
>    needs a stride-aware next_region builder (join the engine's OWN stride-1 sequences).
> 3. `compute_region_transition.py --per-fold` is hardwired to `CHECK2HGI` → emits stride-9 log_T while the
>    mtime guard passes (copy+touch) → the +8..+12pp stale-log_T trap → make it stride/engine-aware.

> 4. **Stride-1 tail-window label skew (M1, `WINDOWING_AUDIT.md`)** — at stride=1 each user emits ~8 OOB
>    tail windows whose target is the user's LAST POI (leak-free but skews the label distribution toward
>    end-of-history). DECIDE before the board build: gate the tail (`emit_tail=False` for stride=1, changes
>    numbers) or keep+document. Not a launch *blocker*, but a board-recipe decision.

These three are P3 infra (the board is post-freeze). The parameterization + provenance + the driver skeleton
ship now; the driver is inert (refuses to run) until they land — so it can't be a stumble.

### A 4th infra OOM — the STL reg-ceiling eval — FIXED 2026-06-22 (`fix(p1): S2-analog CPU val metric`)
Distinct from the 3 build-blockers above (this is the **eval** path, not the build). `p1_region_head_ablation.
_train_single_task` materialised the FULL val logit `[N_val, C]` on the GPU (`torch.cat(all_logits)`) before
scoring → OOM at large-C overlap scale (TX overlap: 766083×6553×4B ≈ 20 GB; CA worse). The MTL trainer got
S1/S2 (`OOM_MEMORY_FIX.md`); the STL ceiling never did. **Fix:** `_should_chunk_val_metric(n_val, n_classes)`
**auto-routes the val metric to CPU when the full val logit would exceed `P1_S2_AUTO_BUDGET_GB` (default 4 GB,
matching MTL's `MTL_S2_AUTO_BUDGET_GB`)** — DEFAULT-ON, so it fires WITHOUT any env; `MTL_CHUNK_VAL_METRIC=1` /
`P1_CHUNK_VAL_METRIC=1` force it. Dataset stays on GPU; only the val logits move → the GPU cat is gone.
Identical at reporting precision (`compute_classification_metrics` is device-agnostic + already chunked; rank
uses strict `>`). **Pinned by `tests/test_scripts/test_p1_val_chunk_guard.py`** (gate logic + CPU≡CUDA ≤1e-6).

## Reproduction / desync TRAPS — NEVER do
1. Never flip `core.py:17` MIN_SEQ 5→10 globally (desyncs frozen v11/v14 rebuild + confounds Lane-2).
2. Never flip `core.py:26` stride None→1 globally (8.5× rows everywhere, OOMs large states, double-counts the base change). Keep overlap engine-/board-scoped.
3. Never put `--compile`/`--tf32` in canon or as a code default (breaks byte-identical reproduction of frozen §0.1; they're perf knobs, not recipe identity). Board-execution-only, applied uniformly.
4. Never bake MIN_SEQ/stride/compile/tf32 into any canon bundle (canon = reproduction identity, reused by `--canon v11/v12/v15`).
5. Never "restore" the alt-opt flags (`--alternating-optimizer-step`/`--alpha-no-weight-decay`/`--min-best-epoch`) into v16 — champion G is onecycle no-alt-opt (B9 carries them; that's the small-state recipe).
6. Never trust a bare `--task mtl` run for paper numbers (dev-seed 42; the guard now warns).
7. Never run a freeze-grade comparison on torch ≠ 2.11.0+cu128.
8. Never "fix" CA/TX region-MTL OOM by lowering bs=2048 (region-MTL diverges at smaller bs — it's GPU routing).
9. Never pin a single seed into canon to "fix" the seed trap (champion G is a {0,1,7,100} multi-seed result).
10. Never revert p1's `_should_chunk_val_metric` CPU-val guard or disable its default-on auto-budget (it stops the
    STL reg-ceiling OOMing at TX/CA overlap scale; CPU≡GPU at reporting precision). `tests/test_scripts/
    test_p1_val_chunk_guard.py` pins it. (Mirror lesson: a memory fix here was once silently reverted by a merge —
    `33fe18da`→`dade24ad`, OOM_MEMORY_FIX.md.) Don't "optimise" it back to a GPU `torch.cat` of the full val logit.
