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
| checkpoint-selector | geom_simple | argparse default (v12/v15/v16 omit it from bundle) | recommend pin into bundles | ◻ pending (identity-preserving) |
| build provenance (stride, min_seq) | recorded | not recorded | recommend ExperimentConfig/RunManifest fields + sidecar | ◻ pending |

## Guards added (`train.py:_preflight_canon_guards`, WARN-only → numerically inert; `MTL_STRICT=1` hard-fails)
1. **dev-seed 42** — canon-active MTL with no `--seed` → WARN (paper-grade needs {0,1,7,100}).
2. **wrong substrate** — canon-active MTL where `--engine` ≠ the bundle's pinned substrate → WARN.
3. **torch build** — `torch != 2.11.0+cu128` → WARN (2.12 rewrote TopK → reg Acc@10 tie-break shift).

## The biggest anti-stumble lever (recommended, pending user sign-off)
**One canonical P3-board driver** (`scripts/closing_data/p3_board.*`) as the ONLY sanctioned build+run path:
(1) regenerate the base at `--min-seq 10 --stride 1` (the only place these non-default values live);
(2) `train.py --canon v16 --compile --tf32 --seed <s>` per cell, uniformly (all-cells-same compile/tf32);
(3) `--canon` pinned per cell + PID-suffix rundir capture (no `ls -dt|head`); (4) run the torch/seed/engine
preflight first. Today the board is assembled ad-hoc from scattered `run_*` scripts — that *is* the stumble.

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
