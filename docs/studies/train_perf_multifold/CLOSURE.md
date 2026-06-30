# train_perf_multifold — STUDY CLOSURE

**Status: CLOSED 2026-06-30.** Started 2026-06-26 as a perf/tooling study (4 goals); grew into a batch-size +
per-head-LR investigation that produced a board-wide quality win. All proposed work delivered; future work below.

---

## 1 · Original goals (PLAN.md §0, user 2026-06-26) — ALL DONE

| # | Goal | Status | Evidence |
|---|---|---|---|
| 1 | **Reproduce + validate** AL champion-G MTL + STL ceilings on the A40 ≈ RESULTS_BOARD | ✅ | `baseline_runs/` — MTL 63.18/69.73, STL 55.73/69.98 (all ≤0.14 of board) |
| 2 | **Audit train pipeline for speed (quality-neutral)** | ✅ | P1 (STAN `.any()` graph-breaks 10→2, eager byte-identical + unit test) + P3 (OOD cache) + P6 (pin CPU batches); the `--profile` audit tool (`src/training/profiling.py`) |
| 3 | **Multi-fold fan-out** (5 folds → separate procs, one rundir; `--only-folds`) | ✅ | `--only-folds`/`--run-id`/`--per-fold-seed`, `aggregate_folds.py`, `run_folds_fanout.sh`. 5-way concurrent: 0 conflicts, fold-1 solo == concurrent **byte-identical** |
| 4 | **Opportunistic cleanup** | ✅ | engine-allow-list dedup (paths.py SoT), mtl_eval import/typo, dataset.py comment |

**Hard invariant held throughout:** every change gated by an AL champion-G + STL A/B within fold-std; byte-identical
changes stayed byte-identical; frozen §0.1 contract intact.

## 2 · Emergent work (layered on after the 2026-06-26 close) — ALL DONE

| Area | Outcome | Where |
|---|---|---|
| **Champion-aligned defaults** | `MTL_SKIP_INERT_LOGT` default-on + **auto-fp32** for large-C MTL (byte-identical) | `mtl_cv.py`, [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`] |
| **mtl_cv decomposition** | extracted `_optimizer_micro_step` + `_run_validation_epoch` + builders, each parity-gated | `mtl_cv.py`, `DECOMPOSITION_PLAN.md` |
| **System reference** | champion config + full env/flag table | [`../../SYSTEM_REFERENCE.md`](../../SYSTEM_REFERENCE.md) |
| **Batch-size study** | bs=8192 small-state cat WIN (AL +0.36/AZ +0.75 n=20); FL regression diagnosed | `BATCH_SIZE_SWEEP.md`, `RESULTS_SUMMARY.md` |
| **Per-head-LR bug + fix** | `--cat-lr/reg-lr/shared-lr` were INERT under onecycle → `MTL_ONECYCLE_PER_HEAD_LR` (opt-in, default-OFF byte-identical) + `--adam-beta2` | `helpers.py`, [`../../future_works/per_head_lr_onecycle_fix.md`] |
| **Per-head cat-lr 1e-3 WIN** | board-wide quality lever, **n=20 confirmed**: AL +0.64/AZ +1.52 over uniform; FL bs8192+cat-lr-1e-3 BEATS bs2048 champion +0.17cat/+0.20reg +7% faster | [`../closing_data/perhead_lr_n20.md`](../closing_data/perhead_lr_n20.md) |
| **FL mechanism** | the bs8192 FL cat regression = pure **cat-LR overshoot** (reg-capture refuted by isolation decomposition) | `BATCH_SIZE_SWEEP.md` §FL MECHANISM FINAL |
| **Quality-improvement catalog** | 27 levers ranked/verified (swarm) — on standby | `QUALITY_IMPROVEMENTS.md` |
| **Execution flows doc** | canonical v14-embedding + gated-overlap + train flows | [`../../flows/README.md`](../../flows/README.md) |

## 3 · Headline result
**bs=8192 + per-head cat-lr 1e-3** (`MTL_ONECYCLE_PER_HEAD_LR=1`) beats the frozen champion at every tested state
(n=20): AL +1.0 / AZ +2.3 cat; FL +0.17 cat / +0.20 reg AND ~7% faster — flipping the earlier "keep bs=2048 at FL".
Enabled by fixing the latent "per-head LR inert under onecycle" bug.

## 4 · Future work (documented, NOT done)

**Promotion / follow-up experiments**
- [ ] **Promote `bs=8192 + cat-lr 1e-3` into the champion board-wide** (needs `MTL_ONECYCLE_PER_HEAD_LR`; update CANONICAL_VERSIONS + NORTH_STAR + RESULTS_BOARD). Confirm it's not a §0.1-repro break (it's a new canon, not the frozen one).
- [ ] **CA/TX** — the only states not covered by the n=20 per-head confirmation (large-C; auto-fp32 path).
- [ ] **Per-head-LR parity test** — eager byte-identical with `MTL_ONECYCLE_PER_HEAD_LR` OFF (in `future_works/per_head_lr_onecycle_fix.md` checklist).

**Perf / tooling (from PLAN §6)**
- [ ] **P7 compile-once-across-folds** (biggest sequential-run lever; delicate per-fold log_T buffer swap → A/B).
- [ ] **bf16/A40 fp32-attn island** (`_STANAttention`) — DEFERRED A40-bf16 mitigation; validate on the A40 (the grad-NaN only reproduces on Ampere at CA/TX C 6.5–8.5k), NOT H100. Driver `../closing_data/run_bf16_island.sh`.
- [ ] **MED cleanup** — unify `_run_category`/`_run_next` + `category_cv`/`next_cv`; one `build_autocast_ctx`.
- [ ] **Single-task fan-out** (`--run-id`/`--per-fold-seed` for `next_cv`/`category_cv`) — infra is generic; wire it.
- [ ] **n_regions pass-through** so a fan-out process doesn't transiently build all 5 folds for the region-count scan.

**Catalog (standby, user-paused)**
- [ ] **QUALITY_IMPROVEMENTS.md** 27-lever catalog — breaking/quality-change improvements, on standby.

## 5 · Artifacts (this study)
Docs: `PLAN.md`, `DECOMPOSITION_PLAN.md`, `AUDIT_FINDINGS.md`, `BATCH_SIZE_SWEEP.md`, `RESULTS_SUMMARY.md`,
`QUALITY_IMPROVEMENTS.md`, `log.md`, this `CLOSURE.md`. Code: `MTL_ONECYCLE_PER_HEAD_LR`, `--adam-beta2`,
`MTL_SKIP_INERT_LOGT`/auto-fp32 defaults, the profiler + fan-out tooling, the mtl_cv decomposition. PR **#56**.
Results saved to `../closing_data/perhead_lr_n20.md`.
