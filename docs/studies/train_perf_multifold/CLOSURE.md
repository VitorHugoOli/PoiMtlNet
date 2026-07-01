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
- [x] **Promote `bs=8192 + cat-lr 1e-3` as a champion candidate** — DONE 2026-06-30 (advisor-gated): added **`--canon v17`**
      (v16 + bs8192 + new `--onecycle-per-head-lr` CLI flag; `_V16` extracted so v16 stays byte-identical) + the CANONICAL_VERSIONS §v17 /
      NORTH_STAR / RESULTS_BOARD records. **§0.1 (v11) confirmed UNAFFECTED** (separate frozen cosine bundle; per-head fix is
      onecycle-only). **SETTLED 2026-07-01: `DEFAULT_CANON` flipped to v17** (v16 via `--canon v16`; env stays default-OFF, v17 sets the flag). Remaining before §1-headline:
- [~] **CA/TX n=20 for v17 → H100** (2026-07-01). Measured **~52 min/epoch** for CA overlap-MTL at bs8192 fp32
      on the A40 → ~9 days/cell, ~72 days for 8 cells: **infeasible on the A40** (confirms the standing
      "overlap-MTL board is H100-only" finding). Full n=20 packaged for the H100:
      [`../closing_data/CATX_V17_N20_H100_HANDOFF.md`](../closing_data/CATX_V17_N20_H100_HANDOFF.md) +
      `run_catx_v17_n20_h100.sh`. A40 runs a **fold-0 audit** of CA+TX in the meantime
      (`run_catx_v17_audit_1fold.sh`, `catx_v17_audit/`) — confirms the recipe runs end-to-end at the big states.
      + land the **flag-OFF eager-parity test**; then update the §1 headline.
- [ ] **CA/TX** — the only states not covered by the n=20 per-head confirmation (large-C; auto-fp32 path).
- [ ] **Per-head-LR parity test** — eager byte-identical with `MTL_ONECYCLE_PER_HEAD_LR` OFF (in `future_works/per_head_lr_onecycle_fix.md` checklist).

**Code hygiene (from the flows-doc audit, 2026-06-30)**
- [ ] **Move the log_T computation `src ← scripts`** (user-flagged, audit-confirmed). The whole transition-matrix
      kernel — `_log_probs_from_rows`, `build_transition_matrix`, `build_transition_matrix_from_userids`,
      `_build_per_fold`, `save` (+ helpers `_load_graph_maps`, `_resolve_split_engine`) — lives in
      `scripts/compute_region_transition.py`, and **5+ scripts import it as a library** (`scripts/second_dataset/build_inputs.py`,
      `build_chrono_split.py`, `build_region_variant.py`, baselines). `src/` only *consumes* the saved `.pt`. Extract
      to **`src/data/region_transition.py`**, leave `scripts/compute_region_transition.py` as a thin argparse CLI that
      re-exports the names (→ zero importer changes). Behavior-preserving; gate with a re-run → byte-identical `.pt`.
      (NOTE: the gated-overlap windowing is already correctly in `src` — `core.generate_sequences` +
      `builders._resolve_emit_tail`; that one needs no move, and stride-1/min_seq-10 must stay non-default per DEFAULTS_AND_GUARDS.)

**Promotion / follow-up (cont.)**
- [ ] **QUALITY_IMPROVEMENTS A1 — true-fp32 MTL board-wide** (the catalog's #1 ADOPT-rated lever + a *correctness* fix:
      the MTL-vs-STL reg gap is partly an fp16-MTL-vs-fp32-ceiling artifact). auto-fp32 covers C>2000 (DONE); A1's ask
      is to extend fp32 to small-state MTL too. Scope = MTL cells only (STL ceilings already fp32); n=20 re-baseline at FL+CA.
- [ ] **Propagate the per-head win + per-head-LR-inert caveat to the canonical docs** (done in this closeout: NORTH_STAR /
      RESULTS_BOARD / SYSTEM_REFERENCE / CANONICAL_VERSIONS cross-refs added) — full board promotion still gated on CA/TX + parity test.

**Perf / tooling (from PLAN §6)**
- [ ] **bf16/A40 fp32-attn island** (`_STANAttention`) — DEFERRED A40-bf16 mitigation; validate on the A40 (the grad-NaN only reproduces on Ampere at CA/TX C 6.5–8.5k), NOT H100. Driver `../closing_data/run_bf16_island.sh`.
- [ ] **MED cleanup** — unify `_run_category`/`_run_next` + `category_cv`/`next_cv`; one `build_autocast_ctx`; the `helpers` warmup-builder extraction (log.md:406).
- [ ] **Single-task fan-out** (`--run-id`/`--per-fold-seed` for `next_cv`/`category_cv`) — infra is generic; wire it.
- [ ] **n_regions pass-through** so a fan-out process doesn't transiently build all 5 folds for the region-count scan.
- [ ] **Unit tests for the new flags** — `MTL_ONECYCLE_PER_HEAD_LR` (per-group max_lr) + `--adam-beta2` are functionally verified but not pinned by CI (added in this closeout).

**Decided / declined (NOT open)**
- ✗ **P7 compile-once-across-folds** — MEASURED + DECLINED (commit `ac30b6d9`; log.md §P7): the in-process inductor cache
  already amortizes the per-fold compile (folds 2–5 are cache-hits, ~1s ep-1, not the 37s fold-1 compile), so P7's gain is
  ~0% and not worth the hot-path/log_T-buffer risk. Adversarial re-eval: "decline stands at scale."

**Catalog (standby, user-paused)**
- [ ] **QUALITY_IMPROVEMENTS.md** 27-lever catalog (besides A1 above) — breaking/quality-change improvements, on standby.

## 5 · Artifacts (this study)
Docs: `PLAN.md`, `DECOMPOSITION_PLAN.md`, `AUDIT_FINDINGS.md`, `BATCH_SIZE_SWEEP.md`, `RESULTS_SUMMARY.md`,
`QUALITY_IMPROVEMENTS.md`, `log.md`, this `CLOSURE.md`. Code: `MTL_ONECYCLE_PER_HEAD_LR`, `--adam-beta2`,
`MTL_SKIP_INERT_LOGT`/auto-fp32 defaults, the profiler + fan-out tooling, the mtl_cv decomposition. PR **#56**.
Results saved to `../closing_data/perhead_lr_n20.md`.
