# Follow-up audit findings (PR #56 feedback round)

Synthesis of the user's 6 follow-ups. Raw per-agent reports: [`AUDIT_REPORTS_RAW.md`](AUDIT_REPORTS_RAW.md)
(workflow `wf_5f410c3d-ef8`, 9 agents). Code changes are on `study/train-perf-multifold`.

## #6 · Timing — AL champion-G (5 folds × 50 ep, fp32, compiled, A40)
| run | wall | note |
|---|---|---|
| **baseline** (pre-perf) | **654 s** | sequential |
| **optimized** (P1+P3+P6) | **634 s** | sequential; ~3% (AL is small-model-bound) |
| **optimized + 5-fold parallel** (one A40) | **695 s** | board-grade (cat 63.47/reg 69.75); **slightly SLOWER** |

**Key finding (honest):** on a SINGLE GPU the fan-out does **not** speed up wall-clock. 5-way concurrency
saturates the GPU (99% util, up from ~34% solo) but pays 5× startup/compile + 5× the redundant per-process
n_regions scan (each process transiently builds all 5 folds) + contention. The fan-out's value is **multi-GPU /
multi-machine scaling** (fold-per-GPU → ~5×), **resumability** (re-run one fold after a crash), and **subset
granularity** (`--only-folds`), NOT single-GPU speed. Fixing the n_regions-scan-builds-all-5-folds inefficiency
(pass the region count explicitly) would cut the per-process overhead — see SLIMMING/future-work below.

## #2 · High per-fold variance / fold-5 outlier — DIAGNOSED (report it, don't refit)
Reproduced the frozen seed-0 split (96 326 rows, 1 101 users, 1 052 regions — matches the board). Fold sizes +
`next_category` class proportions are **identical** across folds (stratification works), so it is NOT a size or
class-imbalance artifact. Fold 5 drew a **harder user cohort** (the held-out unit is the *user*; only ~220/fold):
- **reg deficit ← self-transition (stay) rate**: fold-5 val users self-loop 29.6% vs 32.9% global (lowest of all
  folds). Across folds **stay-rate vs reg@10 r = +0.907**. (+ most cold/OOD regions 46, least-popular targets.)
- **cat deficit ← per-user category concentration** (a macro-F1 effect): fold-5 users are the most concentrated
  (entropy 1.380 vs 1.40–1.42); across folds **entropy vs cat-F1 r = +0.973**.
- Joint: stay-rate predicts both heads (r +0.907 reg / +0.800 cat); fold-5 best-epochs are earliest (genuinely
  lower ceiling, not under-training).

**Verdict:** expected, structural CV variance from user-grouped folds — deterministic for seed-0, not a bug.
**Report as-is; the n=20 multi-seed grid ({0,1,7,100}×5) re-draws the user partition 4× and tightens the CI.** A
composite-key stratification (bucket users by stay-rate × modal category before SGKF) would shrink fold variance
but **changes the frozen split → future-works robustness ablation only, never a silent swap.**

## #4 · next_getnext_hard / next_stan_flow_dualtower vs faithful STAN — NO BUG
Audited `NextHeadSTAN` + `next_stan_flow` + `next_stan_flow_dualtower` against `research/baselines/stan/model.py`.
**Verdict: the champion reg path has no correctness defect.** Every divergence from faithful STAN is an
**intentional deviation** because ours is a STAN-*inspired head over a learned Check2HGI substrate*, not raw-input
STAN (the substrate already encodes per-check-in ST context; Δt/Δd aren't in `next_region.parquet`). The repo's own
`docs/baselines/next_region/stan.md` already declares the in-house head "the substrate-as-input version of STAN".
Two optional hardenings (both bit-identical on today's left-packed data → need an AL A/B to confirm, deferred):
- **U1 (recommended hardening):** the last-valid-step pooling uses count-based `last_idx = num_valid-1`, correct
  for left-packed sequences but silently wrong under interior/right pads. Replace with the explicit last-True
  position (`S-1 - valid[:,::-1].argmax`, as in `next_region.py:137-141`) → alignment-robust. Bit-identical now.
- **U2 (cosmetic):** fully-padded rows have a softmax-index/readout-index mismatch — unreachable on real data.

## #1 · P1 compile-drift — GATED (done); bit-exact-repro claim CORRECTED in Phase 5c
`MTL_STAN_LEGACY_MASK=1` restores the guarded `.any()` masking at all 3 STAN sites (graph breaks 2→10).
Default off = fast path; **eager is bit-exact either way**.
> ⚠ **CORRECTION (Phase 5c):** the mask does NOT actually give bit-exact `--compile` reproduction. With a FRESH
> inductor cache, mask on==off (both AL 63.18/69.73); the prior "frozen-cell" 63.44/69.82 was a PERSISTENT-cache
> compile session, not the mask. The compiled number is governed by the cache/compile session, which the mask
> gate doesn't control; `mask=1` only restores the SLOWER graph-break path. **Decision: leave it OFF (perf); eager
> is the deterministic ground truth.** `run_al_baseline.sh` no longer pins it (see log.md §Phase 5c).

## #3 · bf16/A40 fix — fp32 attention island (done; large-state validation deferred)

> ⚠ **CORRECTION (advisor 2026-06-27):** NOT moot/closed-by-fp32. The board closes the NaN TWO ways — A40 true-fp32 (TX) AND H100-native-bf16 (CA, the §1 bf16 headline; Hopper has no Ampere grad-NaN). The island is the **only surgical A40-bf16 path** → DEFERRED, validate ON THE A40 (not H100; the NaN only reproduces on Ampere) when an A40 large-state bf16 run is wanted. Defer OK (root cause is a hypothesis; bf16 ~0 wall-clock on H100), but 'moot' was wrong.
`MTL_STAN_FP32_ATTN=1` runs the masked-softmax attention in fp32 even under bf16/fp16 autocast (the documented
NaN mechanism). **No-op under true fp32** (gate = `is_autocast_enabled(device)`) → board path byte-identical by
construction. AL bf16 smoke: runs **clean both ways** (0 skips under `MTL_STRICT=1`); island shifts bf16 numerics
slightly (fp32-precision recovery). ⚠ AL (C=1109) can't reproduce the NaN; the NaN states are CA/TX (C=6.5–8.5k),
whose bf16 overlap-MTL is H100-only / multi-hour here. **To validate on the real failure:** run the TX bf16 cell
(`scripts/closing_data/a40_task2_tx_mtl_bf16.sh`) with `MTL_STAN_FP32_ATTN=1` and assert 0 non-finite skips +
reg ≈ the clean fp32 67.02 (vs the void bf16 −2.37) — on the H100 or a long A40 run.

## Adversarial review (workflow `wf_2360804d-0ea`, 4 lanes) — verdict + actions
Independent skeptic review of what the eager AL parity can't catch (leaks, removed invariants, compile-drift):
- **leak-and-folds: SAFE** — `_classify_pois` verbatim + leak guard intact (val-exclusive POI never enters train);
  `n_regions` precompute provably equals the all-fold scan; multi-fold fan-out has **no val→train leak** (membership
  is fixed by base-seed SGKF before any reseed; per-fold log_T keyed by real id + base seed). 1 RISK fixed below.
- **comment-trim-invariants: SAFE** — token-strip confirms comment/docstring-only; every leak guard + env contract survived.
- **extraction-byte-identity: byte-identical in eager** (incl. the KD `gate!="none"` path AL doesn't exercise) — but
  flags the P1 compile-drift (below).
- **gates-and-bf16:** the gates/bf16-island/profiler are true no-ops when off; flags the P1 compile-drift as the one
  non-byte-identical change.

**Actions taken:**
- ✅ **full_summary.json fan-out guard** — under `--run-id` it's stamped `_fanout` (subset warning → use
  `aggregate_folds.py`) so a partial can't be cited as the full run. + a "NOT for frozen-cell reproduction" caveat in
  `run_folds_fanout.sh` (the fan-out's `--per-fold-seed` RNG stream differs from legacy sequential).
- ✅ **P1 compile-drift — keep P1 on; `MTL_STAN_LEGACY_MASK` UNPINNED (revised Phase 5c).** P1 stays default-on
  (perf). The mask gate works (graph breaks 2→10) but Phase-5c proved it does NOT give bit-exact compiled
  reproduction (fresh-cache mask on==off; the compiled number is cache-session-governed). So pinning it only
  costs the slower graph-break path for no reproducibility gain → `run_al_baseline.sh` no longer pins it, and
  CLAUDE.md is corrected to say "leave it off; eager is ground truth." See log.md §Phase 5c.

## #5 · Per-file train-flow slimming — grinding the A/B-gated extractions with a parity harness

Built a **fast metric-parity harness** ([`parity_check.sh`](parity_check.sh)): champion MTL on AL, 2 folds × 8 ep,
EAGER fp32 (deterministic, ~55s/run). Captures the per-fold VAL metric CSVs **and** a selection digest
(`primary_epoch` + `primary_task_metrics`, timing stripped). A behavior-preserving refactor must keep both
**byte-identical** vs `golden`. Verified reproducible (golden == golden_b). This gates every hot-numeric extraction
(the scored val/train metric path AND the checkpoint-selection path).

**Applied + VALIDATED this round (each parity byte-identical + unit-tested where pure):**
- `metrics.py` + `mtl_cv.py` + `mtl_eval.py`: extract `_streamed_cls_metrics()` — the C>256 hand-rolled metric
  reconstruction shared by the S1 streaming train-metric and the S2 chunked val-metric (dedup ~22 lines across 2
  files). **A/B-gated scored path** → parity golden==eval_dedup. Unit test vs the full-logit computation.
- `mtl_eval.py`: extract `_decide_chunk_val()` (the S2 chunk gate). Parity golden==eval_chunk.
- `mtl_cv.py`: extract `_compute_joint_selectors()` — the 5 joint scalars + selector dispatch out of
  `train_model`'s epoch loop (−55 lines inline). **Selection-path A/B-gated** → parity golden==selectors
  (incl. the selection digest). Unit test (scalars + dispatch + non-region fallback).
- `helpers.py`: `_overwrite_base_lr()` (3 identical scheduler blocks → 1). `experiment.py`: `step()` 38→13 lines
  via `_emit_adapter_fold_end()` + `_save_fold_partial_safe()`. Both test-covered; step() parity golden==exp_step.
- (earlier) `mtl_eval` redundant import + `mtl_creterion`→`mtl_criterion`; `experiment` redundant import + dead
  branch; `helpers` dead `DataLoader` import + merged `typing`; `dataset.py` comment; engine-list dedup.

> Note: line counts are ~flat because each extraction adds a helper ≈ the code it removes — the win is
> **structural** (leaner hot functions, named+tested helpers, dedup), not raw lines. The only large *line*
> reducer is trimming the ~45 dated audit-codename narration comments; that is behaviorally trivial (parity
> passes — comments don't execute) but it strips the repo's in-code institutional memory, so I left it as a
> deliberate, reviewable pass rather than auto-stripping (see "remaining" below).

**Roadmap — SAFE-now (no A/B; apply incrementally, run the test suite after each):**
- `experiment.py`: extract `_emit_adapter_fold_end` (197-205) + `_save_fold_partial_safe` (210-222) → `step()` 38→~15 lines.
- ✅ `helpers.py` `_overwrite_base_lr`; ✅ `experiment.py` `step()` decomp; ✅ `mtl_eval.py` chunk-decision helper.
- `folds.py`: §1/2/3/6/7/8/9/10 (no tensor/split/RNG change); keep `rebuild_dataloaders`/`load_folds`/shuffle untouched.
- `storage.py`: SAFE plot/diagnostic extractions; `train.py`: dead imports + guard/loss-calib extraction.
- Across all: trim dated audit-codename narration to its load-bearing invariant (preserve env-var contracts + leak guards).

**A/B-gated extractions — ALL DONE (gated by `parity_check.sh`):**
- ✅ `train.py` **runner merge** (`_run_category`/`_run_next` → `_run_single_task`) — single-task config (`run_stl`):
  golden_stl == runner_merge.
- ✅ `train.py`/`folds.py` **n_regions pass-through** (precomputed on `_LazyFoldMapping.n_regions`; skips the all-fold
  scan → also fixes the #6 fan-out per-process overhead) — golden == nreg (resolves 1109).
- ✅ `mtl_cv.py` **KD block** (`_log_t_kd_loss`) — KD-on variant (`--log-t-kd-weight 0.2`): kd_golden == kd_check AND
  no-KD golden == nokd_check.
- ✅ `mtl_cv.py` joint-selectors (selection digest) + S1 streaming-metric; ✅ `mtl_eval.py` streamed-metric + chunk.
- ✅ `folds.py` **`_classify_pois`** (legacy-MTL user-isolation partition) — pure verbatim extraction, gated by a
  unit test (classification + leak guard + order + 200-POI inline-equivalence; the legacy path isn't in the MTL harness).

**Narration comment-trim — DONE (−77 lines, all 7 files)** via the verified applier (`apply_comment_trims.py`,
proves no non-string code changed) + parity (MTL & STL byte-identical). Kept every invariant; cut the dated codenames.

**Bigger train_model / fold-builder decompositions — DONE (multi-seed gated, advisor SAFE per phase).**
See [`log.md`](log.md) §Phase 1–4. Each extraction gated byte-identical at AL seeds 0+1 (champion no-op) and,
where a path the champion can't exercise was touched, by a focused unit test:
- **Phase 1** — `_flatten_encoder` (module fn); `folds.py` `_resolve_task_input`, `_load_and_validate_check2hgi_data`, `_classify_pois`.
- **Phase 2** — `train_with_cross_validation` setup: `_build_mtl_optimizer`, `_build_scheduler`, `_build_task_criteria`, `_apply_stream_freezes`.
- **Phase 3** — batch-loop loss declutter: `_log_c_kd_loss` + `_cat_kd_loss` (mirror `_log_t_kd_loss`; −~85 lines; KD-on parity + unit test vs inline ref).
- **Phase 4a** — `_resolve_per_fold_priors` (the ~205-line per-fold log_T/log_C leak-guard block out of the fold loop; verbatim).
- **Phase 4b** — `MTL_SKIP_INERT_LOGT=1` opt-in: skip the per-fold log_T load when provably inert (the champion) → no log_T files needed; byte-identical, default-off.
- **Phase 5a** — `_optimizer_micro_step` (the should_step body) — DONE on user request (commit 29b43c2f). The
  under-gating concern was resolved by building branch-coverage parity variants: `--gradient-accumulation-steps 2`
  (partial-group rescale) + `--alternating-optimizer-step` (alt-inactive zero), both byte-identical, plus 6
  deterministic unit tests. Advisor SAFE.
- **Phase 5b** — `_run_validation_epoch` (the per-epoch validation→history block) — DONE on user request (commit
  25f6476a). Champion s0/s1 byte-identical + 121 integration tests; advisor SAFE.
- **Phase 5c** — vs-main validation (`golden==main_golden` byte-identical, eager) + the MTL_STAN_LEGACY_MASK
  correction (the mask gives NO bit-exact compiled repro; unpinned). See log.md §Phase 5c.

> NOTE (2026-06-26): the two items above were originally listed as "deliberately declined" (interface width +
> under-gating); the user asked for them and both were extracted with the branch-coverage gating that closed the
> under-gating concern. The decline is REVERSED.

**STILL OPEN (lower value, documented):** the `helpers` warmup-builder extraction (`setup_scheduler` inlines two
builders); the full 4-file `build_autocast_ctx` unify (only the 2-file *eval* ctx + the `_ood_from_streamed` dedup
are done — commit 52e387df); `category_cv`/`next_cv` `run_cv` merge. All gate-able with the same `parity_check.sh`.
