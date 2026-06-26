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

## #1 · P1 compile-drift — GATED (done)
`MTL_STAN_LEGACY_MASK=1` restores the guarded `.any()` masking at all 3 STAN sites → bit-exact `--compile`
reproduction of a pre-P1 frozen cell. Default off = fast path; eager is bit-exact either way.

## #3 · bf16/A40 fix — fp32 attention island (done; large-state validation deferred)
`MTL_STAN_FP32_ATTN=1` runs the masked-softmax attention in fp32 even under bf16/fp16 autocast (the documented
NaN mechanism). **No-op under true fp32** (gate = `is_autocast_enabled(device)`) → board path byte-identical by
construction. AL bf16 smoke: runs **clean both ways** (0 skips under `MTL_STRICT=1`); island shifts bf16 numerics
slightly (fp32-precision recovery). ⚠ AL (C=1109) can't reproduce the NaN; the NaN states are CA/TX (C=6.5–8.5k),
whose bf16 overlap-MTL is H100-only / multi-hour here. **To validate on the real failure:** run the TX bf16 cell
(`scripts/closing_data/a40_task2_tx_mtl_bf16.sh`) with `MTL_STAN_FP32_ATTN=1` and assert 0 non-finite skips +
reg ≈ the clean fp32 67.02 (vs the void bf16 −2.37) — on the H100 or a long A40 run.

## #5 · Per-file train-flow slimming — SAFE wins applied; hot extractions = validated roadmap
The frozen-§0.1 contract forbids a sweeping unvalidated refactor of the hot files, so: **applied now** the
unambiguously-safe, non-numeric items; **deferred** every hot-numeric extraction behind a metric-parity A/B.

**Applied (this round, no A/B — verified by the test suite):**
- `experiment.py`: removed the redundant in-`step()` `import logging`; dropped the dead `hasattr(timer,...)` branch.
- `helpers.py`: removed the dead `DataLoader` import; merged the split `typing` imports.
- (earlier) `mtl_eval.py` redundant import + `mtl_creterion`→`mtl_criterion`; `dataset.py` comment; engine-list dedup.

**Roadmap — SAFE-now (no A/B; apply incrementally, run the test suite after each):**
- `experiment.py`: extract `_emit_adapter_fold_end` (197-205) + `_save_fold_partial_safe` (210-222) → `step()` 38→~15 lines.
- `helpers.py`: extract `_overwrite_base_lr(...)` for the 3× base-LR block (304-330).
- `mtl_eval.py`: comment trim (biggest single win), chunk-decision helper, shared `256` constant, narrowed `except`.
- `folds.py`: §1/2/3/6/7/8/9/10 (no tensor/split/RNG change); keep `rebuild_dataloaders`/`load_folds`/shuffle untouched.
- `storage.py`: SAFE plot/diagnostic extractions; `train.py`: dead imports + guard/loss-calib extraction.
- Across all: trim dated audit-codename narration to its load-bearing invariant (preserve env-var contracts + leak guards).

**Roadmap — A/B-gated (run one `--canon none` check2hgi_next_region cell, assert per-task diagnostic-best bit-identical):**
- `train.py`: runner merge (`_run_category`/`_run_next`), KD block, fold-subset slicing, **n_regions scan → pass-through** (also fixes the fan-out per-process overhead in #6).
- `mtl_cv.py`: KD/prior block, joint-selector math (1107-1187), streaming train-metric consolidation (keep `guard_finite_step` name — test-imported).
- `mtl_eval.py`: streamed-metric + OOD dedup into `metrics.py`; shared autocast ctx (`build_autocast_ctx`).
- `folds.py`: region-label + `_classify_pois` extraction (verify `fold_set_digest` equality).

**Realistic slim** (per the audit): mtl_cv ~15% SAFE-now / ~25–35% with A/B'd extractions; mtl_eval ~25–30%;
experiment ~clarity (step 38→15); helpers/train.py mostly narration. The largest lever is trimming the dated
narration comments to invariants — safe but high-volume; do it as a focused, reviewable pass.
