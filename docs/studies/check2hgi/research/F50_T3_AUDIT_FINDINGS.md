# F50 T3 — Critical-Review Audit Findings (2026-04-29)

**Source:** independent research-engineering agent commissioned to find OTHER bugs similar in nature to the F1-vs-metric selector mismatch (`F50_T3_TRAINING_DYNAMICS_DIAGNOSTICS.md` §5.5 / `MTL_FLAWS_AND_FIXES.md` §2.10).

**Agent prompt:** "Hunt for bugs of the form 'report metric X at the epoch chosen by metric Y', per-fold aggregation glitches, cross-substrate inconsistencies, leakage in fold-derived artefacts, and selector-mismatch siblings."

**Output:** 8 critical issues (≥±1 pp impact), 7 subtle issues, 5 style issues.

---

## Critical issues (CHANGE PAPER NUMBERS)

### C1 — STL ablation has the SAME selector bug (mirrored)  ✅ FIXED 2026-04-29 (forward-only)
- **File**: `scripts/p1_region_head_ablation.py:427-460`
- **Mechanism**: `_train_single_task` selects by **top10_acc** then writes F1/MRR/Acc@1 at THAT epoch. Mirror of the MTL F1-bug, opposite direction.
- **Impact**: every MTL-vs-STL paired comparison is ~3-4 pp biased AGAINST MTL because (a) MTL's reported top10 was at F1-best epoch, (b) STL's reported F1 is at top10-best epoch. Compounds the metric-definition mismatch.
- **Fix landed**:
  - `_new_per_metric_tracker` / `_update_per_metric_best` track each canonical metric's best epoch independently.
  - `_train_single_task` now emits `per_metric_best` alongside the legacy top10-best snapshot (backward-compatible primary fields).
  - Aggregator emits `aggregate_per_metric_best` with `<reported>_at_<selector>_best_{mean,std}` for every metric pair (all 25 combos).
  - 5 new synthetic tests pin the contract (`test_p1_per_metric_tracker.py`).
- **Posthoc note**: per-epoch metrics aren't logged to disk in the STL pipeline (only kept in memory), so OLD STL runs cannot be retroactively rescored. Future STL P1 invocations will produce `per_metric_best` automatically.

### C2 — `TaskConfig.primary_metric = ACCURACY` is dead code  ✅ FIXED 2026-04-29
- **File**: `src/tasks/presets.py:100`, `src/tracking/experiment.py:79-86`, `scripts/train.py:119-140 + 181-200`
- **Mechanism**: `TaskConfig.primary_metric` declared per-task (e.g. `next_region: ACCURACY`) but **never read**. `MLHistory` constructed without `monitor=` kwarg → every `BestModelTracker` defaults to F1.
- **Impact**: design intent was for region task to use Acc@1-best. F1 macro on 1109/4702-class is noisy; F1-best drifts to ep 7-13 while Acc@1-best/Acc@10-best/MRR-best ≈ ep 3-6. **This is the root cause of the F1-vs-top10 mismatch.**
- **Fix landed (commit pending)**:
  - `src/tracking/fold.py`: `FoldHistory.__init__` accepts `task_monitors: Mapping[str, str]` per-task override.
  - `src/tracking/experiment.py`: `MLHistory.__init__` accepts `task_monitors` and propagates to every fold.
  - `scripts/train.py`: both `_run_mtl` (legacy) and `_run_mtl_check2hgi` build `task_monitors` from `task_set.task_*.primary_metric.value` and pass to `MLHistory`.
  - `src/training/runners/mtl_cv.py:643-644`: `task_*_improved` checks now read each task's monitor key (was hardcoded F1; mismatched the tracker post-C2).
  - Tests: 3 new in `tests/test_tracking/test_ml_history.py` pin the contract; 162 tracking+training + 36 integration tests pass.
- **Forward semantics**: future MTL runs on CHECK2HGI_NEXT_REGION will track Acc@1-best for `next_region`. Past runs reported top10 at F1-best; future runs report top10 at Acc@1-best. **Not directly comparable.** Use the `per_metric_best` dict in `diagnostic_best_epochs[task]` (added 2026-04-29) for cross-run comparison on a fixed metric.

### C3 — Cross-fold callback state leak  ✅ FIXED 2026-04-29
- **File**: `src/training/callbacks.py:96-208`, `src/training/runners/mtl_cv.py:272`
- **Mechanism**: `EarlyStopping.stop_training` and `ModelCheckpoint.best` persist across folds (no `on_train_begin` reset). If EarlyStopping fires in fold 1, folds 2-5 stop at ep 1 immediately.
- **Impact**: silently destroys folds 2-5 if EarlyStopping is used. **Currently DORMANT** in our runs (EarlyStopping not used; `--no-checkpoints` set) but landmine for future.
- **Fix landed**: `EarlyStopping.on_train_begin` resets `stop_training/wait/best/best_epoch`; `ModelCheckpoint.on_train_begin` resets `self.best`. Runners already invoke `cb.on_train_begin` per fold (`mtl_cv.py:272`). Tests: 2 new in `test_callbacks.py`; 24/24 pass.

### C4 ⭐ — `α · log_T` graph prior built from FULL dataset → val→train leakage
- **File**: `scripts/compute_region_transition.py:61-117`, consumed in all `next_getnext_hard*` heads
- **Mechanism**: `build_transition_matrix(state)` reads ALL rows of `sequences_next.parquet`, counts every `(last_region, target_region)` pair, normalises and saves once. Every fold's reg head loads the **same** log_T — including transition counts derived from THAT FOLD'S VAL DATA.
- **Impact**: small per-row inflation, compounds. May explain a fraction of the F50 D6 fold-1 spike to top10=77.93. Affects **every** `next_getnext_hard` / `next_getnext_hard_hsm` / `next_tgstan` / `next_stahyper` / `next_getnext` result. Conservative estimate: 0.5-2 pp inflation on val top10 / MRR.
- **Strongest non-obvious finding in the audit.**
- **Fix**: rebuild log_T per fold from train rows only. Test: re-run FL H3-alt with per-fold log_T; expect 0.5-2 pp drop on val top10.

### C5 — Macro-F1 averaging differs by class cardinality  ✅ FIXED 2026-04-29
- **File**: `src/tracking/metrics.py:64-92` (hand-rolled, num_classes>256) vs `:243-245` (torchmetrics, ≤256)
- **Mechanism (refined via empirical replication)**: torchmetrics with `zero_division=0` does NOT divide by `num_classes` as the original audit claimed; it divides by the count of classes with EITHER support OR predictions. Hand-rolled used `support > 0` only — so FP-only classes (predicted but never seen as targets) were silently dropped from the average. On `next_region` (4702 classes; model often predicts ~2000 distinct classes; only ~1000-1500 with val support), this inflated macro-F1 by 1.5-3×.
- **Impact**: `region.f1` was ~1.5-3× higher than torchmetrics-equivalent. acc_macro had the same flavor of bias.
- **Fix landed**: `_handrolled_cls_metrics` now uses `relevant = (support > 0) | (predicted > 0)` for both `acc_macro` and `f1_macro` denominators. Verified equivalence with torchmetrics on FP-only edge case + high-cardinality random predictions; 2 new regression tests pin the contract; 498 + 36 integration tests pass.
- **Forward semantics**: future MTL/STL `region.f1` will be 1.5-3× lower than past reports. Cross-run comparison must use newly aggregated runs.

### C6 — σ convention mismatch (sample n-1 vs population n)  ✅ FIXED 2026-04-29
- **File**: `src/tracking/storage.py:138` (`statistics.stdev`, n-1) vs `scripts/p1_region_head_ablation.py:560,621` (`np.std`, n)
- **Mechanism**: MTL aggregates use sample std (n-1). STL ablation uses population std (n). Ratio for n=5: √(5/4) ≈ 1.118 → STL stds ~12% smaller.
- **Impact**: any "within 1σ" or "MTL ± σ overlaps STL ± σ" claim is biased. AZ/AL paired-test narrative slightly wrong.
- **Fix landed**: both p1_region_head_ablation.py call sites now use `np.std(vals, ddof=1)` to match MTL-side `statistics.stdev`. n=1 edge case returns 0.0 (np.std with ddof=1 returns NaN). Future P1 STL aggregates will be ~12% larger than past reports — STL/MTL paired comparisons should use newly aggregated runs.

### C7 — Three "best" definitions in same `full_summary.json`
- **File**: `src/tracking/storage.py:143-201, 447-532`
- **Mechanism**: `_collect_performance` aggregates at **joint-best** epoch; `_collect_task_best_performance` separately aggregates at **per-task F1-best** epoch as `diagnostic_task_best`; `fold_info.json` writes `primary_checkpoint.task_metrics[task].accuracy/f1` at joint-best AND `diagnostic_best_epochs[task].metrics` with everything at F1-best.
- **Live observation** (`_0045/`): primary_checkpoint epoch=47 has next_region.accuracy=0.478 while diagnostic_best_epochs has accuracy=0.507 (epoch 6). 3-17 pp gap presented as same model+fold.
- **Fix**: stamp `aggregation_basis: "joint_best" | "per_task_f1_best" | "per_metric_best"` next to every aggregate. (Partially addressed by 2026-04-29 fix in `storage.py` adding `per_metric_best` sub-dict.)

### C8 — Fold determinism not verified across substrates  ✅ FIXED 2026-04-29
- **File**: `src/data/folds.py:717-723, 400-429`
- **Mechanism**: `StratifiedGroupKFold(shuffle=True, random_state=42)` is deterministic *given input row order*. Parquet row order is preserved if file is written deterministically — but if regenerated between runs, can differ. `--no-folds-cache` (default in every runpod script) skips the freeze.
- **Impact**: paired tests with mis-aligned folds are silently invalid (treats non-paired as paired → larger Wilcoxon p-values, hides effects rather than inflates).
- **Fix landed**:
  - `src/data/fold_digest.py`: pure helpers `compute_fold_set_digest(manifests)` (SHA-256 over normalised train/val userid sets) and `digest_compatible(a, b)`.
  - `FoldCreator.save_split_manifests` now also emits `fold_set_digest.json` (top-level) and stamps the digest into every per-fold manifest.
  - 8 unit tests in `tests/test_data/test_fold_digest.py` pin the contract: identical partitions → equal digest; swap/seed/userid changes → different digest; auxiliary fields don't leak.
- **Usage by future paired-test code**: load both runs' `fold_set_digest.json`; refuse to compute Wilcoxon if `digest_compatible(a, b)` is False.

---

## Subtle issues (≤1 pp impact)

- **S1**: `top_k_accuracy` vs `_rank_of_target` tie handling differs (~0.1-0.5 pp under fp16 autocast)
- **S2**: 256-class boundary for hand-rolled F1 is hard cutoff (will silently jump conventions if any task moves above 256)
- **S3**: `setup_per_head_optimizer` puts NashMTL learnable weights in reg group; `setup_optimizer` uses model-LR default — different effective NashMTL update rates
- **S4**: `compute_class_weights` defaults absent classes to weight=1.0 (dormant unless `--use-class-weights` set)
- **S5**: `_save_reports` partial-save and full-save logic duplicated (correctness-equivalent today; drift risk)
- **S6**: posthoc tool returns `value_at_epoch=NaN` silently dropping folds; aggregator doesn't filter NaN
- **S7**: `_compute_gradient_cosine` runs on `batch_idx==0` only — diagnostic CSV under-reports cosine variability

## Style issues

- **Sty1**: `f50_delta_m.py:250` dead code (`wmrr` computed but never used, suspicious 2× factor)
- **Sty2**: `posthoc_best_epoch.py` — `f1_at_metric_best` computed but not in summary JSON
- **Sty3**: hardcoded `/Volumes/Vitor's SSD/...` paths in 2 analysis scripts
- **Sty4**: posthoc tool returns std=0 for n=1 instead of NaN
- **Sty5**: `extract_alpha.py` assumes `save_best_only=True` checkpoint convention; warns aren't emitted

---

## Recommended fix priority

| # | Issue | Effort | Impact | Recommendation |
|---|---|---|---|---|
| **C4** | Graph prior val leakage | ~1h dev + verification re-run | 0.5-2 pp on every `next_getnext*` result | **DO NOW** |
| **C2** | `primary_metric` dead code | ~1h dev | Root cause of selector bug; prevents recurrence | DO BEFORE NEXT RUN |
| **C1** | STL selector mirror bug | ~30 min posthoc tool extension | 3-4 pp bias on every MTL-vs-STL comparison | DO POSTHOC |
| **C5** | macro-F1 cardinality | ~30 min dev + re-aggregation | 5-15 pp on region.f1 cross-state | DO BEFORE PAPER |
| **C7** | Multiple best-definitions | ~30 min dev | Documentation / `aggregation_basis` stamp | DO ALONG WITH C2 |
| **C3** | Callback state leak | ~30 min dev | Dormant; landmine for future | DO BEFORE ANYONE USES EarlyStopping |
| **C6** | σ convention | ~10 min dev (one decision) | 12% scale on σ | DO BEFORE PAPER |
| **C8** | Fold determinism | ~1h dev | Silent paired-test invalidity | DO BEFORE FUTURE CROSS-RUN PAIRED TESTS |
| S1-S7, Sty1-5 | Various | varies | low | clean up opportunistically |

