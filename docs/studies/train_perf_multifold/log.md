# train_perf_multifold — running log

> Study branch: `study/train-perf-multifold` (off `main`). Machine: **a40-wk** (single A40 46 GB, fp32 board protocol).
> Goal (user, 2026-06-26): (1) reproduce AL champion-G + STL ceilings and confirm vs `closing_data/RESULTS_BOARD.md`;
> (2) audit the whole train pipeline (inputs→loaders→model→hparams→train→eval→metrics) for **quality-neutral**
> speedups; (3) make multi-fold fan-out feasible — N parallel single-fold processes writing into ONE shared
> execution rundir, with a `--only-folds 2,3` CLI + clean aggregation; (4) opportunistic cleanup (dead comments,
> duplication, best practices). Also: investigate the A40-Ampere bf16 backward-NaN.

See `PLAN.md` for the design + findings. This file is the chronological work log.

---

## 2026-06-26

### Session start — orientation
- Branched `study/train-perf-multifold` off `main`. GPU free (A40, 45 GB).
- Read `RESULTS_BOARD.md` + `TX_A40_BF16_NAN.md` + `HANDOFF_A40.md`. Board protocol = seed 0 × 5 folds,
  engine `check2hgi_dk_ovl`, MIN_SEQ=10, **fp32 on A40** (`MTL_DISABLE_AMP=1`; Ampere bf16 backward grad-NaN).
- **AL targets** (RESULTS_BOARD §1 board/H100): STL cat 55.87 · MTL cat 63.56 (+7.69) · STL reg 69.99 ·
  MTL reg 69.81 (−0.18). **§1b A40 same-device champion-G** (the right comparand for this box):
  cat **63.2481 ±2.02** [63.34,64.63,64.73,64.19,59.34] · reg **69.6458 ±3.32** [71.85,68.81,73.17,70.74,63.66]
  (`docs/results/closing_data/a40/al_champG_a40_s0.json`; reg = full top10 = indist·(1−ood) at indist-best ep,
  cat = macro-F1 at f1-best ep, per-task diagnostic-best fold-mean; scorer `a40_score_matched.py`).
- **AL substrate present on disk** (`output/check2hgi_dk_ovl/alabama/`): embeddings/region/poi symlinked to v14
  `check2hgi_design_k_resln_mae_l0_1`; `input/{next,next_region}.parquet` + provenance (min_seq=10, stride=1,
  emit_tail=false ✓); per-fold log_T `region_transition_log_seed0_fold{1..5}.pt` (mtime 21:03 ≥ next_region 21:01
  → **fresh, not stale**). Folds are **1-indexed** (fold1..fold5). → AL can run immediately, no rebuild.
- Rundir layout: `results/<engine>/<state>/<task>_lr..._bs..._ep..._<ts>_<pid>/` with
  `summary/{full_summary.json, summary_next_*_metrics.csv}`, `folds/fold{N}_info.json` (diagnostic-best epochs —
  the field to trust, per memory), `metrics/fold{N}_next_*_{train,val}.csv`, `diagnostics/`, `model/`, `plots/`.

### Launched understanding workflow `wf_9104d098-482` (7 parallel Opus agents)
Map: AL CLI runbook · train.py entrypoint/rundir · fold-execution arch (seam for fan-out) · perf-audit data path ·
perf-audit model+train-loop · bf16/A40 precision map · cleanup survey. Results → PLAN.md.

### Understanding workflow returned (7 reports, 781k tok, 162 tool-uses) → PLAN.md written
Key findings folded into PLAN.md §3–§5. Headlines:
- **AL CLI** verified against the real board drivers (`board_h100_mtl.sh` fp32 arm + `board_h100_ceiling.sh`).
  ⚠ The board champion-G uses the **v14** log_T dir (`output/check2hgi_design_k_resln_mae_l0_1/$STATE`), not the
  overlap dir; I run the board driver verbatim (removes my own DOF; v14↔overlap log_T diff is sub-fold-std).
- **Perf**: both audit lanes converge on a byte-identical set — drop the data-dependent `.any()` graph-breaks in
  the STAN reg head, defer train-metric D2H to epoch-end, cache the per-epoch OOD label set, pin CPU-resident
  batches (CA/TX), compile-once-across-folds, gate diagnostics. Quality-RISKING (excluded): fused AdamW, SDPA,
  bf16-default, removing the AMP gate.
- **Multi-fold**: 3 blockers — (B1) rundir name embeds ts+pid; (B2) per-fold artifacts named by in-memory
  POSITION not real fold id → collisions; (B3) **no per-fold reseed** so `--only-fold k` ≠ fold-k of a sweep.
  Design = `--run-id` (shared rundir) + real-fold-id artifact naming + opt-in `--per-fold-seed` + new
  `aggregate_folds.py`. Default bare path stays byte-identical (board-safe).
- **bf16/A40**: no proven byte-identical fix; fp32 is the validated answer. Candidate = fp32 attention island in
  `_STANAttention` (A/B-only). Safe launcher hardening = auto-fp32 for large states on Ampere.
- **Cleanup**: ~0 commented-out code (clean); ~45 stale narration comments; 6 duplication clusters
  (the literal 21-engine allow-list dup'd in train.py + folds.py is #1).

### Baseline run launched (background, `bavja82cq`)
Driver `run_al_baseline.sh` (fixed: `/usr/bin/time` absent on this box → bash timing; robust new-rundir capture).
champion-G smoke (fold0/2ep) passed: 1109 regions, log_T fold1 resolved, 0 NaN under MTL_STRICT, ~12 batch/s.

### ✅ AL baseline RESULT — reproduces the board (Goal 1 DONE)
| metric | my A40 baseline | board §1b (A40) | board §1 (H100) | Δ vs §1b |
|---|---|---|---|---|
| MTL cat (champ-G) | **63.18** ±1.84 | 63.25 | 63.56 | −0.07 |
| MTL reg (champ-G) | **69.73** ±3.26 | 69.65 | 69.81 | +0.08 |
| STL cat ceiling | **55.73** ±1.83 | 55.87 | 55.87 | −0.14 |
| STL reg ceiling | **69.98** ±3.56 | 69.99 | 69.99 | −0.01 |
All within ≤0.14 pp (fold-std 1.8–3.6). **Δcat +7.45 (beats), Δreg −0.26 (matches)** = exact board story.
0 non-finite skips, late best-epochs. JSONs in `baseline_runs/`. **Wall (A40 fp32, compiled):**
champ-G **654s** (~11 min/5f), STL cat **98s**, STL reg **93s** — the reference for measuring perf gains.

### User steer (mid-flight): build an ephemeral profiler/audit tool in src/
Not in MLHistory (that's the persistent record) — a debug/monitoring tool that lives during execution like logs:
monitor perf/pace/quality + surface code pain-points. → `src/training/profiling.py` (RunProfiler), opt-in
(`--profile`/`MTL_PROFILE=1`), zero-overhead off. pynvml present (GPU-util sampling), dynamo counters usable
(compile/recompile audit). This operationalizes Goal-2 (audit pain points) as a reusable tool.

### Profiler validated + perf P1/P3/P6 applied + AL A/B
- Profiler flagged the real pain points on the unfixed champion path: `GPU-STARVED util 34% (p50 0%)` +
  `GRAPH BREAKS: 10` (the audit's P1). Zero-overhead when off.
- **P1** (3 STAN `.any()` graph-breaks removed): graph breaks **10 → 2**; masking **proven byte-identical** in
  eager (`tests/test_models/test_stan_mask_equivalence.py`, 3/3). Under `--compile` it shifts FP-reduction order
  (compile's own drift) → ≤0.3pp/fold, within fold-std, mean preserved.
- **P3** (cache per-epoch OOD label set on fold_history; byte-identical). **P6** (pin CPU-resident batches;
  inactive for GPU-resident AL, helps CA/TX; byte-identical).
- **AL perf A/B (full 5f)**: MTL cat 63.18→**63.44**, reg 69.73→**69.82** (both within fold-std, mean ↑); STL cat
  55.73→**55.73 bit-identical** (single-task path untouched → confirms isolation). champG wall 654→634s.
  **No quality loss.** Deferred P2a/P5 (non-bottlenecks on sensitive paths), P7 (only helps sequential).

### Multi-fold fan-out implemented
- CLI `--only-folds 2,3` / `--run-id NAME` (shared rundir leaf, implies `--per-fold-seed`) / `--per-fold-seed`
  (reseed seed+fold_id before each fold → fold-k order-independent).
- `MLHistory.run_id` + `fold_label()` (real-fold-id artifact naming → no collision); default path byte-identical.
- `train_with_cross_validation(per_fold_seed=…)` reseed-before-materialize loop. n_regions consistency automatic.
- `scripts/aggregate_folds.py` (per-fold val CSVs by real id → fold_aggregate.json + presence gate) +
  `scripts/run_folds_fanout.sh` (throttled per-fold processes → one rundir → aggregate).

### ✅ User steer: 5 concurrent folds into ONE rundir — NO CONFLICT (proven)
`run_folds_fanout.sh al_concurrent_test 0,1,2,3,4 5` — all 5 launched simultaneously (pids 2713734-38),
GPU 21.6 GB / 99% util (5-way concurrency *saturates* the GPU → cures the single-process starvation), 23.9 GB free.
- **failures=0**; all 5 folds present, named by **real fold id** (`fold1..fold5_info.json` + 10 val CSVs) — no collision.
- aggregate complete (n=5 both tasks): cat 64.15, reg 68.12 (8-epoch numbers, expected lower than 50-ep champion).
- **Byte-identity proof:** fold 1 run SOLO (no contention) vs fold 1 in the concurrent run → `next_region` and
  `next_category` val CSVs **byte-identical** (char-for-char). → `--per-fold-seed` makes fold-k a pure function of
  (seed, fold_id); concurrency/order doesn't affect numerics; the fan-out aggregate is fully reproducible.

### Cleanup (behavior-preserving)
- Dedup the literal 21-member MTL-check2hgi engine allow-list (was duplicated byte-for-byte in train.py +
  folds.py) → single source `configs/paths.MTL_CHECK2HGI_ALLOWED_ENGINES`, imported in both. py_compile + import OK.
- mtl_eval: removed redundant `_rank_of_target` re-import + fixed `mtl_creterion`→`mtl_criterion` typo (param,
  unused-in-body, passed positionally → safe). dataset.py: corrected the wrong "not imported by any code path"
  comment (it IS imported by `scripts/p1_poi_head_ablation.py`).

### Tests + commit + PR
- 905 pass in touched modules (+ new mask test). 3 pre-existing embedding-test failures confirmed unrelated
  (fail on clean HEAD with my work stashed). CLAUDE.md + studies/README updated for discoverability.
- Committed on `study/train-perf-multifold` → **PR #56** (off main, not merged).

### ✅ Capstone: full 50-epoch champion-G via 5-way concurrent fan-out
`run_folds_fanout.sh al_champG_fan50_s0 0,1,2,3,4 5` (compiled, per-fold inductor caches, fp32, 50 ep).
failures=0, all 5 folds present by real id. Canonical matched score + `aggregate_folds.py` (complete, n=5):
**cat 63.4717 ±1.83** [63.63,65.12,64.52,64.14,59.95] · **reg FULL top10 69.7454 ±3.13** [72.07,68.77,73.11,
70.57,64.21], late best-epochs. → the **full-scale fan-out is board-grade** (board §1b 63.25/69.65; sequential
perf A/B 63.44/69.82) — Δcat +7.7 (beats), Δreg −0.25 (matches). Evidence: `capstone_fanout50_runs/`.

### 🐞 Bug found + fixed by the capstone
`HistoryStorage._folder_name` **lowercases** the run_id (`al_champG…` → `al_champg…`), so the orchestrator's
case-sensitive rundir glob missed it (folds ran fine, auto-aggregate step skipped). Fixed: orchestrator globs
with the lowercased run_id; `--run-id` help notes the lowercasing. (Demonstrates the fan-out is robust — the
folds completed cleanly; only the post-hoc locate was case-buggy.)

## Status: COMPLETE. All 4 goals + the profiler tool + the concurrent-conflict test delivered. PR #56.

---

## 2026-06-26 — Follow-up round (user feedback on PR #56)

Six follow-ups. Tracking + outcomes:

### #1 — Gate the P1 compile-drift (DONE)
`MTL_STAN_LEGACY_MASK=1` restores the original guarded `.any()` masking at all 3 STAN sites (single-sourced flag
in `next_stan/head.py`, imported by the flow/dualtower heads) → **bit-exact `--compile` reproduction** of a pre-P1
frozen cell. Default off = fast vectorised path. (Eager is bit-exact either way.) Verified: flag default False,
env=1 propagates to all heads, mask-equivalence test still 3/3.

### #3 — bf16/A40 fix: fp32 attention island (CODE DONE; GPU smoke pending)
`MTL_STAN_FP32_ATTN=1` runs the masked-softmax attention (QK^T→+bias→mask→softmax→AV) in **fp32 even under bf16/
fp16 autocast** — targets the documented A40-Ampere bf16 backward-NaN mechanism (degenerate softmax in the anneal
tail at large C). **No-op under true fp32** (gate = `is_autocast_enabled(device)`, False under MTL_DISABLE_AMP=1)
→ board path byte-identical BY CONSTRUCTION. Tests (`test_stan_fp32_attn_island.py`): no-op without autocast +
finite & fp32-precision-recovering under bf16. ⚠ **Unvalidated on the actual NaN case** — AL (C=1109) doesn't NaN
in bf16; the NaN states are CA/TX (C=6.5-8.5k), whose bf16 overlap-MTL is H100-only / multi-hour on this A40
(env note). To validate on the real failure: run the TX bf16 cell (`a40_task2_tx_mtl_bf16.sh`) with
`MTL_STAN_FP32_ATTN=1` and check 0 non-finite skips vs the void −2.37 — on the H100 or a long A40 run.
Model+training suite 252 passed (no regression).

### #2/#4/#5/#6 — audit workflow `wf_5f410c3d-ef8` (9 agents, 651k tok) DONE → see [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md)
- **#6 timing**: baseline 654s · optimized 634s · **opt+5-parallel 695s (slightly SLOWER on one GPU)**. Fan-out
  doesn't speed a single GPU (5× startup + per-process n_regions scan + contention; GPU already 99%). Value =
  multi-GPU scaling + resumability + subset granularity. Board-grade aggregate (cat 63.47/reg 69.75).
- **#2 fold variance**: fold-5 = harder USER cohort (held-out unit is the user). reg deficit ← low self-transition
  rate (29.6% vs 32.9%; stay-rate vs reg@10 r=+0.907); cat deficit ← high per-user category concentration
  (entropy vs cat-F1 r=+0.973). Expected structural CV variance, NOT a bug → report it, n=20 multi-seed tightens.
- **#4 STAN audit**: NO correctness bug in the champion reg path; divergences from faithful STAN are intentional
  (head-over-substrate). Two optional bit-identical hardenings (U1 alignment-robust pooling, U2 cosmetic) deferred.
- **#3 bf16 smoke**: AL bf16 runs CLEAN both ways (0 skips, MTL_STRICT=1); island shifts numerics (fp32 recovery).
  Real NaN validation (TX bf16) deferred to H100. Code+tests done.
- **#5 slimming**: applied SAFE non-numeric wins (experiment.py redundant import + dead branch; helpers.py dead
  DataLoader import + merged typing). Full per-file SAFE-now vs A/B-gated roadmap in AUDIT_FINDINGS.md §5. Tests
  379 passed. Hot-numeric extractions deferred behind metric-parity A/B (frozen-§0.1 contract).

## Follow-up status: #1 ✅ #3 ✅(code+AL) #2 ✅(diagnosed) #4 ✅(no bug) #6 ✅(measured) #5 ✅(safe applied + roadmap)

### #5 grind — A/B-gated extractions with a fast parity harness
Built `parity_check.sh`: AL champion MTL, 2 folds × 8 ep, EAGER fp32 (~55s, deterministic). Captures per-fold VAL
CSVs **+ a selection digest** (primary_epoch + primary_task_metrics, timing stripped). Verified reproducible
(golden==golden_b). Gates each refactor byte-identically on BOTH the scored and selection paths.
Extractions DONE (5 files, each parity-clean + tested):
- `_streamed_cls_metrics` (metrics.py) — dedup S1 train-metric (mtl_cv) + S2 val-metric (mtl_eval), ~22 lines. **A/B**.
- `_decide_chunk_val` (mtl_eval) — the S2 chunk gate. **A/B**.
- `_compute_joint_selectors` (mtl_cv) — 5 joint scalars + selector dispatch, −55 inline. **selection-path A/B**.
- `_overwrite_base_lr` (helpers) — 3 scheduler blocks → 1. `step()` decomp (experiment) 38→13. SAFE + tests.
3 new unit tests (streamed-metric==full-logit, joint-selectors, fp32-island earlier). All suites green.
Line counts ~flat (helper ≈ removed code) — the win is structural; the big line-reducer (narration comment trim)
left as a reviewable pass (strips institutional memory). Remaining: train.py runner-merge (needs single-task
parity config) + n_regions pass-through + KD block (needs `--log-t-kd-weight 0.2` variant) + folds.py. See AUDIT_FINDINGS §5.

### Comment trim (−77 lines) + runner merge — answering "line counts ~flat" + "remaining A/B items"
- **Comment trim across all 7 train-flow files (−77 lines)**: a 7-agent pass proposed comment/docstring-only edits;
  applied through a verifier (`apply_comment_trims.py`) that PROVES no non-string code changed (tokenize-compare),
  then behaviorally confirmed (378 tests + parity byte-identical MTL+STL). Cut dated codenames (AUDIT-C2, F50/F48,
  C21/C7, HANDOFF_AUDIT X*, T2P.0/T4.0a, "Phase 4a", "bit us once…"); KEPT every invariant (env-var contracts,
  per-fold log_T leak guard, WD-peel-α/β, absent-class fill, fp16 tie-break / OOM budget, "API compat"). This is the
  genuine line reducer (the extractions were structural, ~flat).
- **Single-task parity config** added (`parity_check.sh run_stl`, --task next, 2f×8ep eager) for the runner-merge gate.
- **Runner merge** (train.py): `_run_category`/`_run_next` (~95% identical) → `_run_single_task` + 2 wrappers, −38 lines.
  Gated: golden_stl==runner_merge + golden==runner_merge_mtl byte-identical. **A/B-gated, done.**

A/B-gated extractions DONE (6): streamed-metric dedup, chunk-decision, joint-selectors, base-LR, step(), runner-merge
— each parity byte-identical + tested. **Remaining**: n_regions pass-through (also fixes #6 fan-out overhead — needs
plumbing n_regions through the lazy fold mapping), KD block (`--log-t-kd-weight 0.2` parity variant), folds._classify_pois.

### Remaining A/B-gated items — ALL DONE (3/3)
- **n_regions pass-through**: `_LazyFoldMapping.n_regions` precomputed (= per-fold train∪val max, since each K-fold's
  train∪val = all rows); train.py reads it, skips the all-fold scan (fixes the #6 fan-out overhead). golden==nreg (1109).
- **KD block** `_log_t_kd_loss`: extracted out of the per-batch loop (−73 lines). Gated by `--log-t-kd-weight 0.2`:
  kd_golden==kd_check AND golden==nokd_check (both byte-identical).
- **`_classify_pois`** (legacy-MTL user-isolation POI partition): pure verbatim extraction; unit test (classification
  + leak guard + order + 200-POI inline-equivalence). Champion parity unaffected.
Also: runner merge (`_run_single_task`, single-task parity config) earlier this batch.
**Full suite 608 passed.** 14 commits on PR #56. All A/B-gated extractions + the −77-line comment trim landed,
each parity/test byte-identical-verified.

### KD clarification + full-AL verification + last open items
- **KD/log_T**: `log_T` = per-fold region-transition (Markov-1) prior. The CHAMPION runs it **OFF**: v16 bundle +
  board driver both set `--log-t-kd-weight 0.0`; the `0.2` is the separate v12 code default (`_V12_LOG_T_KD_DEFAULT_W`),
  not the champion. So `_log_t_kd_loss` hits the no-op path on the champion (golden==nokd_check, byte-identical).
- **Full AL champion-G re-run (KD off, 5f×50ep, ALL changes applied)**: **cat 63.4421 / reg 69.8181** — IDENTICAL
  to the perf A/B (63.44/69.82), within fold-std of the board (§1b 63.25/69.65; §1 63.56/69.81). STL cat 55.7273,
  reg 70.00. **The entire slimming round (extractions + comment trim + KD extraction + gates) preserved the
  champion byte-for-byte.** champG wall 621s. (`final_verify_runs/`).
- **Last open items DONE**: `_ood_from_streamed` (OOD dedup, scored path) + `eval_autocast_ctx` (shared eval
  autocast, mtl_eval+mtl_validation) — parity byte-identical MTL+STL, unit-tested.
- **Adversarial review workflow** (`wf_2360804d-0ea`, 4 lanes): leak/fold-correctness, comment-trim invariants,
  extraction byte-identity (incl. the KD gate path AL doesn't exercise), gates/bf16/profiler no-op-when-off.
