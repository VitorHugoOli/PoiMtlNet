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

### #2/#4/#5 — audit workflow `wf_5f410c3d-ef8` running (fold variance + STAN-head correctness + 7-file slimming).
### #6 — timing run `bg1guitbt` (5-parallel fan-out wall) in flight. Sequential: baseline 654s, optimized 634s.
