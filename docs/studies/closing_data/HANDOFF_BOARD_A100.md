# HANDOFF — board launch · **A100** (CUDA) · branch `study/board-a100`

> One-machine handoff. You are the **A100** lane. The shared governing rule, launch sequence, and the per-machine
> branch+PR process live in the **index**: [`HANDOFF_BOARD.md`](HANDOFF_BOARD.md) — read it once, then work only
> from this file. Recipe/guards ground-truth: [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md)
> + [`../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`](../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md);
> sequence + device rule: [`../EXECUTION_PLAN.md §12–§13`](../EXECUTION_PLAN.md); stats:
> [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md); matched scorer + baseline design:
> [`RUN_MATRIX.md §0, §2.5`](RUN_MATRIX.md); champion-G recipe: [`../../NORTH_STAR.md`](../../NORTH_STAR.md).

---

## 0 · SCOPE BOUNDARY (what THIS machine does — and what it must NOT touch)

**The A100 does exactly two pre-freeze cells, then (post-freeze) owns the heavy by-state partition.**

1. **Task 1 — A100-equivalence A/B, A100 HALF.** Run the **byte-identical** FL champion-G MTL command as the A40
   (seed 0, 5 folds, compiled+tf32). Record `cat macro-F1` + `reg top10_acc` to **4 dp**. **Confirm
   `|Δ| ≤ ±0.05 pp` on BOTH heads vs the A40 BEFORE any by-state parallelization across the two CUDA cards.**
2. **Task 2 — EARLY CA gated-overlap reg cell.** CA is the **largest** state (8501 regions) → the **most at-risk**
   for the δ_reg = 2 pp margin (the mechanism warns the reg gap may widen vs FL's −1.2 at more regions). Build
   the CA overlap engine, then the matched **B-A2 reg** pair — STL `next_stan_flow` reg ceiling on overlap **+**
   champion-G MTL reg on overlap — 1 seed × 5 folds; report **Δreg vs δ_reg = 2 pp**.

**Do NOT do (these belong to other machines):**
- **Do NOT run the TX early reg cell** — that is the **A40** (`study/board-a40`).
- **Do NOT build the light substrate-column baseline embeddings** (CTLE / POI2Vec / skip-gram / one-hot) — those
  are the **M2 Pro** (`study/board-m2pro`).
- **Do NOT run the matched-head baseline COMPARISON on a Mac**, and **do NOT** pull a state's baselines onto a Mac
  while its STL/MTL is here — every paired comparison stays on ONE device-class (the governing rule).
- **Do NOT** trigger the P2 freeze or launch the full P3 board — that is the orchestrator's call.
- **Do NOT** commit to `main`, do NOT merge any branch. Work only on `study/board-a100`.

> **A100 notes:** the A100's larger memory gives more headroom for CA (largest), but the same rules apply —
> **auto-fit, NEVER `MTL_DATASET_GPU=1`** (forces ~31 GB redundant copies → OOM even here). **The TORCHINDUCTOR
> cache is per-box** — the A100 has its OWN `TORCHINDUCTOR_CACHE_DIR`; the **first** compiled cell on the A100
> pays the one-time break-even warmup (~13 recompiles → 0 on reuse), independent of the A40's cache. Do not share
> a cache dir across boxes.

---

## 1 · PARALLELISM on the A100 (single GPU)

The A100 is **one GPU → GPU cells are strictly SERIAL** (Task 1's FL MTL, then Task 2's CA STL-reg, MTL-reg one
at a time). Overlap onto **CPU / host RAM** while a GPU cell runs:

- **CONCURRENT (CPU, while a GPU cell runs):**
  - **Build the CA overlap engine on CPU** (`build_overlap_probe_engine.py california 1`) while the FL Task-1 MTL
    cell occupies the GPU. CA re-windowing is the heaviest of the build steps but is still pure pandas/numpy.
  - **Build the CA seeded per-fold log_T** (`compute_region_transition.py … --per-fold --seed 0`) — CPU; before
    the CA reg cells.
  - **Matched-rescore + Δ aggregation** of a *finished* cell's JSONs (`r0_matched_rescore.py`) while the next GPU
    cell trains.
- **STRICTLY SEQUENTIAL (single A100):** every `scripts/train.py --task mtl …` and every
  `p1_region_head_ablation.py …` cell. Never launch two GPU cells at once. The shared (per-box)
  `TORCHINDUCTOR_CACHE_DIR` means the first compiled cell warms the cache; later cells reuse it (0 warmup).

---

## 2 · THE PINS (checklist — verify EVERY one before EVERY paired cell)

- [ ] **torch == 2.11.0+cu128.** `python -c "import torch;print(torch.__version__)"` → must equal `2.11.0+cu128`.
      `train.py` WARNs on mismatch; `p3_board.sh` hard-refuses. (2.12 rewrote TopK → reg Acc@10 tie-break shifts.)
- [ ] **Freshness preflight before EVERY `--per-fold-transition-dir` run.** The seeded per-fold log_T
      (`region_transition_log_seed{S}_fold{N}.pt`) must be **newer** than the engine's `next_region.parquet`,
      else +8…+12 pp leaks silently into reg Acc@10. Guard: `src/data/log_t_freshness.py`
      (`assert_log_t_fresh` / `assert_per_fold_dir_fresh`); `p1_region_head_ablation.py` calls it inline.
      Manual mtime check (CA):
      ```bash
      stat -c '%y %n' output/check2hgi_dk_ovl/california/region_transition_log_seed0_fold*.pt
      stat -c '%y %n' output/check2hgi_dk_ovl/california/input/next_region.parquet
      # if any log_T mtime < next_region.parquet mtime → rebuild (see §3)
      ```
- [ ] **The B-A2 trap — assert the STL ceiling is itself on OVERLAP windowing.** The CA reg ceiling MUST be built
      on the SAME gated-overlap base it is compared to, scored on the SAME FULL `top10_acc`. It runs with
      `--engine-override check2hgi_dk_ovl`; if it ever falls back to non-overlap `check2hgi`, the Δreg is void.
      **Fail loud.**
- [ ] **The gate guard — never train on a stale ungated / min_seq≠10 overlap build.** `folds._warn_if_ungated_overlap`
      WARNs (and `MTL_STRICT=1` HARD-FAILS) on `emit_tail=True` or `min_sequence_length ≠ 10`. **Run every cell
      with `MTL_STRICT=1`.** Board windowing = **stride-1, GATED, MIN_SEQ=10** — `build_overlap_probe_engine.py
      california 1` auto-gates and defaults min_seq=10.
- [ ] **auto-fit — NEVER `MTL_DATASET_GPU=1`** (CA especially). Default auto-fit keeps the large CA overlap
      dataset CPU-resident; `MTL_DATASET_GPU=1` forces ~31 GB redundant copies → OOM with no CPU fallback.
- [ ] **Matched scorer = FULL `top10_acc`, fp32-eval, BOTH MTL and STL sides** (B-A2). cat = **macro-F1**.
      `scripts/mtl_improvement/r0_matched_rescore.py` is the validated converter.
- [ ] **Commit the result JSONs (C28; closes audit F3-3).** MTL + STL + baseline JSONs, per cell (PID-suffixed
      rundir, per-run seed echo).
- [ ] **Device-class rule.** Every paired comparison end-to-end on ONE device-class (A100 = CUDA tf32+compile).
      **Until the A/B confirms `|Δ| ≤ ±0.05 pp`, do NOT pair an A100 cell against an A40 cell** — that confirmation
      is the whole point of Task 1.
- [ ] **Seeds.** The A/B + the early CA cell are **1 seed (seed 0)**. The full P3 board (post-freeze) is
      **{0, 1, 7, 100}**.

**STALE-log_T extra warning:** the engine builder does NOT rebuild log_T; an old log_T survives across a
re-windowing. Always rebuild CA's log_T AFTER building/rebuilding its overlap engine.

---

## 3 · EXACT COMMANDS

> Shared env every CUDA cell exports (same as the A40, but a **per-box** inductor cache):
> ```bash
> export MTL_CHUNK_VAL_METRIC=1            # S2 chunked val metric (needed at 8.5× overlap scale; essential for CA's large C)
> export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
> export OMP_NUM_THREADS=24
> export MTL_STRICT=1                      # gate guard hard-fails on a stale ungated/min_seq≠10 overlap build
> export MTL_COMPILE_DYNAMIC=1             # one symbolic-shape graph (compile path)
> export TORCHINDUCTOR_CACHE_DIR=<A100-LOCAL persistent path, e.g. ~/.inductor_cache_board_a100>   # NOT shared with the A40
> ```

`V14=check2hgi_design_k_resln_mae_l0_1` (frozen v14 substrate — never rebuilt) ·
`OVL=check2hgi_dk_ovl` (gated stride-1 overlap engine).

### 3a · Build the overlap engine (CPU — run while a GPU cell is busy)
```bash
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py california 1
#                                                                                    state stride
# auto-gates at stride==1 (emit_tail=False), min_seq defaults to 10. Symlinks v14 embeddings/region/poi.
```

### 3b · Build the seeded per-fold log_T (CPU — after the engine, before any reg cell)
```bash
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state california --per-fold --seed 0
# emits region_transition_log_seed0_fold{1..5}.pt; then verify freshness (§2 mtime check).
```

### 3c · Task 1 — FL champion-G MTL cell (the A/B half) — BYTE-IDENTICAL to the A40
Run the **exact same command the A40 runs** (champion-G = `gated_overlap_g.sh` recipe + `--compile --tf32`):
```bash
.venv/bin/python scripts/train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
    --state florida --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower \
    --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/florida --no-checkpoints
```
> The command MUST be byte-identical to the A40's (§3c of `HANDOFF_BOARD_A40.md`) — only the box differs. Capture
> the rundir by PID suffix. Score with `r0_matched_rescore.py` → cat macro-F1 + reg FULL top10_acc to **4 dp**.
> Confirm `|Δ| ≤ ±0.05 pp` vs the A40 on BOTH heads **before** any by-state split.

### 3d · Task 2 — matched B-A2 reg pair on CA overlap (1 seed × 5 folds)
The matched pair is exactly what `scripts/pre_freeze_gates/fl_overlap_compare.sh` runs for FL; for CA run the two
reg arms (B-A2):

**STL reg ceiling on overlap** (verbatim arg shape from `fl_overlap_compare.sh` Cell 2 — `--engine-override $OVL`
is the B-A2 windowing-match; add `--compile --tf32`):
```bash
.venv/bin/python scripts/p1_region_head_ablation.py --state california --heads next_stan_flow \
    --input-type region --region-emb-source check2hgi_design_k_resln_mae_l0_1 \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --engine-override check2hgi_dk_ovl \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/california \
    --folds 5 --epochs 50 --seed 0 --target region \
    --compile --tf32 \
    --tag ca_ovl_stl_reg_s0
# AGGREGATE line Acc@10 = STL reg ceiling (FULL top10_acc, fp32). This IS the B-A2 ceiling — on overlap.
```

**Champion-G MTL on CA overlap** — the §3c command with `--state california` (auto-fit; NEVER `MTL_DATASET_GPU=1`;
CA is the largest, so `MTL_CHUNK_VAL_METRIC=1` is essential). Then:

**Δreg** = (champion-G MTL FULL top10_acc) − (STL `next_stan_flow` reg ceiling FULL top10_acc), both on the CA
overlap windowing. Report Δreg against **δ_reg = 2 pp**: `|Δreg| ≤ ~1.5 pp` → inside; `> 2 pp` → flag for the
reg-claim re-scope (NOT a stop — §5). CA is the highest-region state, so this is the load-bearing margin check.

---

## 4 · PROCESS

1. **Branch `study/board-a100`** (off `main`; never commit to `main`, never merge another lane).
2. **Open a DRAFT PR early** and push as you go (same pattern as PR #26–#29).
3. **Commit INCREMENTALLY** — per cell / per built artifact / per verdict: a small commit with the **result JSON**
   + a **one-line finding** (e.g. `A100 A/B FL s0: cat 78.3187 / reg 75.5210 (4dp) — Δ vs A40 pending`).
4. When the A/B half, the CA engine build, or the CA reg pair completes → **flag the PR for audit**. The
   orchestrator audits, gives further instructions, and merges/reconciles. You do not merge.
5. End commit messages with the required `Co-Authored-By:` trailer; end the PR body with the required
   `🤖 Generated with [Claude Code]` line. Only on `study/board-a100`.

---

## 5 · STOP conditions (A100-specific)

- **A/B `|Δ| > ±0.05 pp`** (vs the A40's byte-identical FL cell, either head) → by-state partition becomes
  mandatory and cross-GPU absolutes carry a caveat. **STOP and surface to the user/orchestrator** before any
  by-state parallelization.
- **CA Δreg > 2 pp** → this is **NOT a stop** (overlap is ADOPTED unconditionally). Record it and **flag for the
  reg-claim framing** — CA is the largest state, so a CA breach is exactly the case the early cell exists to
  catch: the paper re-scopes the reg claim honestly per EXECUTION_PLAN §12/§13 (non-inferior where it holds;
  composite-reg panel is the supportive fallback, never the headline).
- **Any guard failure → STOP:** freshness-preflight `StaleLogTError`, OOM (do NOT lower bs=2048 to fix it —
  region-MTL diverges at smaller batch; it's GPU routing, use auto-fit), or `_warn_if_ungated_overlap` under
  `MTL_STRICT=1`.
- **torch ≠ 2.11.0+cu128** → STOP.
