# HANDOFF — board launch · **A40** (CUDA) · branch `study/board-a40`

> One-machine handoff. You are the **A40** lane. The shared governing rule, launch sequence, and the per-machine
> branch+PR process live in the **index**: [`HANDOFF_BOARD.md`](HANDOFF_BOARD.md) — read it once, then work only
> from this file. Recipe/guards ground-truth: [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md)
> + [`../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md`](../pre_freeze_gates/OVERLAP_BOARD_FINDINGS.md);
> sequence + device rule: [`../EXECUTION_PLAN.md §12–§13`](../EXECUTION_PLAN.md); stats:
> [`STATISTICAL_PROTOCOL.md`](STATISTICAL_PROTOCOL.md); matched scorer + baseline design:
> [`RUN_MATRIX.md §0, §2.5`](RUN_MATRIX.md); champion-G recipe: [`../../NORTH_STAR.md`](../../NORTH_STAR.md).

---

## 0 · SCOPE BOUNDARY (what THIS machine does — and what it must NOT touch)

**The A40 does the precision gate + its early cells, then (post-freeze) owns its by-state partition.**

0. **Task 0 — PRECISION-EQUIVALENCE A/B (bf16 vs fp32) — NEW, run FIRST** (RUN_MATRIX §0a; gates the whole
   board's precision). FL champion-G MTL, seed 0, 5 folds, compile+tf32 fixed on BOTH arms (commands in §3a0):
   - Arm X (bf16): `MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1`
   - Arm Y (fp32): `MTL_DISABLE_AMP=1`
   Record matched `cat macro-F1` + reg FULL `top10_acc` per-fold + 5f-mean to **4 dp**. **Decision:**
   `|Δcat|,|Δreg| ≤ 0.05 pp` ⇒ board standardizes **bf16**; else **fp32**. Also report Arm-X reg vs the FL STL
   reg ceiling 76.71 (expect ≳ 0). **STOP and post the table for the user** — the chosen precision freezes into
   §0 before Task 1/2 and the full board. (Cross-checks the H100 fp32 fold-1 anchor: reg 77.71 / cat 79.43.)
1. **Task 1 — A100-equivalence A/B, A40 HALF.** ONE FL champion-G MTL cell on the gated-overlap engine, seed 0,
   5 folds, compiled+tf32, **in the Task-0 chosen precision**. Record `cat macro-F1` + `reg top10_acc` to **4 dp**.
   The A100 runs the *byte-identical* command on its half. PASS = `|Δ| ≤ ±0.05 pp` on BOTH heads.
2. **Task 2 — TX gated-overlap reg cell — ⚠ RE-RUN under the chosen precision.** The prior TX B-A2 result
   (`tx_ba2_s0.json`, Δreg −2.41, `mtl_reg_best_epochs=[4,50,4,4,5]`) is a **fp16 ep30-collapse artifact → VOID**
   (same signature as CA; see `CA_MTL_DIVERGENCE.md`). Re-run the matched **B-A2 reg** pair — STL `next_stan_flow`
   reg ceiling on overlap (already fp32, can REUSE the prior ceiling 64.96) **+** champion-G MTL reg on overlap
   **in bf16/fp32** — 1 seed × 5 folds; report the corrected **Δreg vs δ_reg = 2 pp**. ~11 h for the TX MTL cell.

**Do NOT do (these belong to other machines):**
- **Do NOT run the CA early reg cell** — that is the **A100** (`study/board-a100`). CA is the largest state
  (8501 regions) and is allocated to the A100's larger memory.
- **Do NOT build the light substrate-column baseline embeddings** (CTLE / POI2Vec / skip-gram / one-hot) — those
  are the **M2 Pro** (`study/board-m2pro`).
- **Do NOT run the matched-head baseline COMPARISON on a Mac**, and **do NOT** pull a state's baselines onto a Mac
  while its STL/MTL is here — every paired comparison stays on ONE device-class (the governing rule).
- **Do NOT** trigger the P2 freeze or launch the full P3 board — that is the orchestrator's call after both early
  cells are recorded and the RUN_MATRIX is signed.
- **Do NOT** commit to `main`, do NOT merge any branch. Work only on `study/board-a40`.

---

## 1 · PARALLELISM on the A40 (single GPU)

The A40 is **one GPU → GPU cells are strictly SERIAL** (Task 1's FL MTL, then Task 2's STL-reg, MTL-reg run one
at a time). But several things overlap onto **CPU / host RAM** while a GPU cell is running:

- **CONCURRENT (CPU, while a GPU cell runs):**
  - **Build the next state's overlap engine on CPU.** `build_overlap_probe_engine.py texas 1` is pure
    pandas/numpy (re-windowing + symlinks) — run it while the FL Task-1 MTL cell occupies the GPU, so TX inputs
    are staged before Task 2 needs them.
  - **Build/refresh the TX seeded per-fold log_T** (`compute_region_transition.py … --per-fold --seed 0`) — CPU
    work; do it before the TX reg cells.
  - **Matched-rescore + Δ aggregation** of a *finished* cell's JSONs (`r0_matched_rescore.py`, the per-fold
    aggregation in `fl_overlap_compare.sh`) while the next GPU cell trains.
- **STRICTLY SEQUENTIAL (single A40):** every `scripts/train.py --task mtl …` and every
  `p1_region_head_ablation.py …` cell. Never launch two GPU cells at once on the A40 (they contend for the 46 GB
  card and confound timing). The shared `TORCHINDUCTOR_CACHE_DIR` means the **first** compiled cell pays the
  one-time break-even warmup; every later cell on this box reuses the cache (~13–21 % faster, 0 warmup).

---

## 2 · THE PINS (checklist — verify EVERY one before EVERY paired cell)

- [ ] **torch == 2.11.0+cu128.** `python -c "import torch;print(torch.__version__)"` → must equal `2.11.0+cu128`.
      `train.py` WARNs on mismatch; `p3_board.sh` hard-refuses. (2.12 rewrote TopK → reg Acc@10 tie-break shifts.)
- [ ] **Freshness preflight before EVERY `--per-fold-transition-dir` run.** The seeded per-fold log_T
      (`region_transition_log_seed{S}_fold{N}.pt`) must be **newer** than the engine's `next_region.parquet`,
      else +8…+12 pp leaks silently into reg Acc@10. Guard module: `src/data/log_t_freshness.py`
      (`assert_log_t_fresh` / `assert_per_fold_dir_fresh`); `p1_region_head_ablation.py` calls it inline.
      Manual mtime check:
      ```bash
      stat -c '%y %n' output/check2hgi_dk_ovl/texas/region_transition_log_seed0_fold*.pt
      stat -c '%y %n' output/check2hgi_dk_ovl/texas/input/next_region.parquet
      # if any log_T mtime < next_region.parquet mtime → rebuild (see §3)
      ```
- [ ] **The B-A2 trap — assert the STL ceiling is itself on OVERLAP windowing.** The reg ceiling MUST be built on
      the SAME gated-overlap base it is compared to, scored on the SAME FULL `top10_acc`. In Task 2 the STL reg
      ceiling runs with `--engine-override check2hgi_dk_ovl` (overlap) — if it ever falls back to non-overlap
      `check2hgi`, the Δreg is void. **Fail loud**, do not paper over a windowing mismatch.
- [ ] **The gate guard — never train on a stale ungated / min_seq≠10 overlap build.** `folds._warn_if_ungated_overlap`
      WARNs (and `MTL_STRICT=1` HARD-FAILS) if the overlap engine sidecar says `emit_tail=True` or
      `min_sequence_length ≠ 10`. **Run every cell with `MTL_STRICT=1`** so a stale ungated build cannot silently
      train (it bit AL once → a phantom −2.5 pp). The board windowing is **stride-1, GATED (`emit_tail=False`),
      MIN_SEQ=10** — `build_overlap_probe_engine.py <state> 1` auto-gates at stride==1 and defaults min_seq=10.
- [ ] **auto-fit — NEVER `MTL_DATASET_GPU=1` for CA/TX (or any state).** The default auto-fit keeps the large
      overlap dataset CPU-resident (TX GPU peak ~6 GB, ~160 s/epoch). `MTL_DATASET_GPU=1` forces ~31 GB of
      redundant per-fold copies → OOM with no CPU fallback. Leave it unset.
- [ ] **Matched scorer = FULL `top10_acc`, fp32-eval, on BOTH MTL and STL sides** (B-A2). cat = **macro-F1**.
      `scripts/mtl_improvement/r0_matched_rescore.py` is the validated converter (full = indist × (1 − ood_frac)).
- [ ] **Commit the result JSONs (C28; closes audit F3-3).** The MTL headline + STL + baseline JSONs were missing
      before — commit them per cell (PID-suffixed rundir, per-run seed echo).
- [ ] **Device-class rule.** Every paired comparison end-to-end on ONE device-class. The A40 is CUDA tf32+compile;
      never pair an A40 cell against an MPS-built or A100-built cell (until the A/B confirms ±0.05 pp).
- [ ] **Seeds.** The A/B + the early TX cell are **1 seed (seed 0)**. The full P3 board (post-freeze) is
      **{0, 1, 7, 100}**.

**STALE-log_T extra warning:** the engine builder does NOT rebuild log_T, and `train.py` does not validate its
freshness beyond the mtime guard — an old log_T survives across a re-windowing. Always rebuild log_T for the
state+seed AFTER building/rebuilding its overlap engine.

---

## 3 · EXACT COMMANDS

> Paths below use the A40 box layout (`REPO=/home/vitor.oliveira/PoiMtlNet`, `PY=.venv/bin/python`,
> `PYTHONPATH=src`) exactly as the committed drivers set them. The shared env every CUDA cell exports:
> ```bash
> export MTL_CHUNK_VAL_METRIC=1            # S2 chunked val metric (needed at 8.5× overlap scale)
> export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
> export OMP_NUM_THREADS=24
> export MTL_STRICT=1                      # gate guard hard-fails on a stale ungated/min_seq≠10 overlap build
> export MTL_COMPILE_DYNAMIC=1             # one symbolic-shape graph (compile path)
> export TORCHINDUCTOR_CACHE_DIR=/home/vitor.oliveira/.inductor_cache_board   # ONE shared persistent cache
> ```

`V14=check2hgi_design_k_resln_mae_l0_1` (frozen v14 substrate — never rebuilt) ·
`OVL=check2hgi_dk_ovl` (gated stride-1 overlap engine).

### 3a0 · Task 0 — precision-equivalence A/B (bf16 vs fp32), FL, seed 0, 5f — RUN FIRST
Both arms are the EXACT §3c champion-G FL command (same env above, `--compile --tf32`); they differ ONLY by the
precision env vars prefixed on the line. Run serially; score BOTH with `r0_matched_rescore.py` (fp32 re-forward).
```bash
# Arm X — bf16 train, fp32 eval (the proposed board default)
MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1 \
  .venv/bin/python scripts/train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
    --state florida --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/florida --no-checkpoints

# Arm Y — full fp32 train + eval (the reference). Identical except the prefix:
MTL_DISABLE_AMP=1   .venv/bin/python scripts/train.py ... (same flags as Arm X) ...
```
Report a 4-dp table: per-fold + 5f-mean cat macro-F1 and reg FULL top10_acc for X and Y, `|Δ|` per head, and
Arm-X reg vs the FL STL reg ceiling **76.71**. **Decision rule:** `|Δcat|,|Δreg| ≤ 0.05 pp` ⇒ bf16; else fp32.
**STOP — post the table for the user before Task 1.** (Note: fp32 (Arm Y) is ~2–3× slower; bf16 keeps speed.)

### 3a · Build the overlap engine (CPU — run while a GPU cell is busy)
```bash
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py texas 1
#                                                                                    state stride
# auto-gates at stride==1 (emit_tail=False), min_seq defaults to 10 (board value).
# Symlinks v14 embeddings/region/poi; rebuilds ONLY the overlapping next/sequences/next_region.
```

### 3b · Build the seeded per-fold log_T (CPU — after the engine, before any reg cell)
```bash
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state texas --per-fold --seed 0
# emits region_transition_log_seed0_fold{1..5}.pt; then verify freshness (§2 mtime check).
```
> ⚠ The early cells point `--per-fold-transition-dir` at `output/$V14/$ST` (the gated_overlap_g.sh / fl_overlap_compare.sh
> convention — the v14 dir holds the seeded log_T the trainer reads). When you rebuild the overlap engine, make
> sure the log_T the trainer will read is **newer than the overlap `next_region.parquet`** it scores against.

### 3c · Task 1 — FL champion-G MTL cell (the A/B half), compiled + tf32, seed 0
Champion-G is exactly the `gated_overlap_g.sh` invocation. For the A/B add the compile knobs (env above) and the
`--compile --tf32` flags. Canonical G command (verbatim from `scripts/pre_freeze_gates/gated_overlap_g.sh`):
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
> ⚠ Two notes vs the raw script: (1) the committed `gated_overlap_g.sh` does **not** pass `--compile --tf32` (it
> was the uncompiled R1-comparable run) — the **A/B and the whole board ADD them** (PINNED, RUN_MATRIX §0 /
> DEFAULTS_AND_GUARDS). (2) `gated_overlap_g.sh florida 0 50 5` runs the same recipe minus the compile flags; to
> reuse the script as-is wrap it and add the flags, or run the explicit command above. Capture the rundir by PID
> suffix (`mtlnet_*_${pid}`), not `ls -dt | head`. Score with `r0_matched_rescore.py` → cat macro-F1 + reg FULL
> top10_acc to **4 dp**; hand the two numbers to the orchestrator for the |Δ| ≤ ±0.05 pp check vs the A100.

### 3d · Task 2 — matched B-A2 reg pair on TX overlap (1 seed × 5 folds)
The whole matched pair (STL cat ceiling + STL reg ceiling + champion-G MTL + aggregation) is exactly what
`scripts/pre_freeze_gates/fl_overlap_compare.sh` does for FL; for TX run the two reg arms (B-A2):

**STL reg ceiling on overlap** (verbatim arg shape from `fl_overlap_compare.sh` Cell 2 — `--engine-override $OVL`
is the B-A2 windowing-match; add `--compile --tf32` for the board recipe):
```bash
.venv/bin/python scripts/p1_region_head_ablation.py --state texas --heads next_stan_flow \
    --input-type region --region-emb-source check2hgi_design_k_resln_mae_l0_1 \
    --override-hparams freeze_alpha=True alpha_init=0.0 \
    --engine-override check2hgi_dk_ovl \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/texas \
    --folds 5 --epochs 50 --seed 0 --target region \
    --compile --tf32 \
    --tag tx_ovl_stl_reg_s0
# reads the AGGREGATE line: Acc@10 = STL reg ceiling (FULL top10_acc, fp32). This IS the B-A2 ceiling — on overlap.
```

**Champion-G MTL on TX overlap** — same command as §3c with `--state texas` (~11 h; auto-fit; NEVER
`MTL_DATASET_GPU=1`). Then:

**Δreg** = (champion-G MTL FULL top10_acc) − (STL `next_stan_flow` reg ceiling FULL top10_acc), both on the TX
overlap windowing. Report Δreg against **δ_reg = 2 pp**: `|Δreg| ≤ ~1.5 pp` → comfortably inside; `> 2 pp` →
flag for the reg-claim re-scope (NOT a stop — see §5). Cat is a superiority side-check (it strengthens under
overlap, +3.1…+3.8 historically at FL).

---

## 4 · PROCESS

1. **Branch `study/board-a40`** (off `main`; never commit to `main`, never merge another lane).
2. **Open a DRAFT PR early** and push as you go (same pattern as PR #26–#29).
3. **Commit INCREMENTALLY** — per cell / per built artifact / per verdict: a small commit carrying the **result
   JSON** + a **one-line finding** (e.g. `A40 A/B FL s0: cat 78.3210 / reg 75.5183 (4dp) — pending A100 pair`).
   Never one giant end-of-run commit.
4. When the A/B half, the TX engine build, or the TX reg pair completes → **flag the PR for audit**. The
   orchestrator audits, gives further instructions, and merges/reconciles. You do not merge.
5. End commit messages with the required `Co-Authored-By:` trailer; end the PR body with the required
   `🤖 Generated with [Claude Code]` line. **Do not push/commit unless this handoff's tasks produce artifacts to
   commit** — and only on `study/board-a40`.

---

## 5 · STOP conditions (A40-specific)

- **A/B `|Δ| > ±0.05 pp`** (vs the A100's byte-identical FL cell, either head) → by-state partition becomes
  mandatory and cross-GPU absolutes carry a caveat. **STOP and surface to the user/orchestrator** before any
  by-state parallelization.
- **TX Δreg > 2 pp** → this is **NOT a stop** (overlap is ADOPTED unconditionally). Record it and **flag for the
  reg-claim framing**: the paper re-scopes the reg claim honestly per EXECUTION_PLAN §12/§13 (non-inferior where
  it holds; the 2-model composite-reg panel is the supportive fallback, never the headline).
- **Any guard failure → STOP:** freshness-preflight `StaleLogTError`, OOM, or `_warn_if_ungated_overlap` under
  `MTL_STRICT=1`. Do not work around it — fix the root cause (rebuild log_T / re-gate the engine) or surface it.
- **torch ≠ 2.11.0+cu128** → STOP; do not run a freeze-grade comparison on the wrong build.
