# HANDOFF — REDUCED board · **H100** (CUDA, Hopper) · branch `study/board-h100`

## 🚀 NEXT-AGENT KICKOFF — connect the H100, then start here
**State as of 2026-06-23 (handoff author):** the GPU is currently a **CPU-only fallback** (`torch.cuda.is_available()==False`);
the H100 is **not attached** — the user must restart/attach the Lightning studio to get it. The filesystem PERSISTS.
Everything else is ready, so on a fresh GPU window you can start in minutes:
- **Env READY:** `torch 2.11.0+cu128` + matching pyg stack already installed in the studio venv (`.venv`). No reinstall.
- **Prep DONE on disk (do NOT rebuild unless the freshness preflight fails):** v14 substrate + gated-overlap engine
  built for AL/AZ/FL/CA (and GA/TX), and **seed-0 per-fold log_T present for AL/AZ/FL/CA (5 folds each)** under
  `output/check2hgi_design_k_resln_mae_l0_1/<state>/region_transition_log_seed0_fold*.pt`.
- **Trainer hardened (on `main`, commit `27b4bc5a`):** `mtl_cv.py` has the default-on `guard_finite_step` —
  board runs set **`MTL_STRICT=1`** so any NaN event ABORTS loud (no more silent ep30 collapse). bf16 via
  `MTL_AUTOCAST_BF16=1`. Test: `tests/test_training/test_mtl_nonfinite_guard.py` (10/10).
- **ON CONNECT, in order:** (1) `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
  → must be a Hopper H100; (2) `git checkout study/board-h100 && git merge main` — **the guard fix + this patched
  doc live on `main` (commit `27b4bc5a`); the board branch MUST contain them before you run** (confirm with
  `git log --oneline | grep 27b4bc5a` and `grep -q guard_finite_step src/training/runners/mtl_cv.py`). Keep results
  on the board branch / its own PR — do NOT push results to `main`; (3) re-export the board env (§1 block);
  (4) **log_T freshness preflight** (`src/data/log_t_freshness.py`,
  enforced by `MTL_STRICT=1`); (5) start **§1 — the AL precision gate (bf16-vs-fp32)**, which BLOCKS all other MTL.
  After the gate verdict, §3 (AZ+FL co-scheduled) then §4 (CA — note its restart-risk caveat: launch CA at the
  START of a fresh window so it can clear ep30 before the next ~1-2 h restart). **STOP for the user** at: the
  precision-gate table; any CA bf16 run that still collapses at/after ep30; any freshness/OOM/leak-guard failure.

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). The H100 carries the **precision gate** + the small/mid
> states + **CA last**. **1 seed (0) × 5 folds** everywhere. Incremental commits + own PR; do NOT touch main or
> another lane. The prior fp16 results on this branch are VOID (CA_MTL_DIVERGENCE.md) but the branch is mergeable
> as a record — we reset/re-run on top.

## ⭐ LEARNINGS CARRIED IN (read [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md) in full before running anything)
These are the hard-won facts from the audit that produced this reduced board. They change *why* you run each cell.
1. **The MTL trainer ran fp16 autocast with NO GradScaler — that, not tf32, caused the CA "ep30 collapse".**
   At CA scale the 8501-wide reg logits overflow fp16's 65504 ceiling → NaN → `clip_grad_norm_` (inf-norm →
   coef 0) zeros the finite grads and turns the offending grad into `inf*0 = NaN`, which `optimizer.step()`
   writes into the SHARED cross-attn backbone → both heads collapse to the ~3% floor. Deterministic (seed 0),
   scale-dependent (FL 1/5 @ep50 benign; CA all-folds @ep30 fatal; TX must be assumed at risk). **All prior MTL
   numbers trained under this fp16 harness.**
2. **The fix is precision: bf16 autocast** (`MTL_AUTOCAST_BF16=1`, fp32 exponent range, no overflow, no
   GradScaler needed). fp32 (`MTL_DISABLE_AMP=1`) is the slow reference. **bf16 is validated healthy through
   ep28 only — the ep30 cliff is NOT yet definitively cleared** (studio restart truncated it 2 epochs short).
   ⇒ **CA bf16 must run clean past ep30; if it still collapses, STOP and escalate (the fix is insufficient).**
3. **A default-on fail-loud guard now exists in `mtl_cv.py` (`guard_finite_step`).** A healthy run is unaffected
   (byte-identical). On a non-finite grad/loss it **skips the poisoning step** by default; with **`MTL_STRICT=1`
   it ABORTS** (board runs set `MTL_STRICT=1` → a NaN event fails loud instead of silently collapsing). Regression
   test: `tests/test_training/test_mtl_nonfinite_guard.py`. The guard is defense-in-depth — **precision (bf16) is
   still the real fix**; do not rely on the skip to "rescue" a run.
4. **HEADLINE for the paper — correct precision CLOSES, and at FL REVERSES, the MTL-vs-STL reg gap.** The
   "MTL sacrifices reg" finding was largely a fp16-harness artifact (a crash + a systematic understatement),
   compounded by a **precision MISMATCH**: MTL reg (fp16) was being compared against an fp32 STL reg ceiling.
   In matched fp32: AL closes ~half the gap (−0.38→−0.18); **FL fold-1 reg 77.71 > STL ceiling 76.71** (cat
   79.43 > 75.15) — beats BOTH ceilings. **Score and report Δreg/Δcat for every state under the chosen precision;
   this is now a positive result for the central claim, not a caveat.**
5. **Re-baseline scope = MTL cells ONLY.** The STL **reg** ceiling is already TRUE fp32 (`p1_region_head_ablation.py:83`,
   no autocast) → **do NOT re-run it** (REUSE AL 69.98 / FL 76.71 / CA 63.48). The STL **cat** ceiling is fp16,
   but cat is precision-insensitive (AL 63.44→63.48) → re-run only for Δcat parity, low priority.

## 0 · SCOPE — H100 owns: AL, AZ, **FL**, CA (full cell set each) + the precision gate
Run in this order (cheap → expensive). Each state = MTL champion-G + its STL ceilings, seed 0 × 5f. **FL moved
here from the A40** (2026-06-23, user) so it can be **co-scheduled with the small states** (§1a) — the small
states underutilize the 80 GB H100, so packing FL alongside uses the idle capacity. The A40 then does TX only.

## 1a · PARALLELISM — co-schedule FL with a small state (the time optimization)
The H100 has headroom the small states (AL 113k, AZ 236k rows) cannot fill. Run **two `train.py` processes at
once** to use it: a small-state MTL (~tiny) **+ FL MTL** (~13 GB peak). Combined ≈ 15-18 GB ≪ 80 GB → fits.
Rules for co-scheduling:
- **Distinct `TORCHINDUCTOR_CACHE_DIR` per process** (e.g. `..._board_a` / `..._board_fl`) — two processes
  compiling into ONE cache dir race and corrupt it.
- Distinct PID-suffixed rundirs (default). Both fp32-scored independently by `scripts/closing_data/h100_score_matched.py`
  (`<rundir> --seed 0`; writes `h100_matched_score.json` in the rundir — that is the matched-precision FULL-top10/macro-F1 sidecar; `r0_matched_rescore.py` is the method it implements, not a runnable script here).
- Co-scheduling does **not** affect metrics (only wall-clock sharing) → the per-state Δ's stay valid.
- **The AL gate (§1) runs alone-ish FIRST** (it's the priority that unblocks the precision choice); during it,
  do the **CPU** prep for FL/AZ/CA (overlap-engine build + log_T) and optionally FL's **STL ceilings** (GPU,
  precision-free). FL **MTL** starts after the gate verdict, co-scheduled with **AZ MTL** (§3).

## 1 · ⚡ STEP 0 — precision gate (bf16 vs fp32) at AL — RUN FIRST, BLOCKS all MTL
This doubles as the AL MTL cell. Two arms, identical except the env prefix; both scored fp32 by
`scripts/closing_data/h100_score_matched.py <rundir> --seed 0`. Arm X = **bf16** (`MTL_AUTOCAST_BF16=1`), Arm Y =
**fp32** (`MTL_DISABLE_AMP=1`) — both avoid the fp16 overflow, so this gate is bf16-vs-fp32, NOT fp16-vs-anything.
```bash
# env (per-box, once). MTL_STRICT=1 makes the new guard_finite_step ABORT on any NaN event (fail-loud).
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
# build AL overlap engine + seeded log_T (CPU; ~minutes)
PYTHONPATH=src .venv/bin/python scripts/mtl_improvement/build_overlap_probe_engine.py alabama 1
PYTHONPATH=src .venv/bin/python scripts/compute_region_transition.py --state alabama --per-fold --seed 0

# Arm X — bf16 (proposed default):
MTL_AUTOCAST_BF16=1 MTL_DISABLE_AMP_EVAL=1 \
  .venv/bin/python scripts/train.py --task mtl --task-set check2hgi_next_region --engine check2hgi_dk_ovl \
    --state alabama --seed 0 --epochs 50 --folds 5 --batch-size 2048 \
    --mtl-loss static_weight --category-weight 0.75 \
    --cat-head next_gru --reg-head next_stan_flow_dualtower \
    --reg-head-param raw_embed_dim=64 --reg-head-param fusion_mode=aux \
    --reg-head-param freeze_alpha=True --reg-head-param alpha_init=0.0 \
    --task-a-input-type checkin --task-b-input-type region --log-t-kd-weight 0.0 \
    --scheduler onecycle --max-lr 3e-3 --cat-lr 1e-3 --reg-lr 3e-3 --shared-lr 1e-3 \
    --model mtlnet_crossattn_dualtower --compile --tf32 \
    --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/alabama --no-checkpoints
# Arm Y — full fp32 (reference): same line, prefix  MTL_DISABLE_AMP=1  instead.
```
**Decision rule:** `|Δcat|,|Δreg| ≤ 0.05 pp` (4 dp, per-fold + mean) bf16-vs-fp32 ⇒ board standardizes **bf16**
(fast, equal quality); else **fp32**. **STOP and post the 4-dp table for the user.** Keep the chosen arm's AL run
as the AL MTL result. **AL anchors (seed0, 5f, gated overlap, already measured — sanity-check against these):**
STL reg ceiling **69.98**; fp16 MTL reg 69.60 (Δ −0.38, cat 63.44); **fp32 MTL reg 69.80 (Δ −0.18, cat 63.48)**.
bf16 should land at fp32 (≈69.80 / ≈63.48). A bf16 arm near 69.60 (the fp16 value) would mean bf16 isn't taking
effect — check `MTL_AUTOCAST_BF16=1` is exported into the process.

## 2 · STEP 1 — AL ceilings (precision-free; can run during/after the gate)
```bash
# STL cat ceiling
.venv/bin/python scripts/train.py --task next --engine check2hgi_dk_ovl --state alabama --seed 0 \
    --folds 5 --cat-head next_gru --compile --tf32   # macro-F1 ceiling
# STL reg ceiling (fp32 already) — REUSE prior 69.98 if a valid 5f artifact exists; else:
.venv/bin/python scripts/p1_region_head_ablation.py --state alabama --region-emb-source check2hgi_dk_ovl \
    --region-head next_stan_flow --seed 0 --folds 5 --per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/alabama
```

## 3 · STEP 2 — AZ + FL **co-scheduled** (the parallelism, post-gate)
Build the AZ **and** FL overlap engines + their seeded log_T on CPU first (as §1; do this during the AL gate).
Then launch TWO processes at once (§1a — distinct `TORCHINDUCTOR_CACHE_DIR` each):
- **AZ** (regions 1547, small): the §1 chosen-precision champion-G command, `--state arizona` + AZ STL ceilings.
- **FL** (regions 4703, ~13 GB): the same chosen-precision command, `--state florida`,
  `--per-fold-transition-dir output/check2hgi_design_k_resln_mae_l0_1/florida` + FL STL ceilings (reg ceiling
  **76.71 fp32 — REUSE** if a valid 5f artifact exists). FL fp32 anchor: fold-1 reg **77.71 > ceiling 76.71**
  (gap closed/reversed — expect FL MTL to meet/beat both ceilings).
Both seed 0 × 5f, chosen precision (no second A/B). They share the GPU; metrics are unaffected.

## 4 · STEP 3 — CA (LAST — heaviest, restart-risky, run ALONE)
CA (8501 regions, 3.17M rows) is where fp16 collapsed at ep30. bf16 is validated **healthy only through ep28**
(restart-truncated 2 epochs short) — so CA is also the **definitive ep30+ confirmation of the fix**. The full run
is ~40 min but the studio restarts every ~1-2 h, so **start CA FIRST in a fresh window** and let it run clean.
Build CA overlap engine + log_T, then the chosen-precision MTL + STL ceilings (reg ceiling **63.48 fp32 — REUSE**,
don't re-run). Auto-fit dataset; **NEVER `MTL_DATASET_GPU=1`** (OOM). If a restart truncates a fold, resume that
fold only.
**🛑 GATE:** CA MTL must run past ep30 with **zero non-finite events** (with `MTL_STRICT=1` a NaN aborts loudly;
watch the log for `[NONFINITE-GUARD]`). If CA still collapses at/after ep30 under bf16, **STOP and escalate** —
bf16 would be insufficient and the board needs GradScaler/fp32 instead. Only once a clean ep30+ CA run exists is
the bf16 standardization fully proven; record the real CA Δreg vs 63.48 (expected ≥ the AL/FL pattern, not the
VOID −5.23).

## 5 · PROCESS / PINS / STOP
- Branch `study/board-h100`; **incremental commits** (per cell + result JSON + 1-line finding); push as you go.
- Every `--per-fold-transition-dir` run: **log_T freshness preflight** (`src/data/log_t_freshness.py`); log_T
  mtime > the overlap `next_region.parquet` mtime. `MTL_STRICT=1` hard-fails a stale/ungated build.
- **STOP for the user:** the precision-gate table (§1); any OOM / freshness / leak-guard failure; any MTL run
  that still NaN-collapses under bf16 (would mean the fix is insufficient — escalate).
- Do NOT run TX (A40) or the baselines (Macs). Do NOT merge.
