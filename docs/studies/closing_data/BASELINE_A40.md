# BASELINE handoff — **A40** (CUDA, Ampere) · self-contained · 2026-06-24

> **You are the A40. Read ONLY this file, then execute.** Phase = baselines. Your job: the **CSLSL cascade
> (role-3) at the small/mid states** — the published multi-task alternative to our parallel joint model. Start
> when your current run frees the card (~1 h). Decisions: `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`;
> results → [`RESULTS_BOARD.md`](RESULTS_BOARD.md); cross-machine map → [`BASELINE_DISTRIBUTION.md`](BASELINE_DISTRIBUTION.md).
>
> **Protocol:** seed 0 × 5 folds (n=5), gated stride-1 overlap engine `check2hgi_dk_ovl`, **fp32**, leak-free
> per-fold train-only priors, user-disjoint folds. Numbers from committed JSONs only.

## 0 · Setup (once)
```bash
cd <A40 repo>; git checkout main && git pull
export PYTHONPATH=src
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
```

## 1 · What CSLSL is (so you frame it right)
`scripts/baselines/b4_cascade.py` is the **cascade (category→region) multi-task baseline** — the dominant published
alternative to parallel MTL. It is a PINNED SC variant that reuses the **exact champion heads on the frozen
Check2HGI substrate**, with the only varying factor being the directed cat→region cascade edge (vs our parallel
joint). So it isolates **cascade-vs-parallel** cleanly. Comparand = **our MTL champion-G** (NOT the STL ceiling).

## 2 · Tasks (when the card frees, ~1 h) — seed 0 × 5f
```bash
for S in alabama arizona; do
  python scripts/baselines/b4_cascade.py --state $S --seed 0 --folds 5 --epochs 50
done
#  Preflight will STOP with the exact build command if the per-fold log_T is missing, e.g.:
#    python scripts/compute_region_transition.py --state $S --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5
#  (the dk_ovl engine itself: python scripts/mtl_improvement/build_overlap_probe_engine.py $S 1)
```
- **CA / TX: only if cheap / spare time** (wide-region 8501/6553 heads; the `perf(mtl)` #33 fix on main makes them feasible but long). Default: skip CA/TX cascade for the deadline.
- **If the H100 is saturated**, you may instead take **CSLSL @ FL** (`b4_cascade.py --state florida …`) — you're the stable card for the longer run.

## 3 · Clean-Δ note (important)
The cascade's comparand is **champion-G**, whose AL/AZ rundirs are on the **H100**. For a strictly same-device Δ,
**re-run champion-G AL/AZ on the A40 alongside the cascade** (cheap at AL/AZ) and compare on this card — OR accept
the documented cross-GPU **±0.05 pp** caveat (acceptable for this internal cascade-vs-parallel ablation, where the
signal ≫ 0.05 pp). State which you did in the result.

## 4 · Validation + outputs
- **Expected:** the cascade is **≈ champion-G or slightly worse** on the joint objective (our parallel model should
  beat or match it at equal/lower cost). A cascade that *beats* champion-G by a wide margin → check the cascade edge
  wiring (`b4_cascade.py` docstring D-points) before trusting.
- Commit the cascade result JSON per state (rundir under `results/…`); record the cat/reg + joint-objective number
  in `RESULTS_BOARD.md` (the MTL-comparator block) + `MACS_BOARD_RESULTS.md`. **n=5 provisional.**
- Do **NOT** run the FL representation block (H100's job), the CTLE diagnosis (M4's job), or any dropped baseline
  (CTLE-SC ladder / region-SC / STAN-faithful at scale).
