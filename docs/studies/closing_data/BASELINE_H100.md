# BASELINE handoff — **H100** (CUDA, Hopper) · self-contained · 2026-06-24

> **You are the H100. Read ONLY this file, then execute.** Phase = baselines (the MTL board is done). Your job:
> the **Florida representation block (role-2)** + **CSLSL-cascade at FL (role-3)** — keep ALL of it on this one
> card so every Δ is device-internal-clean. Decisions are locked in `../../../articles/[mobiwac]/BASELINE_HANDOFF.md`;
> results consolidate to [`RESULTS_BOARD.md`](RESULTS_BOARD.md); cross-machine map in [`BASELINE_DISTRIBUTION.md`](BASELINE_DISTRIBUTION.md).
>
> **Protocol (non-negotiable):** seed 0 × 5 folds (n=5), gated stride-1 overlap engine `check2hgi_dk_ovl`,
> **fp32**, leak-free per-fold train-only priors, user-disjoint `StratifiedGroupKFold(groups=userid, y=next_category, seed 0)`,
> matched metric (cat macro-F1; reg Acc@10 checkin-modality). Numbers from committed JSONs only.

## 0 · Setup (once)
```bash
cd <H100 repo>; git checkout main && git pull        # MUST have the merged fixes (below)
export PYTHONPATH=src
export MTL_CHUNK_VAL_METRIC=1 MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
```
**Code fixes you MUST have (all merged to main — `git pull` is enough):** `perf(mtl)` 4.5× wide-head (#33),
p1 reg-head `collate_fn=_batched_collate` (else reg crashes `stack [B,9,64] vs [B]`), C-1/C-2/C-3 reg fixes
(comparand reg = checkin-modality; per-engine per-fold log_T), `comparand_check2hgi_sc.py`, min_seq=10 pin.
**Smoke first:** `python scripts/p1_region_head_ablation.py --engine check2hgi_dk_ovl --state alabama --only-fold 0 --heads next_stan_flow --input-type checkin --epochs 2` must train (not crash).

## 1 · Data check (FL) — confirm before running
| Artifact | Path | Note |
|---|---|---|
| v14 substrate | `output/check2hgi_design_k_resln_mae_l0_1/florida/{embeddings,region_embeddings,poi_embeddings}.parquet` | gate for `build_overlap_probe_engine` (FL ~3.3 G) |
| CTLE per-fold cells | `output/board_baselines/ctle/florida/s0_f{0..4}/embeddings.parquet` | 5 folds, each w/ `LEAK_MARKER.txt`="TRAIN-ONLY per fold" (leak gate). From SSD or rebuild. |
| check2hgi graph maps | `output/check2hgi/florida/` | input build + region symlinks |
Keep ≥30 G free (dk_ovl FL `next.parquet`+`next_region.parquet` ≈ 17 G). FL dk_ovl rowcount = **1,274,418**.

## 2 · Tasks (FL, seed 0 × 5f) — run in this order

**(A) Check2HGI-SC comparand @ FL** — the reference for the CTLE Δ. Run now (no gate).
```bash
python scripts/mtl_improvement/build_overlap_probe_engine.py florida 1
python scripts/compute_region_transition.py --state florida --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5
python scripts/closing_data/comparand_check2hgi_sc.py --state florida --folds 5 --heads cat reg
#  -> docs/results/closing_data/baseline_compare/florida_check2hgi_sc.json
```

**(B) CTLE-E2E @ FL** — CTLE's true (fine-tuned) strength; the headline CTLE number. Run now (no gate).
```bash
python scripts/baselines/ctle_e2e.py --state florida --seed 0 --folds 5
#  -> results/ctle_e2e_b1/florida/ctle_e2e_seed0.json  (record number in MACS_BOARD_RESULTS.md + RESULTS_BOARD §4)
```

**(C) feature-concat control @ FL** — "is the gain just feature injection, not the hierarchy?" Run now (no gate).
HGI per-place vector ⊕ raw per-visit features (category one-hot + hour/day sin/cos) → `next_gru` head, same folds.
No new embedding training. **Needs a thin builder:** extend `scripts/probe/build_design_a_concat.py` (it currently
concats c2hgi⊕HGI; swap the 2nd stream for raw features), then run the cat head on it. *(If the wrapper isn't on
main yet, ping the orchestrator — it's ~10 lines.)*

**(D) CTLE-SC @ FL** — ✅ **CLEARED 2026-06-24** (M4 CTLE-diagnosis verdict: the AL frozen CTLE-SC cat 17.8 below
the bigram floor is a **REAL CTLE frozen-substrate weakness, NOT a pipeline/leak bug** — proceed). Evidence:
[`../../results/closing_data/baseline_compare/alabama_ctle_DIAGNOSIS.md`](../../results/closing_data/baseline_compare/alabama_ctle_DIAGNOSIS.md)
— frozen CTLE embedding is real/non-degenerate, the substrate is genuinely swapped (cosine(CTLE,check2hgi)=0.01),
and the IDENTICAL head reaches 55.6 on check2hgi vs 17.8 on CTLE (same rows/folds) → the head learns; only the
substrate differs. The §3 sanity-gate "if CTLE cat ≈ Check2HGI cat → bug" is satisfied (Δ is huge, as expected).
```bash
python scripts/closing_data/mac_baseline_compare.py --state florida --baseline ctle --cells-root output --folds 5 --heads cat reg
#  -> docs/results/closing_data/baseline_compare/florida_ctle.json
```

**(E) CSLSL cascade @ FL (role-3)** — champion-G FL is on the H100 → clean same-device cascade-vs-parallel Δ.
```bash
python scripts/baselines/b4_cascade.py --state florida --seed 0 --folds 5 --epochs 50
#  (preflight will tell you to build the per-fold log_T if missing — same command as (A))
```

## 3 · Validation shapes (sanity-gate every cell)
- **CTLE-SC cat** ≈ 17–20; **Check2HGI-SC cat** ≈ 75 (FL ceiling 75.15) → Δcat large. If CTLE cat ≈ Check2HGI cat, the substrate isn't being swapped — **bug, stop.**
- **reg (checkin-modality)**: CTLE-SC reg ≈ Check2HGI-SC reg (region is a near-tie — the honest finding). If reg is bit-identical across baselines or lands at region-modality scale, C-1/C-2 regressed — **stop**.
- **CTLE-E2E cat** > frozen CTLE-SC cat (E2E undersells less); still well below ours.
- Comparand cat should match the existing FL dk_ovl cat ceiling `docs/results/closing_data/h100/florida_s0_stl_cat_ceiling.json` (75.147).

## 4 · Traps (these cost the Mac run hours — pre-empt)
1. **reg collate crash** — guaranteed if main is stale; `git pull`. Smoke (§0).
2. **`compute_region_transition` host-RAM** — materialises N×576 float32 (FL large); needs CPU RAM, guarded by `MTL_RAM_HEADROOM_GB`. GPU-irrelevant.
3. **min_seq desync** — CTLE-SC and the comparand must share min_seq=10; verify `len(check2hgi_ctle/florida/input/next.parquet) == len(check2hgi_dk_ovl/florida/input/next.parquet)` before trusting (D).
4. **wide-head speed** — the `perf(mtl)` #33 fix is on main; without it FL reg is ~70× slower.
5. **`build_next_region_for` is a Python loop over 1.27 M rows** — minutes, not a hang.

## 4b · Code fixes needed to run these baselines on a conda/CUDA board

Running the baselines on a **conda/CUDA** board (no `.venv`, fp16-unsafe single-task trainer) surfaced three
code bugs (1–3, **fixed in this PR**) plus one operational gotcha (4). The fixes are env-gated /
interpreter-agnostic / behaviour-preserving, so they're no-ops on the Mac/MPS board and on `main`'s existing
behaviour. Any other CUDA/conda board (e.g. the A40) benefits automatically.

1. **fp16 NaN collapse at FL scale (the important one).** `src/training/runners/_single_task_train.py`
   (train loop) and `src/training/shared_evaluate.py` (eval) ran an **unconditional `torch.autocast(float16)`
   on CUDA with no GradScaler and no off-switch** (MPS/CPU always got fp32 via `nullcontext` — why the Mac
   AL/AZ cells were clean but the H100 FL cat head `loss=nan` ~ep12, freezing F1 at 0.69 ≪ the 75.15 ceiling).
   The board protocol (§0) mandates **fp32** anyway. Patch: gate both autocast contexts on
   `DISABLE_AMP` / `MTL_DISABLE_AMP` (same convention the MTL trainer already uses), then export
   **`DISABLE_AMP=1 MTL_DISABLE_AMP=1`** for every CUDA baseline run. Validated NaN-free at FL.
   Background: memory `mtl-fp16-autocast-no-gradscaler` (now updated to note the STL path bites too).
2. **`.venv/bin/python` hardcoded for child subprocesses** — `scripts/closing_data/mac_baseline_compare.py`
   (`PY` const, ~L45; also imported by `comparand_check2hgi_sc.py`) and `scripts/pre_freeze_gates/run_a2.py`
   (`PY` const, ~L21). The conda board has no `.venv` → instant `FileNotFoundError`. Patch: `PY = os.environ.get("BASELINE_PY") or sys.executable`
   (correct on every board — child uses the parent interpreter). Or just `export BASELINE_PY=$(which python)`.
3. **CTLE-E2E window reconstruction used an unstable sort** — `scripts/baselines/ctle_e2e.py`
   `_build_dk_ovl_windows` sorted check-ins with pandas' default `quicksort` (unstable), but the canonical
   dk_ovl builder (`data/inputs/builders.py` `generate_next_input_from_checkins`) uses a **stable
   `kind='mergesort'`**. On the rows sharing a `(userid, datetime)` (28 at FL) the tie reordered differently,
   shifting 2/1.27 M window targets → the script's own `next_category` row-alignment assert fired and CTLE-E2E
   crashed before training. Patch: `kind="mergesort"` on both sort sites (the assert is kept — the windowing
   was made faithful, not weakened). Validated: all three alignment asserts pass at FL.
4. **(operational, no code change) Step-1 (`build_overlap_probe_engine`) is non-idempotent** — it
   unconditionally re-windows ~8 G of parquet. If `output/check2hgi_dk_ovl/<state>/input/{next,next_region}.parquet`
   already exist with the expected rowcount (FL = 1,274,418), **skip it**; just (re)build the per-fold log_T
   and run the comparand.

> **Task (C) note (resolution, not a patch):** the "feature-concat control" the handoff says "needs a thin
> builder" is **already implemented and more rigorous** — it's the closed A2 control via
> `scripts/pre_freeze_gates/run_a2.py --states <st> --seeds 0 --tasks category region` (the `hgifeat` arm =
> HGI ⊕ Check2HGI's exact per-visit node features: category one-hot + hour/dow sin/cos, alignment-asserted).
> No new builder needed; HGI inputs must exist (`scripts/pre_freeze_gates/setup_hgi_inputs.py`).

## 5 · Outputs + honesty
Commit per cell: `docs/results/closing_data/baseline_compare/florida_{ctle,check2hgi_sc}.json` (NOT gitignored);
record CTLE-E2E + CSLSL numbers in `MACS_BOARD_RESULTS.md` + `RESULTS_BOARD.md §4`. **n=5 provisional**; CTLE
framed as "even in its best (E2E) form CTLE is well below ours" — never "we crushed CTLE". Do **NOT** run CA/TX
CTLE-SC (dropped) or any region-SC (quarantined).
