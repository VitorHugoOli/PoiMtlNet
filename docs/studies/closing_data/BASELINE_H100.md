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

**(D) CTLE-SC @ FL** — ⚠ **GATED on the M4 CTLE-diagnosis** (the recorded frozen CTLE-SC cat AL 17.8 is *below*
the bigram floor; the M4 confirms whether that's a real CTLE weakness or a pipeline bug — do NOT run until the
M4 posts its verdict).
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

## 5 · Outputs + honesty
Commit per cell: `docs/results/closing_data/baseline_compare/florida_{ctle,check2hgi_sc}.json` (NOT gitignored);
record CTLE-E2E + CSLSL numbers in `MACS_BOARD_RESULTS.md` + `RESULTS_BOARD.md §4`. **n=5 provisional**; CTLE
framed as "even in its best (E2E) form CTLE is well below ours" — never "we crushed CTLE". Do **NOT** run CA/TX
CTLE-SC (dropped) or any region-SC (quarantined).
