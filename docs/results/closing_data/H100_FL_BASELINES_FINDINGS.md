# H100 FL baseline block — findings (2026-06-25)

All cells **seed 0 × 5 folds, fp32** (`DISABLE_AMP=1` — the #43 fix; the CUDA single-task trainer
NaN-collapses in fp16 at FL scale). Raw JSONs committed in PR #45. This doc settles the numbers +
readings; it does NOT restate the canonical board (see `RESULTS_BOARD.md`).

## 1 · FL representation block (role-2)

| Cell | base | cat macro-F1 | reg Acc@10 | JSON |
|---|---|---|---|---|
| **Check2HGI-SC comparand** | dk_ovl | **73.47 ± 0.55** | 72.71 ± 1.14 | `baseline_compare/florida_check2hgi_sc.json` |
| **CTLE-E2E** (B1, headline CTLE) | dk_ovl | **29.65** | 61.47 | `results/ctle_e2e_b1/florida/ctle_e2e_seed0.json` |
| **CTLE-SC** (frozen) | dk_ovl | **27.98** (fold-0 provisional; full 5f running) | **73.00** (fold-0) | `baseline_compare/florida_ctle.json` |
| **A2 feature-concat** hgi → hgifeat | hgi set-a | 32.01 → 32.85 | 70.28 → 70.05 | `P1/region_head_florida_*_A2_*_s0.json` |

**Readings**
- **CTLE is well below ours even in its best (E2E, fine-tuned) form**: cat 29.65 vs our comparand 73.47
  (Δ ≈ −44 pp). Frozen CTLE-SC is weaker still (the M4 AL diagnosis confirmed the below-bigram-floor signal
  is real CTLE behaviour, not a pipeline bug). Frame as "even at its best, CTLE ≪ ours" — never "we crushed it".
- **CTLE-SC (frozen) — the cat/reg split is the honest two-axis story** (fold-0 provisional; full 5f running):
  **cat 27.98 ≪ our comparand 73.47** (the *substrate* drives category — frozen CTLE can't compete), but
  **reg Acc@10 73.00 ≈ comparand 72.71 — a near-tie** (region is driven by the per-fold log_T prior, which is
  substrate-independent, exactly as §3 predicts). So CTLE-SC confirms: cat is where Check2HGI wins; region is a
  near-tie across substrates.
- **The Check2HGI cat lift is hierarchical-infomax learning, not feature injection** (A2): HGI ⊕ Check2HGI's
  exact per-visit node features (category one-hot + hour/dow sin/cos) lifts cat only **+0.84 pp** (32.01 → 32.85),
  closing just **~9 %** of the gap to the v14 substrate (40.81). Reg is inert (70.28 → 70.05, NS; HGI leads
  region anyway). Consistent with the closed AL/AZ A2 (7–8 %).

## 2 · CSLSL cascade @ FL — the **set-a FENCE** (role-3 cross-check)

> The **canonical** FL cascade is on **dk_ovl** and was re-homed to the A40 (`RESULTS_BOARD §1b`, #44 — a
> dead-tie). This H100 cell is the **set-a fence**: the v14 `design_k` set-a base (stride-9 non-overlap),
> completing the FL row M4's MPS run OOM'd on. **Do NOT compare across** to the dk_ovl champion.

| arm (v14 set-a) | cat macro-F1 | reg Acc@10 |
|---|---|---|
| cascade (directed cat→region, cross-attn severed) | 72.00 ± 0.83 | 72.63 ± 0.71 |
| parallel (champion-G, cross-attn ON) | 72.08 ± 0.87 | 73.05 ± 0.70 |
| **Δ (parallel − cascade)** | **+0.08 (tie)** | **+0.42** |

**Reading:** at FL the cascade is a **dead tie** with the parallel champion-G (cat +0.08, reg +0.42) —
consistent with the canonical dk_ovl #44 finding. (M4's small-state AL/AZ set-a showed parallel > cascade
on cat; the gap shrinks at FL scale.) JSON: `baseline_compare/florida_cslsl_cascade.json`.

## 3 · Istanbul champion-G @ stride-1 (§6.3 windowing-consistency)

Run on the **H100 per the user** (docs nominally assign Istanbul to the M4 for same-device-Δ — **cross-device
caveat: footnote the §6.3 Istanbul row**). Base rebuilt at stride-1 (271,666 rows / 520 mahalle, overwriting the
set-a base; recoverable). CLI = the `DRY_RUN_RESULTS.md`-verified champion (incl. `--mtl-loss static_weight
--category-weight 0.75`, which the truncated PHASE_V command had dropped). fp32, seed 0 × 5f.

Ran **multi-seed {0,1,7,100} (n=20)** to match the set-a rigour, plus the stride-1 STL cat + reg ceilings.

| Istanbul stride-1 | cat macro-F1 | reg Acc@10 |
|---|---|---|
| **MTL champion (n=20, 4 seeds)** | **59.89 ± 0.12** | **74.28 ± 0.03** |
| &nbsp;&nbsp;per-seed cat / reg | 59.73 / 60.01 / 59.82 / 59.99 | 74.30 / 74.24 / 74.31 / 74.29 |
| **STL ceiling (seed 0)** | 53.20 ± 0.73 | 74.80 ± 0.70 |
| **Δ (MTL − ceiling)** | **+6.69 (gain)** | **−0.52 (parity)** |
| _set-a multi-seed (ref)_ | _60.16_ | _69.79_ |

**Reading — the champion-G signature HOLDS at stride-1, n=20.** cat clears its STL ceiling by **+6.69**
(gain); reg sits at **−0.52** vs its ceiling (parity). Cross-seed variance is tiny (cat ±0.12 / reg ±0.03) →
highly reproducible. vs set-a (cat 60.16 / reg 69.79): cat ~flat, **reg +4.5 is the overlap-window density lift
applied to BOTH the STL ceiling AND the MTL champion — NOT a leak** (per-fold log_T rebuilt on the stride-1
split per seed; user-disjoint folds). JSONs: `second_dataset/istanbul/istanbul_stride1_multiseed_summary.json`,
`istanbul_stride1_s{0,1,7,100}_mtl_fp32_matched_score.json`, `istanbul_stride1_s0_stl_cat_ceiling.json`,
`P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json`.

## Provenance / caveats
- fp32 everywhere (`DISABLE_AMP=1`); n=5 seed-0 provisional.
- set-a fence ≠ canonical dk_ovl cascade (A40). Istanbul on H100 carries a cross-device caveat vs its M4 baselines.
- Cascade/Istanbul MTL scored with `scripts/closing_data/h100_score_matched.py` (cat=macro-F1 diag-best;
  reg=FULL top10_acc indist-best, per-task diagnostic-best, fold-mean).
- **CTLE-SC parallelization (method):** the CTLE-SC compare rebuilds the full 1.27 M-row inputs + frozen
  substrate *per fold*, so sequential 5-fold = ~2.5 h. It now runs **5 single-fold compares in parallel, each in
  an isolated `OUTPUT_DIR=output_ctle_f{f}`** (with a `check2hgi` symlink to the shared canonical substrate;
  cells/`results` shared read-only) → ~1 fold's wall-clock. Required two `mac_baseline_compare.py` fixes:
  `OUT` made `OUTPUT_DIR`-aware (it was hardcoded `output/`, which would split staging from training under an
  override) + a `--only-fold` flag (writes `florida_ctle_f{f}.json`, aggregated afterward). **Leak-safety is
  unchanged** — each fold still pairs its own cell `s0_f{f}` + split + per-fold log_T. Runner: `scripts/run_taskD_parallel_fl.sh`.
