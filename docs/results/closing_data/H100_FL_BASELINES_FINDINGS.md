# H100 FL baseline block — findings (2026-06-25)

All cells **seed 0 × 5 folds, fp32** (`DISABLE_AMP=1` — the #43 fix; the CUDA single-task trainer
NaN-collapses in fp16 at FL scale). Raw JSONs committed in PR #45. This doc settles the numbers +
readings; it does NOT restate the canonical board (see `RESULTS_BOARD.md`).

## 1 · FL representation block (role-2)

| Cell | base | cat macro-F1 | reg Acc@10 | JSON |
|---|---|---|---|---|
| **Check2HGI-SC comparand** | dk_ovl | **73.47 ± 0.55** | 72.71 ± 1.14 | `baseline_compare/florida_check2hgi_sc.json` |
| **CTLE-E2E** (B1, headline CTLE) | dk_ovl | **29.65** | 61.47 | `results/ctle_e2e_b1/florida/ctle_e2e_seed0.json` |
| **CTLE-SC** (frozen) | dk_ovl | _pending_ (M4-cleared; expect ~17–20, frozen-below-floor) | _pending_ | `baseline_compare/florida_ctle.json` |
| **A2 feature-concat** hgi → hgifeat | hgi set-a | 32.01 → 32.85 | 70.28 → 70.05 | `P1/region_head_florida_*_A2_*_s0.json` |

**Readings**
- **CTLE is well below ours even in its best (E2E, fine-tuned) form**: cat 29.65 vs our comparand 73.47
  (Δ ≈ −44 pp). Frozen CTLE-SC is weaker still (the M4 AL diagnosis confirmed the below-bigram-floor signal
  is real CTLE behaviour, not a pipeline bug). Frame as "even at its best, CTLE ≪ ours" — never "we crushed it".
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

| Istanbul | cat macro-F1 | reg Acc@10 |
|---|---|---|
| **MTL champion (stride-1)** | **59.73 ± 0.62** | **74.30 ± 0.72** |
| STL reg ceiling (stride-1) | — | 74.80 ± 0.70 |
| MTL champion (set-a, ref) | 60.15 | 69.79 |
| STL reg ceiling (set-a, ref) | — | 70.37 |

**Reading — the champion-G signature HOLDS at stride-1.** MTL reg 74.30 ≈ STL reg ceiling 74.80 (**−0.50,
parity**, same as set-a's −0.58), and cat clears its ceiling. The **+4.5 pp reg vs set-a (69.79 → 74.30) is the
overlap-window density lift applied to BOTH the STL ceiling and the MTL champion — NOT a leak** (per-fold log_T
rebuilt on the stride-1 split; user-disjoint folds). JSONs: `second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json`,
`P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json`.

## Provenance / caveats
- fp32 everywhere (`DISABLE_AMP=1`); n=5 seed-0 provisional.
- set-a fence ≠ canonical dk_ovl cascade (A40). Istanbul on H100 carries a cross-device caveat vs its M4 baselines.
- Cascade/Istanbul MTL scored with `scripts/closing_data/h100_score_matched.py` (cat=macro-F1 diag-best;
  reg=FULL top10_acc indist-best, per-task diagnostic-best, fold-mean).
