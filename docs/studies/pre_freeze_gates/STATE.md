# pre_freeze_gates — STATE

**Status:** SCAFFOLDED, not launched · **Machine:** A40 · **Created:** 2026-06-14
**Onboarding:** [`AGENT_PROMPT.md`](AGENT_PROMPT.md) · **Family DAG:** [`../PRE_FREEZE_PROGRAM.md`](../PRE_FREEZE_PROGRAM.md)

## Level / blocking
- Level 1 (pre-freeze gates). Blocks: `closing_data` P2 FREEZE. Runs after/with `mtl_frontier` on the A40.

## Gate queue
| Gate | Type | State | Verdict |
|---|---|---|---|
| A2 feature-concat control | interpretation | ✅ RESOLVED (M4, 2026-06-17) | **ON NULL — claim STRENGTHENED.** concat closes only 7–8% of v14 cat gap (12–18% of v11); reg inert. See `A2_RESULTS.md`. |
| A4 transductivity bound | disclosure | ✅ RESOLVED (M4, 2026-06-17), AL+FL | **ON NULL both axes.** Reg ≈0 (AL −0.33, FL −0.12pp; validated vs A2). Cat (substrate-driven axis) measured via in-coverage POI proxy ≈0 (AL +0.29 @67%, FL +0.00 @87% cov). One-paragraph defusal; no re-anchoring. Caveat: POI proxy + cold-POI/contextual residual → inductive-Check2HGI future-work. See `A4_RESULTS.md`. |
| Overlapping-windows adopt/keep | base change (effect already validated AL) | ✅ VALIDATED (A40, 2026-06-19) | **ADOPT supported.** AL n=3 cat +8.12/reg +3.57; FL seed0 cat +3.64/reg +0.62 — positive both heads both scales (scale-dependent, never negative); leak/training-length/population audits clean. See `LANE2_OVERLAP_VALIDATION.md`. P3 rebuilds sequences/log_T/folds at stride-1 (reuses v14 embeddings). |

> **NOTE (2026-06-17):** the HANDOFF said A2/A4 SYNC the canonical v14 + HGI artifacts. They were
> absent from this M4 (and HGI was missing from the rsync list), so per user direction the substrates
> were **rebuilt locally**. A2's internal contrast is self-consistent (all arms fresh on this box) and
> HGI-cat fidelity reproduced the board (26.5 vs 25.3). An SSD disconnect mid-run was recovered with no
> lost completed results (`run_a2` is resume-safe).

## Conventions
- A2/A4 are interpretation/disclosure gates — they change what the paper *claims*, not the frozen numbers,
  but still must resolve pre-freeze so the RUN_MATRIX carries the right caveats.
- Overlapping-windows default = KEEP non-overlap (internal consistency). ADOPT only with user sign-off on
  the full-base rebuild, and only pre-freeze. It is a base change, not an MTL lever (rising-tide rule).

## Artifacts & reproduction (A2/A4, M4 run 2026-06-17)
- **Verdicts/numbers:** [`A2_RESULTS.md`](A2_RESULTS.md), [`A4_RESULTS.md`](A4_RESULTS.md).
- **Scripts** (`scripts/pre_freeze_gates/`):
  - `build_hgi.py` — rebuild canonical HGI (CPU, epoch=2000, w_r=0.7) per state.
  - `setup_hgi_inputs.py` — gen HGI next/category inputs + reuse check2hgi region labels (seq row-identity asserted).
  - `postbuild_v14.py` — gen v14 (`check2hgi_design_k_resln_mae_l0_1`) next.parquet + next_region.
  - `a2_features.py` — faithful per-visit feature builder (cat one-hot + hour/dow sin/cos), **placeid-alignment-validated**.
  - `p1_region_head_ablation.py --add-visit-features` — the A2 harness hook (backward-compatible).
  - `run_a2.py` / `a2_analyze.py` / `a2_collect.py` — A2 cell matrix runner + paired Wilcoxon + metric extractor
    (cat = f1-best snapshot; reg = top10-best).
  - `a4_build.py` — train-only v14 per fold (pseudo-state preprocess + design_k; GEOID/placeid remap; preserves
    raw train-only region+POI emb as rebuild insurance).
  - `a4_eval.py` (reg) / `a4_cat_eval.py` (cat POI proxy) — both arms same device/fold; run with `INGRED_DEVICE=cpu`
    (MPS unstable on this box — added an `INGRED_DEVICE` override to `src/configs/globals.py`).
- **Substrate caveat:** HGI + v14 were ABSENT from this M4 (not in the handoff rsync list) → rebuilt locally per
  user direction. A2 contrasts are all within-harness (self-consistent); HGI-cat fidelity reproduced the board.
- **Raw cell JSONs:** A2 cells under `docs/results/P1/region_head_*_A2_*` (committed); A4 result JSONs under
  `results/pre_freeze_gates/a4/` (gitignored — numbers captured in `A4_RESULTS.md`, reproducible via the scripts).

## Decisions log
- 2026-06-19 — **Lane 1/2/3 + baselines closed on the A40** (session `study/pre-freeze-a40`).
  Lane 1: G0.1-advisory + loss-scale advisory both **null** → recipe stays v16. Lane 2: overlap **ADOPT**
  supported (table above); stride-1 leak paths (a/b/c) clean, (d) E2-chrono is a Mac-track empirical item.
  Lane 3: CA+TX v14 built (embeddings-only anchor, seed 42, A40) → **all 6 states hash-manifested**
  (`V14_HASH_MANIFEST.json`) → §0 STOP-condition lifted. Baselines: 7 INCLUDE externals implemented +
  adversarially audited (B3 READY; 6 NEEDS_FIX, cheap fixes applied) — see `closing_data/BASELINES_IMPL_AUDIT.md`.
  Pending user decision: B2a POI2Vec faithfulness. FL new-code scan (S1+S2+auto-fit) STL+MTL in flight.
- 2026-06-17 — **A2 RESOLVED** (claim strengthened: concat closes 2.4–8.3% of the v14 cat gap, AL/AZ/FL) +
  **A4 RESOLVED** (ON NULL both axes: reg ≈0, cat ≈0 via in-coverage POI proxy, AL+FL). M4 run. See results docs.
- 2026-06-14 — scaffolded. A2 from `baseline_gap_analysis.md` Tier-1; A4 from `evaluation_protocol_review.md §4.1`;
  overlapping-windows decision points at the validated memo `docs/future_works/overlapping_windows.md`.
