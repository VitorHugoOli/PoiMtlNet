# RESULTS_BOARD — consolidated closing_data board results (single source of truth)

> **What this is.** The one place that aggregates the reduced-board (champion-G MTL vs STL ceilings, **seed 0 ×
> 5 folds**, gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32-matched scorer) headline numbers that were
> spread across per-machine JSONs + narrative docs. Numbers are read directly from the committed result JSONs
> (paths in §3). Baselines live in [`../../baselines/`](../../baselines/) (per the established schema) — see §4.
>
> ✅ **STATUS (2026-06-24, post-merge + audit `wsftdemmg`).** PRs #35/#36/#37 **MERGED** — all cells below are on
> main. The committed board MTL numbers are **verified trustworthy** (the `mtl_cv` collate bug was a test-fixture
> hard-crash, NOT silent corruption; production path byte-identical at AL — fixture fixed in `077ff136`). **CA is
> 5f-COMPLETE and beats both ceilings — the load-bearing fact that retires the old "region cost grows with
> cardinality" narrative.** Remaining caveat: **TX MTL reg is in-flight (2/5 clean folds)**; cite it as such.

## 1 · Part-2 headline — MTL champion-G vs dedicated STL ceilings (Δ in pp)

Champion-G = `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) + `next_gru` (cat); unweighted CE,
static_weight cw=0.75, onecycle max-lr 3e-3, geom_simple selector; fp32-matched (`r0_matched_rescore.py`).

| State | regions | STL cat | **MTL cat** | **Δcat** | STL reg | **MTL reg** | **Δreg** | precision | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **AL** | 1109 | 55.87 | **63.56** | **+7.69** ✅beats | 69.99 | 69.81 | **−0.18** ≈matches | fp32 | ✅ main |
| **AZ** | 1547 | 57.13 | **63.39** | **+6.26** ✅beats | 59.40 | 59.34 | **−0.06** ≈matches | fp32 | ✅ main |
| **FL** | 4703 | 75.15 | **79.82** | **+4.68** ✅beats | 76.71 | 77.28 | **+0.57** ✅**beats** | fp32 | ✅ main |
| **CA** | 8501 | 70.26 | **77.33** | **+7.07** ✅beats | 63.48 | 65.66 | **+2.18** ✅**beats** | bf16 | ✅ main (5f) |
| **TX** | 6553 | 69.95 | **77.50** ²ᐟ⁵ | **+7.55** ✅beats | 64.96 | **67.13** ²ᐟ⁵ | **+2.17** ✅beats ⏳ | fp32(H100) | ⏳ 2/5 folds |
| **Istanbul** | 520 (mahalle) | 52.10 | **60.16** | **+8.06** ✅beats | 70.37 | 69.79 | **−0.58** ≈matches | fp32 (4 seeds) | ✅ main, GCN substrate |

**Reading (the story, on real data):** MTL **beats the dedicated category ceiling at every state** (+4.7 … +8.1 pp)
AND **beats the region ceiling at the LARGE region counts** (FL 4.7k **+0.57**, CA 8.5k **+2.18** — both 5f;
TX 6.6k **+2.17** on 2/5 clean folds), while **matching within δ=2 pp at the small counts** (AL −0.18, AZ −0.06,
Istanbul −0.58). **CA, the largest region state, is 5f-complete and beats** — that single cell retires the old
"region cost grows with cardinality" (Decision-C) narrative. The earlier fp16-autocast collapse
(`CA_MTL_DIVERGENCE.md`) + the A40-Ampere bf16 grad-NaN were **masking a genuine region win**.

> **Honest framing:** "beats on **category** everywhere; beats on **region** at the large states, matches at the
> small." Do NOT write "beats region everywhere" (AL/AZ/Istanbul are matches-within-margin, slightly negative).
>
> **Caveats that MUST travel with these numbers:**
> 1. **TX is 2/5 clean folds** (H100 `texas_s0_mtl_partial_score.json`, reg 67.13 / cat 77.50, ep49-50, 0 NaN) —
>    label in-flight, never print a 5f mean. The 5f-complete TX cat (75.87, Δ+5.92) is from the **reg-VOID bf16**
>    run; the clean companion to reg 67.13 is cat 77.50.
> 2. **TX Δreg mixes device classes** — A40-fp32 ceiling 64.96 vs H100 MTL 67.13. Fix before a regular-track
>    claim: finish the A40-fp32 TX MTL 5f (same lane as the ceiling) OR re-run the TX reg ceiling on H100.
> 3. **VOID / stale — never cite:** `california_s0_{board,mtl}_partial.json` (fp16 collapse −5.22, superseded by
>    the clean 65.66); TX `tx_ba2_bf16_s0.json` (−2.37) + old fp16 (−2.41) — 74,812 skipped steps, reg best-ep 4-5.
> 4. **n=5 (seed 0 only):** Wilcoxon superiority (cat) is fine (p-floor 0.0312); region matches/beats need a
>    **TOST-power statement** or an "n=5 provisional" label. The {1,7,100} top-up to n=20 is post-deadline.

## 2 · Precision verdict (settled) & schedule ablation (NULL)
- **bf16 ≈ fp32** on quality (Δ≤0.12 pp) and ~0 wall-clock (overlap is data-bound, GPU util 8-25%) →
  small/mid states fp32; large-state bf16 is **not cross-GPU portable** (A40-Ampere grad-NaNs where H100 stays
  finite) → **use fp32 for large-state cells on Ampere**. (`AL_PRECISION_GATE.md`, `FL_PRECISION_GATE.md`, `TX_A40_BF16_NAN.md`.)
- **100-epoch schedule = NULL** (AL cat +0.21/reg −0.39; FL cat −0.53/reg −0.18; OneCycle best-val rides the
  anneal tail at any length) → **frozen 50ep cells stand.** (`EP100_ABLATION_AND_TX_RAM.md`.)

## 3 · File map — where every result lives (the de-spread index)
**MTL + STL matched-score JSONs:** `docs/results/closing_data/`
- `h100/{alabama,arizona}_s0_mtl_fp32_matched_score.json` · `florida_s0_mtl_fp32_5f_matched_score.json` — MTL (main)
- `h100/{alabama,arizona,florida,california}_s0_stl_cat_ceiling.json` — STL cat ceilings (main)
- `h100/california_s0_mtl/california_s0_mtl_final_score.json` — CA MTL final 5f (cat 77.33/reg 65.66) ✅main
- `a40/tx_stl_cat_ceiling_s0.json` (69.95) · `a40/tx_stl_reg_ceiling_s0.json` (64.96) — TX ceilings ✅main
- `h100/texas_s0_mtl/texas_s0_mtl_partial_score.json` — TX MTL **2/5 clean folds** (cat 77.50/reg 67.13) ✅main
- ⚠ VOID: `h100/california_s0_{board,mtl}_partial.json`, `a40/tx_ba2_bf16_s0.json` (fp16/bf16 collapse — do not cite)
- STL **reg** ceilings: `docs/results/P1/region_head_*_dkovl*` (fp32, leak-free per-fold prior)

**Narrative / per-cell docs:** `docs/studies/closing_data/`
- `BOARD_H100_FINDINGS.md` (session consolidation) · `CA_CELL.md` · `TX_CELL.md` · `AL_PRECISION_GATE.md` ·
  `AZ_CELL.md` · `FL_PRECISION_GATE.md` · `EP100_ABLATION_AND_TX_RAM.md` · `TX_A40_BF16_NAN.md` ·
  `CA_MTL_DIVERGENCE.md` (the fp16 root cause) · `MACS_BOARD_RESULTS.md` (baselines, **PR #36**)

**Floors:** `docs/results/P0/simple_baselines/<state>/` (Markov-1 region Acc@10: AL .470 AZ .430 FL .650 CA .521 TX .549).

## 4 · Baselines → `docs/baselines/` (separate home, established schema)
Per the baselines README, the paper's baseline tables read from [`../../baselines/`](../../baselines/)
(`next_category/` + `next_region/`, each with per-baseline `.md` + `results/<state>.json` + `comparison.md`).
**Which baselines the ARTICLE uses (decided 2026-06-24) — the SC distinction matters:**
- ✅ **CTLE-SC (category) IS used — it is the reviewer W3 novelty gate** (`REVIEW_PANEL.md` required-change #2,
  *non-negotiable*: "score CTLE leak-clean and show it loses to Check2HGI attributable to the hierarchy"). It is
  the **representation-isolation** comparison (CTLE-emb → our head vs Check2HGI-emb → our head, matched capacity)
  that substantiates contribution-1 novelty; without it C1's "novel" *evaporates*. **Δcat AL +37.8 / AZ +37.0 /
  Istanbul +28.6**, device-internal-clean (reproduces CUDA within noise). Leak-clean at **AL/AZ + Istanbul
  (stride-1, PR #38)**; FL/CA/TX → CUDA (`BASELINE_H100.md`). POI2Vec/skip-gram/one-hot SC-cat are the
  representation **controls** (§7 checklist).
- ❌ **SC *region* is NOT used in the article.** The pre-fix SC reg was INVALID (substrate-bypass + shared prior +
  stale log_T) — now **quarantined** (`_reg_status: INVALID_PENDING_RERUN` on the AL/AZ `baseline_compare/*.json`).
  Region's substrate-isolation story is weak anyway (it is a near-tie: AL −0.4, AZ −0.3, Istanbul −3.5 where CTLE
  edges ahead on the small 520-region space). **The article's REGION baselines come from native-E2E (HMT-GRN, STAN)
  + Markov-1**, which ARE substrate-sensitive — not from SC. (SC-reg re-run is optional, not article-blocking.)
- ✅ **HMT-GRN (region native-E2E): Mac numbers are CORRECT — AUDITED & CITABLE.** PR #38: the AL gap (Mac 57.05 vs
  recorded M2 62.37) was traced — byte-identical code/base/seed, all leak paths train-only (a leak can't create a
  gap). **Decisive: AL HMT on deterministic CPU reg 56.99 ≈ M4-MPS 57.05 within 0.06 pp, fold-for-fold** → the Mac
  value is correct; the **62.37 is the outlier** (unreproducible, artifacts reclaimed). Use the Mac/CPU HMT-GRN reg:
  **AL 57.1 / AZ 43.7 / FL 63.7 / CA 49.6 / TX 53.9 / Istanbul 60.4** — all **well below our MTL ~65-69**, so we beat
  the sole region-native baseline by a wide margin (all 6 states now done). (Re-verify the old 62.37, not the Mac value.)

## 5 · Provenance legend
✅ main = source JSON on main, verified-readable · ⏳ 2/5 = run in-flight, partial folds (not a 5f mean) ·
⚠ VOID = fp16/bf16-collapse artifact, never cite. All numbers audited against `wsftdemmg` (collate-trust verdict +
adversarial source-verification). Honest headline: **beats on category everywhere; beats on region at the large
states (CA/FL 5f, TX 2/5), matches at the small (AL/AZ/Istanbul within δ=2 pp).**
