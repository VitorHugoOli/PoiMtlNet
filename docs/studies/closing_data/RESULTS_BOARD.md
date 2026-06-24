# RESULTS_BOARD — consolidated closing_data board results (single source of truth)

> **What this is.** The one place that aggregates the reduced-board (champion-G MTL vs STL ceilings, **seed 0 ×
> 5 folds**, gated stride-1 overlap engine `check2hgi_dk_ovl`, fp32-matched scorer) headline numbers that were
> spread across per-machine JSONs + narrative docs. Numbers are read directly from the committed result JSONs
> (paths in §3). Baselines live in [`../../baselines/`](../../baselines/) (per the established schema) — see §4.
>
> ⚠ **STATUS (2026-06-24).** CA + TX cells land via the open board PRs **#35 (H100)** / **#37 (A40)**; the
> `mtl_cv` collate-correctness check + baseline validity are under audit (**workflow `wsftdemmg`**). Each cell
> below carries a **provenance + status** marker. TX reg is **settled** (fp32 5f, 0 skips: **67.02 → +2.06
> beats**; the A40 bf16 −2.37 was a VOID Ampere-bf16 collapse, `TX_A40_BF16_NAN.md`). Reconcile against the
> audit verdict on merge.

## 1 · Part-2 headline — MTL champion-G vs dedicated STL ceilings (Δ in pp)

Champion-G = `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) + `next_gru` (cat); unweighted CE,
static_weight cw=0.75, onecycle max-lr 3e-3, geom_simple selector; fp32-matched (`r0_matched_rescore.py`).

| State | regions | STL cat | **MTL cat** | **Δcat** | STL reg | **MTL reg** | **Δreg** | precision | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **AL** | 1109 | 55.87 | **63.56** | **+7.69** ✅beats | 69.99 | 69.81 | **−0.18** ≈matches | fp32 | ✅ main |
| **AZ** | 1547 | 57.13 | **63.39** | **+6.26** ✅beats | 59.40 | 59.34 | **−0.06** ≈matches | fp32 | ✅ main |
| **FL** | 4703 | 75.15 | **79.82** | **+4.68** ✅beats | 76.71 | 77.28 | **+0.57** ✅**beats** | fp32 | ✅ main |
| **CA** | 8501 | 70.26 | **77.33** | **+7.07** ✅beats | 63.48 | 65.66 | **+2.18** ✅**beats** | bf16 | 🔶 PR #35 |
| **TX** | 6553 | 69.95 | **77.51** | **+7.56** ✅beats | 64.96 | **67.02** | **+2.06** ✅**beats** | fp32 | ⏳ A40 PR |
| **Istanbul** | 520 (mahalle) | — | — | **+8.06** ✅beats | — | — | **−0.58** ≈matches | fp32 (4 seeds) | 🔶 PR #33-merged, GCN substrate |

**Reading (the story, on real data):** MTL **beats the dedicated category ceiling at every state** (+4.7 … +7.7 pp)
AND **beats-or-matches the region ceiling everywhere — including the largest states** (FL **+0.57**, CA **+2.18**;
AL/AZ within ≈0.2; TX **+2.06**). The earlier fp16-autocast collapse (`CA_MTL_DIVERGENCE.md`)
was **masking a genuine region win** — corrected precision flips the old "MTL sacrifices region" into Pareto-positive.

> Per-fold arrays + best-epochs are in the source JSONs (§3). All cat best-epochs are late (ep16-50) and CA/TX
> are healthy to ep49-50 under bf16/fp32 (no ep30 collapse). The **TX bf16 run NaN-collapsed on A40-Ampere**
> (backward-pass grad NaN, device-specific — not the fp16 overflow); the **fp32 re-run is the clean number**
> (`TX_A40_BF16_NAN.md`); H100 ran the same TX cell clean (reg 67.13).

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
- `h100/california_s0_mtl/california_s0_mtl_final_score.json` — CA MTL final 5f (**PR #35** → main on merge)
- `a40/tx_stl_cat_ceiling_s0.json` (69.95) · `a40/tx_stl_reg_ceiling_s0.json` (64.96) — TX ceilings (**PR #37/main**)
- `a40/tx_ba2_fp32_s0.json` — **TX MTL fp32 5f (CLEAN, 0 skips): reg 67.02 / +2.06, cat 77.51 / +7.56** (A40 PR). bf16 (`tx_ba2_bf16_s0.json`) is VOID (Ampere collapse)
- STL **reg** ceilings: `docs/results/P1/region_head_*_dkovl*` (fp32, leak-free per-fold prior)

**Narrative / per-cell docs:** `docs/studies/closing_data/`
- `BOARD_H100_FINDINGS.md` (session consolidation) · `CA_CELL.md` · `TX_CELL.md` · `AL_PRECISION_GATE.md` ·
  `AZ_CELL.md` · `FL_PRECISION_GATE.md` · `EP100_ABLATION_AND_TX_RAM.md` · `TX_A40_BF16_NAN.md` ·
  `CA_MTL_DIVERGENCE.md` (the fp16 root cause) · `MACS_BOARD_RESULTS.md` (baselines, **PR #36**)

**Floors:** `docs/results/P0/simple_baselines/<state>/` (Markov-1 region Acc@10: AL .470 AZ .430 FL .650 CA .521 TX .549).

## 4 · Baselines → `docs/baselines/` (separate home, established schema)
Per the baselines README, the paper's baseline tables read from [`../../baselines/`](../../baselines/)
(`next_category/` + `next_region/`, each with per-baseline `.md` + `results/<state>.json` + `comparison.md`).
The **new board baselines** (CTLE-SC, HMT-GRN, b2b/b2c/poi2vec SC, native-E2E) are being consolidated there from
`docs/results/closing_data/baseline_compare/*.json` + `MACS_BOARD_RESULTS.md` (PR #36) **after the audit verdict +
merge** (the SC **reg** numbers were flagged invalid pending re-run; HMT-GRN is device-labeled `[M4/MPS]`). Early
signal (to be finalized): substrate drives next-cat — **Check2HGI-SC +37.8/+37.0 pp over CTLE-SC at AL/AZ**;
HMT-GRN (sole region-native) reg 62.37 at AL, **below** our MTL ~69.8.

## 5 · Provenance legend
✅ main = source JSON on main, verified-readable · 🔶 PR #N = committed on a board branch, lands on merge ·
⏳ = run in-flight, number not final. All cells re-confirmed against the `wsftdemmg` audit on merge.
