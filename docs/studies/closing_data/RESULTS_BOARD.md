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
> cardinality" narrative.** **TX is now 5f-COMPLETE too** (A40 true-fp32, single-device, 0 skips; +2.06 reg / +7.56 cat — #41); H100 bf16 corroborates to 0.03 pp (#39). **Every board cell is settled; nothing in-flight.**

## 1 · Part-2 headline — MTL champion-G vs dedicated STL ceilings (Δ in pp)

> 🏆 **CHAMPION — `--canon v17` (`DEFAULT_CANON`):** **bs=8192 + cat-lr 1e-3** via `--onecycle-per-head-lr`. The
> per-head cat-LR is a **STATE-SIZE trade** (not a strict board-wide cat win — kept board-wide by user decision
> 2026-07-01 for the small-state gains + single-champion simplicity; reg is neutral+ everywhere):
>
> | state (regions) | v17 cat | comparand cat | Δcat | v17 reg | reg Δ | source |
> |---|---|---|---|---|---|---|
> | AL (1109) | **64.54** | 63.56 | **+0.99** | 69.80 | ≈ | n=20 `perhead_lr_n20.md` |
> | AZ (1547) | **65.84** | 63.39 | **+2.45** | 59.56 | ≈ | n=20 `perhead_lr_n20.md` |
> | FL (4703) | **79.85** | 79.68 (n=20 base) | **+0.17** | 77.42 | **+0.20** | n=20 `perhead_lr_n20.md` |
> | CA (8501) | 77.04±0.20 | 77.33 (bf16 board) | **−0.29** | 65.69±0.30 | **+0.03** | s0-5f `catx_v17_seed0_5f/RESULTS.md` |
> | TX (6553) | 77.23±0.12 | 77.51 (**fp32** board) | **−0.28** | 67.07±0.45 | **+0.05** | s0-5f `catx_v17_seed0_5f/RESULTS.md` |
>
> **Honest read:** v17's cat lever **wins at small/mid states (+0.2…+2.5)** but **costs ~0.28 pp cat at the two
> largest states (CA/TX)**, while **reg ties/beats everywhere**. The large-state dip is **real, not a bf16 artifact**
> — TX's board is fp32 (clean same-precision) and still −0.28 (~2σ), matching CA's −0.29. §1 below stays the **board
> of record** (n=5 seed-0) until CA/TX land at **n=20 on the H100** ({1,7,100}; `run_catx_v17_n20_h100.sh`), which
> firms the large-state Δcat significance. v17 remains `DEFAULT_CANON`; §1 headline updates after the H100 n=20 +
> the flag-OFF parity test.

Champion-G = `mtlnet_crossattn_dualtower` + `next_stan_flow_dualtower` (reg) + `next_gru` (cat); unweighted CE,
static_weight cw=0.75, onecycle max-lr 3e-3, geom_simple selector; fp32-matched (`r0_matched_rescore.py`).

| State | regions | STL cat | **MTL cat** | **Δcat** | STL reg | **MTL reg** | **Δreg** | precision | status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **AL** | 1109 | 55.87 | **63.56** | **+7.69** ✅beats | 69.99 | 69.81 | **−0.18** ≈matches | fp32 | ✅ main |
| **AZ** | 1547 | 57.13 | **63.39** | **+6.26** ✅beats | 59.40 | 59.34 | **−0.06** ≈matches | fp32 | ✅ main |
| **FL** | 4703 | 75.15 | **79.82** | **+4.68** ✅beats | 76.71 | 77.28 | **+0.57** ✅**beats** | fp32 | ✅ main |
| **CA** | 8501 | 70.26 | **77.33** | **+7.07** ✅beats | 63.48 | 65.66 | **+2.18** ✅**beats** | bf16 | ✅ main (5f) |
| **TX** | 6553 | 69.95 | **77.51** | **+7.56** ✅beats | 64.96 | **67.02** | **+2.06** ✅**beats** | fp32 | ✅ main (5f) |
| **Istanbul** | 520 (mahalle) | 53.20 | **59.89** | **+6.69** ✅beats | 74.80 | 74.28 | **−0.52** ≈matches | fp32 (**n=20**, 4 seeds) | ✅ main, stride-1 GCN |

**Reading (the story, on real data):** MTL **beats the dedicated category ceiling at every state** (+4.7 … +7.7 pp)
AND **beats the region ceiling at the LARGE region counts** (FL 4.7k **+0.57**, CA 8.5k **+2.18** — both 5f;
TX 6.6k **+2.06** — all 5f), while **matching within δ=2 pp at the small counts** (AL −0.18, AZ −0.06,
Istanbul −0.52). **CA, the largest region state, is 5f-complete and beats** — that single cell retires the old
"region cost grows with cardinality" (Decision-C) narrative. The earlier fp16-autocast collapse
(`CA_MTL_DIVERGENCE.md`) + the A40-Ampere bf16 grad-NaN were **masking a genuine region win**.

> **Honest framing:** "beats on **category** everywhere; beats on **region** at the large states, matches at the
> small." Do NOT write "beats region everywhere" (AL/AZ/Istanbul are matches-within-margin, slightly negative).
>
> **Caveats that MUST travel with these numbers:**
> 1. **TX is SETTLED 5f** —
>    **SETTLED 5f** (A40 true-fp32 `tx_ba2_fp32_s0.json`: reg 67.02 / cat 77.51, 0 skips, late best-ep [48-50]).
>    The old 75.87 cat came from the reg-VOID bf16 run — superseded; use the fp32 5f.
> 2. **TX Δreg is now single-device** — A40-fp32 MTL 67.02 vs A40-fp32 ceiling 64.96 (both A40) → +2.06 clean.
>    (H100 bf16 67.00 corroborates cross-device to 0.03 pp.) The earlier device-mix caveat is RESOLVED.
> 3. **VOID / stale — never cite:** `california_s0_{board,mtl}_partial.json` (fp16 collapse −5.22, superseded by
>    the clean 65.66); TX `tx_ba2_bf16_s0.json` (−2.37) + old fp16 (−2.41) — 74,812 skipped steps, reg best-ep 4-5.
> 4. **n=5 (seed 0 only):** Wilcoxon superiority (cat) is fine (p-floor 0.0312); region matches/beats need a
>    **TOST-power statement** or an "n=5 provisional" label. The {1,7,100} top-up to n=20 is post-deadline.

## 1b · CSLSL cascade (role-3 baseline) — cascade-vs-parallel, same-device A40

CSLSL/CatDM cascade pattern (`scripts/baselines/b4_cascade.py`): a directed **cat→region** edge (posterior
softmax, `cond_detach` feed-forward) with the symmetric cross-attention **DISABLED** — isolating
**cascade vs parallel** as the ONLY varying factor vs champion-G (identical frozen heads `next_gru`(cat) +
`next_stan_flow_dualtower`(reg), identical `check2hgi_dk_ovl` substrate + per-fold seeded log_T). Seed 0 × 5f,
gated stride-1 overlap (MIN_SEQ=10), **true fp32** (`MTL_DISABLE_AMP=1`, 0 non-finite skips). Comparand =
**champion-G re-run on the SAME A40** for a strict same-device Δ (the §1 champion-G is on the H100).

| State | cascade cat | champ-G cat (A40) | **Δcat** | cascade reg | champ-G reg (A40) | **Δreg** | cascade joint | champ-G joint | **Δjoint** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **AL** | 63.45 ±2.00 | 63.25 ±2.02 | **+0.20** | 69.48 ±3.03 | 69.65 ±3.32 | **−0.17** | 66.39 | 66.37 | **+0.02** ≈tie |
| **AZ** | 63.63 ±1.34 | 63.44 ±1.33 | **+0.20** | 59.18 ±1.83 | 59.36 ±1.79 | **−0.18** | 61.37 | 61.36 | **+0.00** ≈tie |
| **FL** | 79.83 ±0.49 | 79.82 (H100§1) | **+0.01** | 77.27 ±0.95 | 77.28 (H100§1) | **−0.01** | 78.54 | 78.55 | **−0.01** ≈tie |

*(cat = macro-F1; reg = FULL top10_acc = indist·(1−ood) at diagnostic-best epoch; joint = √(cat·reg);
fold-mean ±pstd, matched scorer `a40_score_matched.py`. JSONs:
`docs/results/closing_data/a40/{al,az,fl}_cascade_s0.json` + `{al,az}_champG_a40_s0.json`.
FL champ-G comparand = the §1 board (H100) value — the A40 same-device champ-G FL was **stopped at 4/5 folds**
(re-tasked to W6) after its 4-fold mean (cat 79.596 / reg 76.825, `a40/fl_champG_a40_4f_partial_s0.json`)
reproduced the board 4-fold mean to **±0.006 cat / ±0.16 reg**, so the FL tie rests on the board champ-G,
cross-device ±0.01.)*
**FL (large state, 4703 regions): cascade is again a tie.** vs the §1 board champ-G FL (H100, 79.82/77.28):
**Δcat +0.01 / Δreg −0.01** — essentially identical. FL canonical (`dk_ovl`, 5f, fp32) —
**supersedes the M4 set-a partial** (`baseline_compare/florida_cslsl_cascade.json`, 4-fold MPS-OOM, no comparand).

**Reading:** the cascade is a **dead tie** with our parallel champion-G on the joint objective
(Δjoint ≤ 0.02 pp ≪ fold-std). It trades a hair of category (+0.20) for a hair of region (−0.17/−0.18),
netting ~0 → **our parallel bidirectional cross-attention matches the dominant published multi-task
alternative (cascade) at equal cost**; neither the directed cat→region coupling nor severing the symmetric
channel helps or hurts materially at AL/AZ. **n=5 provisional** (seed 0; {1,7,100}→n=20 post-deadline).
Cascade did NOT beat champion-G (the `b4_cascade.py` wiring sanity check passes). **Cross-device check:**
the A40 champion-G re-run reproduces the board (H100) champion-G — AZ cat/reg within ±0.05 pp,
AL cat −0.31 / reg −0.16 (≤ fold-std). CA/TX cascade deferred (deadline; CA/TX only "if cheap" per handoff).

## 1c · W6 — category-side encoder-isolation probe (mechanism: trunk, not transfer)

Freeze the **region** stream at init (`--freeze-reg-stream`) + reg loss off (`--category-weight 1.0`), train
champion-G on `check2hgi_dk_ovl`, read the **category** head (seed 0 × 5f, true fp32, A40). Tests the §6.2
claim directly (the cat win is "a stronger shared encoder, NOT region→category transfer") — the right direction
vs F49 (which freezes the cat stream).

| State | STL cat ceiling | full-MTL cat (§1) | **probe cat (region frozen)** | Δ vs ceiling | Δ vs full-MTL |
|---|---:|---:|---:|---:|---:|
| AL | 55.87 | 63.56 | **63.50 ±1.74** | **+7.63** | −0.06 |
| AZ | 57.13 | 63.39 | **63.67 ±1.28** | **+6.54** | +0.28 |
| FL | 75.15 | 79.82 | **79.79 ±0.46** | **+4.64** | −0.03 |

**Verdict: W6 CLOSED.** With the region stream frozen-at-init (no learned region signal, cannot co-adapt via
cross-attn K/V), the category head keeps the **entire** joint lift — probe cat ≈ full-MTL cat (±0.3 pp) and
**≫ STL ceiling (+4.6…+7.6 pp)** at all three states. → the category benefit is the **shared trunk
(architecture/capacity), NOT cross-task transfer**. Freeze verified (reg optimizer group = 0 trainable params,
all states; 0 nan). **n=5 provisional.** Full cell: `W6_ENCODER_ISOLATION.md`; JSONs
`a40/{al,az,fl}_w6_freezereg_s0.json`.

## 2 · Precision verdict (settled) & schedule ablation (NULL)
- **bf16 ≈ fp32** on quality (Δ≤0.12 pp) and ~0 wall-clock (overlap is data-bound, GPU util 8-25%) →
  small/mid states fp32; large-state bf16 is **not cross-GPU portable** (A40-Ampere grad-NaNs where H100 stays
  finite) → **use fp32 for large-state cells on Ampere**. (`BOARD_CELLS.md` AL/FL gates, `TX_A40_BF16_NAN.md`.)
- **100-epoch schedule = NULL** (AL cat +0.21/reg −0.39; FL cat −0.53/reg −0.18; OneCycle best-val rides the
  anneal tail at any length) → **frozen 50ep cells stand.** (`EP100_ABLATION_AND_TX_RAM.md`.)

### 2a · Execution modes & default-flips (PR #56, `train_perf_multifold`, merged 2026-07-01)
Perf/tooling PR — **the champion (v16) default training path is byte-identical** (7-dimension adversarial review:
0 blocker/high; check2hgi engine + MTLnet + category heads untouched; the 3 STAN reg-head edits are identity-
equivalent mask refactors / opt-in fp32-attn). Full env/flag reference: [`../../SYSTEM_REFERENCE.md §3`](../../SYSTEM_REFERENCE.md);
guard contract: [`../pre_freeze_gates/DEFAULTS_AND_GUARDS.md`](../pre_freeze_gates/DEFAULTS_AND_GUARDS.md). For reproducing board cells:
- ⚠ **DEFAULT-FLIP — `auto-fp32` for large-C MTL.** A bare `FL/CA/TX` (reg C>2000) MTL run on Ampere+ now defaults
  to **fp32** (was fp16). This only replaces the old fp16 large-C path, which was the ep30 NaN-collapse **bug**
  (`CA_MTL_DIVERGENCE.md`) — never a valid frozen number. **Frozen board cells are unaffected** (every driver sets
  `MTL_DISABLE_AMP=1`/`MTL_AUTOCAST_BF16=1` explicitly → auto-fp32 is inert). Small states (AL/AZ, C<2000) keep fp16.
- ✅ **DEFAULT-ON — `MTL_SKIP_INERT_LOGT`.** The champion no longer loads per-fold log_T (byte-identical; alpha=0
  folds it out). `=0` restores the legacy load+leak-guard.
- **Opt-in modes (default-OFF, byte-identical when off):** `--canon v17` (= v16 + bs8192 + `--onecycle-per-head-lr`;
  the §1 candidate) · fold fan-out (`--only-folds/--run-id/--per-fold-seed` + `run_folds_fanout.sh` + `aggregate_folds.py`)
  · `--profile` · `MTL_STAN_FP32_ATTN` / `MTL_STAN_LEGACY_MASK`.
- ⚠ **Fan-out RNG caveat:** `--per-fold-seed`/`--run-id` reseed `seed+fold_id` → an order-independent baseline,
  **NOT bit-identical to a frozen sequential cell**; never use fan-out to regenerate/extend a §1 frozen cell.

## 3 · File map — where every result lives (the de-spread index)
**MTL + STL matched-score JSONs:** `docs/results/closing_data/`
- `h100/{alabama,arizona}_s0_mtl_fp32_matched_score.json` · `florida_s0_mtl_fp32_5f_matched_score.json` — MTL (main)
- `h100/{alabama,arizona,florida,california}_s0_stl_cat_ceiling.json` — STL cat ceilings (main)
- `h100/california_s0_mtl/california_s0_mtl_final_score.json` — CA MTL final 5f (cat 77.33/reg 65.66) ✅main
- `a40/tx_stl_cat_ceiling_s0.json` (69.95) · `a40/tx_stl_reg_ceiling_s0.json` (64.96) — TX ceilings ✅main
- `a40/tx_ba2_fp32_s0.json` — **TX MTL canonical** (true-fp32 5f, cat 77.51/reg 67.02, 0 skips, single-device) ✅main
- `h100/texas_s0_mtl/texas_s0_mtl_final_score.json` — TX MTL bf16 5f corroboration (cat 77.47/reg 67.00) ✅main
- ⚠ VOID: `h100/california_s0_{board,mtl}_partial.json`, `a40/tx_ba2_bf16_s0.json` (fp16/bf16 collapse — do not cite)
- STL **reg** ceilings: `docs/results/P1/region_head_*_dkovl*` (fp32, leak-free per-fold prior)
- **CSLSL cascade (§1b):** `a40/{al,az}_cascade_s0.json` (B4 cascade) + `a40/{al,az}_champG_a40_s0.json`
  (same-device champion-G comparand) — all A40 true-fp32, dk_ovl, seed 0 × 5f ✅main

**Narrative / per-cell docs:** `docs/studies/closing_data/`
- `BOARD_CELLS.md` (per-state MTL cells + AL/FL precision gates, consolidated) · `BOARD_H100_FINDINGS.md`
  (session consolidation) · `EP100_ABLATION_AND_TX_RAM.md` · `TX_A40_BF16_NAN.md` · `CA_MTL_DIVERGENCE.md`
  (the fp16 root cause) · `FAITHFUL_STAN_FINDINGS.md` (PR #53) · `MACS_BOARD_RESULTS.md` (baselines, **PR #36**)

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
  (stride-1, PR #38)**; **FL = 2/5 folds (PR #47, partial)** — cat ~28 ≪ comparand 73.47 (substrate drives cat),
  reg ~73 ≈ 72.71 (region near-tie); W3 is closed at AL/AZ/Istanbul, FL corroborates (finish 5f if a card frees).
  CA/TX CTLE-SC not run (dropped). POI2Vec/skip-gram/one-hot SC-cat are the
  representation **controls** (§7 checklist). **AL frozen-below-floor (17.8 < bigram ~19.5) DIAGNOSED REAL
  (M4, 2026-06-24)** — not a pipeline/leak bug (substrate genuinely swapped, cosine vs check2hgi 0.01; identical
  head → 55.6 on check2hgi vs 17.8 on CTLE) → **H100 FL CTLE-SC CLEARED** (`baseline_compare/alabama_ctle_DIAGNOSIS.md`).
- ✅ **CTLE-E2E (B1, CTLE's best/fine-tuned form) — FL NOW REAL & COMMITTED** (A40, seed 0 × 5f, leak-clean
  per-fold, fp32-native, wall 113 min): **AL cat 21.14 (final) / 23.94 (best-ep); FL cat 29.69 (final) / 33.45
  (best-ep)**, reg Acc@10 FL 61.44. JSONs `baseline_compare/{alabama,florida}_ctle_e2e_seed0.json`. The seeded
  re-run **reproduces the prior unbacked 29.65 to ±0.04** → it was a real result lacking an artifact, now fixed
  (phantom retired). Even at *best-epoch*, CTLE-E2E (FL 33.4) is **≪ Check2HGI cat (FL 73–75)** — present as the
  E2E rung beside the frozen CTLE-SC ladder; **never "we crushed CTLE."** (script `scripts/baselines/ctle_e2e.py`,
  determinism + best-epoch tracking added 2026-06-25 after a 2-advisor correctness/optimization review.)
- ✅ **Part-1 substrate contrast (Tbl 2) — Check2HGI vs HGI category-STL, NOW ON ONE WINDOWING (gated overlap).**
  PR #50 added the **HGI arm under `check2hgi_dk_ovl`** (new `HGI_DK_OVL` engine + streaming builder reusing the
  frozen overlap sequences → windows **byte-identical** to the Check2HGI arm; gates passed: row counts ==,
  `next_category` labels 100% identical, embeddings genuinely differ). Substrate margin (Check2HGI-board − HGI):
  **AL +29.31 (55.87 vs 26.56) · AZ +27.63 (57.13 vs 29.50) · FL +39.62 (75.15 vs 35.53) · CA +37.95 (70.26 vs
  32.31) · TX +37.47 (69.95 vs 32.48)** — large +27…+40 pp everywhere (HGI POI-level cat is a consistent
  ~0.46–0.52× of Check2HGI check-in-level cat), now windowing-consistent with Part-2. **All 5 Gowalla states
  COMPLETE** (Istanbul HGI optional per §3.4). JSONs `baseline_compare/{state}_hgi_ovl_cat.json`. The Check2HGI
  arm IS the §1 board STL cat ceiling. → **PAPER_PLAN Tbl 2 "non-overlap" caveat dropped** (one windowing for the
  whole paper); paper-side Tbl 2 `.tex` CA/TX cells ready for the orchestrator audit (AL/AZ/FL already filled).
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
  ⚠ **Wording: this is HMT-GRN-*style*** (own LSTM trunk + train-only region-transition prior from raw; **graph module
  + hierarchical beam search dropped**, no next-POI head), NOT a strict reproduction — call it "region-native E2E",
  never "faithful HMT-GRN" (deviation ledger: `../../baselines/next_region/hmt_grn.md`).
- ✅ **STAN faithful (region native-E2E, SECONDARY reference) — RE-IMPLEMENTED & CONVERGED (PR #53).** The old v4
  numbers (AL 34.46 / AZ 38.96, below the Markov floor) were a **collapse artifact — superseded, never cite.** The
  audited **v5** (6 faithfulness fixes: STAN-native **prefix-expansion** sequences, restored matching layer + interval
  embedding, constant-LR convergence; two-agent audit + GO code review; ~85× optimized via `F.embedding`+`torch.compile`,
  audit≈compiled within 0.1 pp) **clears the Markov floor AND stays below our MTL** at every measured state:
  **AL 60.72 / AZ 49.86 / FL 72.99 / Istanbul 61.86** (reg Acc@10, seed 0 × 5f; best-epochs 5–12, genuinely
  converged). **FL now COMPLETE** (5-fold v6 converged, Acc@10 **72.99 ± 0.34**, < our joint reg 77.28; PR #54
  committed the real 5-fold JSON, superseding the fold-0-only 0.7307 checkpoint); **CA/TX footnoted infeasible-at-scale**
  (HMT-GRN + Markov carry CA/TX). STAN sits in the comparability hierarchy as **SECONDARY** (HMT-GRN-style primary;
  ReHDM tertiary). JSONs `docs/results/baselines/faithful_stan_{al,az,istanbul}_5f_200ep_v5_*.json` +
  `faithful_stan_florida_5f_200ep_v6_opt.json`; finding `FAITHFUL_STAN_FINDINGS.md`.
- 🔭 **STAN-`stl_hgi` (STAN on OUR HGI region-embedding substrate, overlap footing) — NOT a paper baseline; FUTURE-
  HEADROOM signal (user steer, 2026-06-26).** At the board overlap footing: AL 70.35 / AZ 59.66 / FL 76.82 (reg Acc@10,
  PR #52) — at AL it **exceeds our MTL champion reg (69.81)**. That is precisely why it is NOT a region baseline we
  report (it would read as "STAN beats us"): it isolates substrate-vs-architecture and shows **our substrate lifts even
  an off-the-shelf region model above our current MTL → the MTL has headroom to grow.** Filed as future work; kept OUT
  of Table 3 and the baseline set. JSONs `baseline_compare/{alabama,arizona,florida}_stan_hgi_ovl_s0.json`.
- ✅ **CSLSL / cascade (B4, role-3 "published MTL alternative") — CANONICAL = the §1b A40 dk_ovl DEAD TIE.** On the
  board base (dk_ovl, true-fp32, same-device champion-G re-run), cascade-vs-parallel is a **dead tie** (Δjoint AL
  +0.02 / AZ +0.00 ≪ fold-std) → our parallel cross-attn **matches** the dominant published MTL alternative at equal
  cost. **Contribution-2 is anchored to the §1 STL-ceiling lift (+4.7…+8.1 cat, +reg at large states), NOT this
  cascade cell** (a tie only rules out that a cheaper cascade would have matched our lift — which it does). See §1b.
  - ⚠ **The M4/MPS `set-a` CSLSL is an APPENDIX CROSS-CHECK ONLY — never a paper-table number.** On the non-board
    v14 set-a base (stride-9, MIN_SEQ=5, data-starved: parallel cat ~51 not the board ~63) it shows parallel ≥
    cascade (Δcat +5.04 AL / +1.62 AZ) — *same direction*, but it **CONTRADICTS the canonical dk_ovl tie if cited as
    headline** (the gap is regime-fragile: coupling helps when data is thin, washes out on the board). Cite ONLY as
    "same-direction corroboration (parallel ≥ cascade, never worse)". MPS==CPU within 0.63pp = the device-trust gate.
    FL set-a OOM'd on MPS (4/5, no comparand — VOID). Docs: `../../baselines/cslsl_cascade.md` (set-a) + `CSLSL_CASCADE.md` (dk_ovl, canonical).
- ✅ **Feature-concat control (role-2, FL) — DONE on disk, no new run.** The A2 `--add-visit-features` (`hgifeat`)
  arm: **HGI⊕raw-features ≈ HGI-alone** (features add only ~+0.8 pp), both **far below Check2HGI** → the category
  gain is the **hierarchy, not feature injection**. (`docs/results/P1/region_head_florida_checkin_5f_30ep_A2_hgifeat_category_s0.json`;
  exact numbers/metric reconcile at write-time — A2 is a 30ep ablation, distinct from the 50ep ceiling.)

> ⚠ **AMP-gate PRE-TABULATION GATE (for the in-flight H100/A40 FL baseline cells).** The fp16 fix (#43) is **opt-in**
> (`DISABLE_AMP=1`/`MTL_DISABLE_AMP=1`). Before trusting/tabulating ANY FL reg/joint baseline number, verify the
> producing run had it set AND the FL JSON shows healthy late best-epochs (no ~ep12 freeze / no ~3% collapsed fold)
> — else the FL wide-reg cell silently NaN'd under buggy fp16. The #43 driver wrappers export it; confirm per cell.

## 5 · Provenance legend
✅ main = source JSON on main, verified-readable ·
⚠ VOID = fp16/bf16-collapse artifact, never cite. All numbers audited against `wsftdemmg` (collate-trust verdict +
adversarial source-verification). Honest headline: **beats on category everywhere; beats on region at the large
states (FL/CA/TX, all 5f), matches at the small (AL/AZ/Istanbul within δ=2 pp).**
