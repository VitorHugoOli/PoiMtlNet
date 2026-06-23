# RUN_MATRIX_REDUCE — deadline-grade 1-seed board (MobiWac, ~2-day budget)

> **Why this exists.** The full board is 6 states × {0,1,7,100} × 5 folds (n=20/cell). For the MobiWac
> deadline (~2026-06-25) we run a **reduced board: seed 0 only × 5 folds (n=5/cell)** — the *minimal valid
> paired-test configuration* (per-fold paired Wilcoxon over 5 folds → p-floor **0.0312**; this is the
> "single-seed ceiling" protocol already sanctioned in the paper's §8). It gives every headline number a
> defensible significance test now; the {1,7,100} top-up to n=20 is a post-deadline refinement, not a
> blocker. Supersedes the full `RUN_MATRIX.md` *for execution scheduling only* — the recipe, precision pin,
> scorer, and pins are inherited verbatim from `RUN_MATRIX.md §0/§0a` and `HANDOFF_BOARD.md §4`.

## 0 · What carries over UNCHANGED from RUN_MATRIX.md
- **Recipe** = champion **G / v16** on the **gated stride-1 overlap** engine `check2hgi_dk_ovl` (RUN_MATRIX §0).
- **Precision** = **bf16 autocast** (`MTL_AUTOCAST_BF16=1`) train + **fp32 eval**, PENDING the §0a gate verdict
  (else full fp32 `MTL_DISABLE_AMP=1`). The fp16 default is BANNED (CA/TX ep30 collapse; CA_MTL_DIVERGENCE.md).
- **Scorer** = `r0_matched_rescore.py` (cat macro-F1; reg FULL `top10_acc`, fp32, BOTH sides).
- **Pins** = seeded per-fold log_T + freshness preflight; `--compile --tf32`; `MTL_STRICT=1`; auto-fit dataset
  (NEVER `MTL_DATASET_GPU=1` at CA/TX); torch 2.11; PID-suffix rundirs.

## 1 · The reduced cell set (seed 0 × 5 folds = n=5)
Only the **MTL cells** are re-runs (fp16-invalid). STL ceilings are precision-settled → **REUSE** where a valid
artifact exists; build only the missing ones.

| Family | Cells | Precision | Status |
|---|---|---|---|
| **MTL champion-G** (the headline, Part 2) | 5 US states (AL,AZ,FL,CA,TX) | bf16/fp32 (gate) | **RE-RUN ALL** (every prior MTL was fp16) |
| **STL cat ceiling** `next_gru` | 5 states | fp16-eval OK (cat ≈ precision-insensitive) | REUSE if present; else build (cheap) |
| **STL reg ceiling** `next_stan_flow` | 5 states | **already fp32** | **REUSE**: TX 64.96, CA 63.48, FL 76.71, AL 69.98; build AZ if absent |
| **Part-1 representation** (Check2HGI vs HGI, STL cat + reg) | 5 states | fp32 (STL) | build on overlap; HGI STL needed |
| **Embedding-quality** (kNN-LOO / silhouette / CKA) | Check2HGI vs HGI | no training | compute from embeddings (cheap, any machine) |
| **Baselines (substrate-column)** one-hot / skip-gram / POI2Vec / CTLE / STAN / Markov-1 | 5 states | fp32 (Macs/MPS) | Macs, device-internal (§3) |
| **Istanbul (external validity)** | provisional | MPS dry-run (≈fp32) | **REUSE the dry-run** as the §6.3 box; proper v14 build = stretch |

## 2 · Device allocation (whole-state-per-device → every per-state Δ is device-internal-clean)
**The rule that makes the split valid:** assign each state's FULL cell set (MTL + its STL ceilings) to ONE
device. Then every reported Δ (MTL−STL, the Fig-3 points) is a same-device difference — valid. Fig 3 plots
**Δ's**, so mixing AL-on-H100 with FL-on-A40 is fine. ONLY an absolute cross-state table needs a device-class
footnote (AL/AZ/FL/CA = H100-Hopper; TX = A40-Ampere; baselines = MPS-fp32). The A100-equivalence A/B is
**NOT required** for the headline under this split (it only buys cross-state *absolute* comparability) → skipped
for the deadline.

| Device | States / work | Order (cheap → expensive) | Handoff |
|---|---|---|---|
| **H100** (fast, interrupt-prone) | precision gate + AL, AZ, **FL**, CA | **AL bf16-vs-fp32 gate** → AL → **AZ‖FL co-scheduled** → **CA last** | `HANDOFF_BOARD_H100.md` |
| **A40** (stable Ampere) | finish POI2Vec, then **TX only** | POI2Vec → **TX** (~11h) | `HANDOFF_BOARD_A40.md` |
| **Macs** (M2 Pro 32G + M4 Pro 24G, MPS=fp32) | ALL baselines, device-internal | one-hot → skip-gram → POI2Vec → CTLE | `HANDOFF_BOARD_MACS.md` |

**Precision gate** (RUN_MATRIX §0a) runs FIRST on the H100 at **AL** (smallest → ~2-3h for both arms; doubles as
the AL MTL result). Verdict → board commits bf16 (likely) or fp32. **STL ceilings / baselines / Part-1 do NOT
wait for it** (they're fp32 already) — only the MTL re-runs do.

## 3 · Baselines on the Macs (device-internal, valid without CUDA)
Substrate-column baseline = **baseline-emb → our STL head**. Run the FULL comparison ON ONE MAC: train our STL
head on the baseline embedding AND on the Check2HGI embedding, same Mac, same fold split → the baseline-vs-ours
Δ is device-internal-clean. MPS runs fp32 (no fp16 autocast) → no precision confound. This offloads ALL baseline
work from the CUDA cards. Seed 0 × 5 folds. (Partial reuse: SSD already has `b2b_skipgram` s0 × 5f, `b2c_onehot`,
`poi2vec` cells — see §4.)

## 4 · Data already in hand (reuse, don't rebuild)
- **SSD** (`/Volumes/Vitor's SSD/ingred/output`): `b2b_skipgram_s{0,1,7,100}_f{0-4}`, `baseline_b2c_onehot64`,
  `board_baselines/{b2b,ctle,poi2vec}`, `baselines/{stan,rehdm,poi_rgnn,mha_pe}`, **v14**, **v11**. (CTLE = AL only.)
- **Reuse**: the fp32 STL reg ceilings (above); the Istanbul/NYC MPS dry-run (`second_dataset/DRY_RUN_RESULTS.md`).
- **Regen, don't move**: `check2hgi_dk_ovl` is **deterministically regennable** from v14+v11 (both on Drive) via
  `build_overlap_probe_engine.py <state> 1` — each CUDA box rebuilds it locally; do NOT push it to Drive.

## 5 · Honesty caveats forced by n=5
- TOST **non-inferiority power is lower at n=5** — report the TOST power statement (§5.3d) or label region
  non-inferiority "n=5 provisional". Wilcoxon **superiority** (cat) is fine (p-floor 0.0312 < 0.05).
- All Part-2 cells stay **PROVISIONAL** until (optionally) topped up to n=20 post-deadline.
- Region wording follows the corrected-precision result (likely **beats/non-inferior**, per the §3 EMERGING
  UPDATE in PAPER_PLAN.md), NOT Decision-C's fp16 "trails 2.4-3pp" (those were collapse artifacts).
