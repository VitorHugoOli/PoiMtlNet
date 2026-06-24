# HANDOFF — REDUCED board · **Macs** (M2 Pro 32 GB + M4 Pro 24 GB, MPS) · branch `study/board-m2pro`

> Deadline-grade 1-seed board (`RUN_MATRIX_REDUCE.md`). **FOCUS (2026-06-24, user): execute exactly TWO baselines
> across all 6 states — `CTLE-SC` and `HMT-GRN`.** These are the two highest-value, deadline-gating baselines; the
> other SC baselines (b2b/b2c/poi2vec) and the other E2E natives are DONE-or-deprioritized (full audit in §6/§7).
> **1 seed (0) × 5 folds**, device-internal on MPS (fp32, no confound). Continue on `study/board-m2pro`.
>
> **▶ RESULTS + ENGINEERING KNOWLEDGE (2026-06-24, M4 Pro run): [`docs/results/closing_data/MACS_BOARD_RESULTS.md`](../../results/closing_data/MACS_BOARD_RESULTS.md)** —
> AL/AZ CTLE-SC (+37pp cat / tied reg) + comparand DONE; HMT-GRN AL/AZ/CA/Istanbul DONE (FL rerun + TX queued); Istanbul ETL + Phase-V substrate rebuilt on Mac; **CTLE-SC FL/CA/TX/Istanbul → CUDA** (24 GB MPS can't hold the 1.27M-3.5M-row builds). 7 bug fixes (p1 collate, min_seq parity, fsq_tree, 4× b3 large-state OOM) + a self-contained CUDA handoff are documented there.

## 0 · SCOPE — CTLE-SC + HMT-GRN, states = AL · AZ · FL · CA · TX · Istanbul (Massive-STEPS)
| Baseline | What / why | Comparand | Command base |
|---|---|---|---|
| **CTLE-SC** (substrate-column, **leak-clean**) | CTLE's per-fold contextual embedding → OUR matched heads. **The reviewer regular-vs-poster GATE (W3): novelty C1 needs CTLE scored leak-clean.** Isolates the REPRESENTATION. | **Check2HGI-SC** (our embedding → same heads) | `mac_baseline_compare.py --baseline ctle` |
| **HMT-GRN** (native-E2E, region-native) | The **sole external MTL / region-native baseline** — the "we beat SOTA on region" row. Isolates the SYSTEM. | **our FULL MTL champion** (~63.6 cat / ~69.8 reg), NOT the STL ceiling | `b3_hmt_grn.py` (defaults to `check2hgi_dk_ovl`) |

Both at **seed 0 × 5 folds**. **Leak-clean = per-fold train-only pretrain** (the original CTLE `--folds 1` built
n_splits=2 → 81.8% of val users leaked into the pretrain corpus; `mac_baseline_compare.py` does per-fold staging +
`--only-fold` to avoid it). Do NOT spend Mac time on b2b/b2c/poi2vec/STAN/ReHDM/MHA+PE/POI-RGNN this pass.

## 1 · STATUS — what's done, what remains (fan-out)
- **HMT-GRN:** ✅ AL done (cat 20.43 / **reg 62.37**, the closest region competitor). **Remaining: AZ, FL, CA, TX, Istanbul.**
- **CTLE-SC (leak-clean):** ⚠ AL **fold-0 only**; folds 1–4 + every other state remain. CTLE embeddings exist on the
  SSD (`board_baselines/ctle`, all states) **but must be PER-FOLD leak-clean** (pretrained on that fold's TRAIN users
  only) — verify the SSD cells are per-fold (not the leaky `--folds 1`); rebuild any that aren't via
  `build_ctle_substrate.py --state <s> --fold <f> --stride 1` for f∈0..4. **Remaining: AL folds 1–4, then AZ/FL/CA/TX/Istanbul.**

## 2 · EXACT COMMANDS (per state; seed 0 × 5f)
```bash
SSD="/Volumes/Vitor's SSD/ingred/output"
# ---- CTLE-SC (cat next_gru + reg next_stan_flow, per-fold leak-safe, C29 stride-1 log_T) ----
# (rebuild per-fold leak-clean CTLE emb first IF the SSD cell is not already per-fold:)
#   for f in 0 1 2 3 4; do PYTHONPATH=src .venv/bin/python scripts/baselines/build_ctle_substrate.py --state <state> --fold $f --stride 1; done
PYTHONPATH=src .venv/bin/python scripts/closing_data/mac_baseline_compare.py \
    --state <state> --baseline ctle --cells-root "$SSD" --folds 5 --heads cat reg
#   -> docs/results/closing_data/baseline_compare/<state>_ctle.json (per-fold + aggregate, vs Check2HGI-SC)

# ---- HMT-GRN (native-E2E; defaults to check2hgi_dk_ovl stride-1; cheap LSTM) ----
PYTHONPATH=src .venv/bin/python scripts/baselines/b3_hmt_grn.py \
    --state <state> --seed 0 --folds 5 --epochs 50          # --device auto-MPS
```
⚠ **Reg-modality match (C-2):** the CTLE-SC reg comparand must be the **checkin-modality** Check2HGI reg ceiling
(`--input-type checkin`), NOT the region-modality 69.73 — `mac_baseline_compare.py` runs reg checkin-modality on
BOTH sides (the C-1/C-3 fixes landed). The earlier bit-identical-69.76 SC reg JSONs are VOID (`RUN_MATRIX_REDUCE §4b`).

## 3 · DEVICE allocation by state (≤2–3 concurrent under `ram_watchdog.sh`)
- **MPS-safe (M2 Pro 32 GB + M4 Pro 24 GB, split the states):** **AZ, FL, Istanbul** (+ AL CTLE folds 1–4). Small/mid.
- **CA, TX — LARGE → likely need a CUDA card.** CTLE per-fold MLM pretrain + the 8501/6553-wide reg head may exceed
  unified memory (the MPS kNN on CA's 169k POIs already triggered an OOM-reboot once). **Flag for the user / offload to
  the A40 or H100** when a card frees up, OR run only HMT-GRN (cheap LSTM) on the Mac and defer CTLE-SC CA/TX to CUDA.

## 4 · Istanbul (Massive-STEPS) — match the champion's base
The Istanbul champion-G (PR #33 Phase V: cat **+8.06** / reg **−0.58** / +17.27 over Markov, 4 seeds) ran on the
**frozen GCN 500-ep mahalle substrate** (`--engine check2hgi` for Istanbul, `phase_v_substrate.py`), NOT a `dk_ovl`
overlap base. **For a valid Δ the Istanbul baselines MUST use the SAME base + windowing as that champion.** Before
running CTLE-SC / HMT-GRN on Istanbul: confirm the Istanbul base (check `output/check2hgi/istanbul`; build the
overlap base only if the champion used overlap — it did NOT, so point `--engine check2hgi` / `--base` at the Phase-V
substrate). Comparand = the Istanbul champion-G, reported as gap-to-ceiling / lift (region counts differ across corpora).

## 5 · PROCESS / STOP
- Branch `study/board-m2pro`; incremental commits per (baseline, state) cell + result JSON.
- **STOP for the user:** MPS OOM on CA/TX (offload to CUDA); any leak-marker / fold-split assertion failure; an
  Istanbul base/windowing mismatch vs the #33 champion. Do NOT run the MTL champion (CUDA) or merge.
- **CAVEATS (`RUN_MATRIX_REDUCE §4b`):** never claim "baselines use our substrate as a base" (overclaim); the SC reg
  comparand is checkin-modality; HMT-GRN comparand is the full MTL champion (not the STL ceiling).

---

## 6 · PROGRESS LOG — DONE so far (2026-06-23, M2 Pro / MPS)

### Data consolidation (§4) — COMPLETE
- **A40 → SSD pull DONE** (5→serial→resilient channels; the flaky external enclosure drops under ≥5 concurrent
  WRITE streams, so the working mode is ≤2-channel + drop-resilient retry + end-of-run hash-verify-and-repull).
  All large-state board cells now on the SSD: **b2b CA/TX (40)**, **ctle FL/CA/TX (60)**, **poi2vec CA (10, seeds{0,1})
  + TX (5, seed0)**. Combined with the Mac-built small states → full board baseline set local.
- **Hash manifest**: [`BASELINES_HASH_MANIFEST.json`](BASELINES_HASH_MANIFEST.json) (per-cell bytes+sha256; gen/merge
  via `scripts/closing_data/baseline_hash_manifest.py`; mirrors V14/HGI manifests). 326 cells (b2b 120, ctle 120,
  poi2vec 80, b2c 6); **TODO**: re-merge to add poi2vec CA/TX (they post-date the first A40 hash).
- **Drive**: Mac-built baselines + manifests mirrored (throttled). ⚠ Mac→Drive is **mirror-mode + tiny local cache**
  → bulk Drive sync is impractical from this Mac; SSD is the durable copy.

### Part-1 + floors (REDUCE §1, non-training) — COMPLETE
- **Embedding-quality** ([`PART1_QUALITY/`](PART1_QUALITY/), 6 states, MPS fp32 ≡ CPU verified): Check2HGI (v14/v11) ≫ HGI
  on **next-cat** (kNN-10 0.98 vs 0.78; linear-probe 0.98 vs 0.69; silhouette 0.56 vs 0.00); **HGI keeps a next-reg
  edge** (POI-granularity); **CKA vs HGI ≈ 0.16** (fundamentally different reps). FL/CA/TX POI-capped at 40k.
- **Simple-baseline floors** (`docs/results/P0/simple_baselines/`): Markov-1 **region** Acc@10 — AL .470, AZ .430,
  FL .650, CA .521, TX .549. (Category Acc@10 degenerate at C=7 → use majority Acc@1 / macro-F1.)

### Matched-head STL comparison (§3) — AL COMPLETE; tooling validated
- **Check2HGI dk_ovl AL ceiling (Mac/MPS, seed0×5f)**: **cat macro-F1 55.72 ± 1.54**, **reg Acc@10 69.73 ± 3.5**
  (reg matches prior 69.98 → pipeline validated).
- **AL baseline Δ (cat VALID; reg INVALID — see bug)** — `docs/results/closing_data/baseline_compare/` (gitignored):

  | substrate | cat macro-F1 ✅ | reg Acc@10 ⛔ |
  |---|---|---|
  | **Check2HGI (ours)** | **55.72** | (69.73, region-modality ceiling) |
  | b2b skip-gram | 24.20 | 69.76 ⛔ |
  | poi2vec | 22.53 | 69.76 ⛔ |
  | b2c one-hot (floor) | 19.50 | 69.76 ⛔ |

  **cat headline (valid)**: substrate drives **next-cat** — Check2HGI **+31…+36 pp** over every baseline.
  **⛔ reg BUG (2026-06-23)**: the reg numbers are **bit-identical across baselines** (per-fold `[73.55,68.95,70.98,71.31,64.02]`)
  → **invalid**. Cause: `p1_region_head_ablation.py --input-type region` hardcodes `_load_region_embeddings(source="check2hgi")`
  (line 149/158) — `--engine` only changes sequences/labels, NOT the region-embedding values, so every baseline fed the SAME
  check2hgi region embedding (+ same shared log_T) into the reg head. The baseline substrate is never used.
  **TWO independent reg defects (2nd advisor audit, 2026-06-23) — reg Δ INVALID until BOTH fixed + re-run:**
  - **C-1 substrate bypass (input)**: `--input-type region` hardcodes the check2hgi region embedding → fed identical input
    to every baseline. **Fixed**: `run_reg` now `--input-type checkin` (baseline's own per-visit embedding → substrate-sensitive).
  - **C-2 modality mismatch**: the recorded Check2HGI reg ceiling **69.73 is region-modality** (~53% scale); baselines now run
    checkin-modality (~20% scale) → apples-to-oranges. **Must recompute the Check2HGI dk_ovl reg ceiling at `--input-type checkin`**
    for a matched Δ. (Region-modality stays a Check2HGI-only ceiling — baselines have no region embedding.)
  - **C-3 log_T leak (C29 code adopted but NOT used)**: the driver's old `stage_logt` **copied the canonical stride-9 log_T and
    touched it** (mtime guard fooled) → its user→fold partition mismatched the stride-1 training split → leak. **Fixed**:
    `stage_logt` now BUILDS the per-engine log_T over the engine's own stride-1 partition via `compute_region_transition
    --engine <eng> --per-fold` (C29 path; payload carries `engine` provenance).
  - **ALL AL reg JSONs in `baseline_compare/` are pre-fix → discard; reg must be re-run for every state after these fixes.**

### Tooling built/validated this session
- `scripts/closing_data/mac_baseline_compare.py` — the §3 driver (per-fold stage → stride-1 input-build → cat via
  `train.py --task next --cat-head next_gru --only-fold f` → reg via `p1_region_head_ablation.py next_stan_flow
  --input-type region --only-fold f` → score → aggregate).
- `scripts/p1_region_head_ablation.py` — added **`--only-fold`** (fold-index-preserving for the seeded log_T;
  **validated bit-identical** to fold-0 of a full run) + baseline engines in `--engine-override` choices.
- `scripts/embedding_eval/{geometry,linear_probe}.py` — `_device()` MPS-aware + `EMBED_EVAL_DEVICE` override.
- **C29** (`scripts/compute_region_transition.py`, from PR #33): engine-aware log_T over the **stride-1** partition.
  ⚠ The CODE was adopted but the driver initially **copied the canonical stride-9 prior instead of building per-engine**
  (2nd-audit C-3) → leak persisted in the AL reg JSONs. NOW `stage_logt` invokes `--engine <eng> --per-fold` (built, not copied).
- Ops: `pull_a40_{serial,parallel_v2,poi2vec}.sh`, `ram_watchdog.sh`, `baseline_hash_manifest.py`.

### Operational findings (for the next states)
- **MPS parallelism**: 4 concurrent AL trainings + pulls → RAM 25% (OOM-reboot risk). **2–3 concurrent is the safe max**;
  `ram_watchdog.sh` kills at <14% free. The earlier full reboot was the **MPS kNN eval on CA** (169k POIs) over-subscribing
  ~40 GB unified memory — NOT the pull/SSD.
- **SSD enclosure** (WD_BLACK SN850X, SMART OK) drops the bus under ≥5 concurrent writes — use ≤2-channel resilient pulls.
- **FL feasible on the Mac** (~4 GB stride-1 inputs, fits): pull `next`/`next_region` from the A40 (`check2hgi_dk_ovl/florida`,
  deterministic) — but its log_T is **stale/pre-C29** (mtime 06-18 < next_region 06-20) → **rebuild with C29** locally.

### PENDING
- **FL b2b + Check2HGI-FL** (device-internal Δ on the Mac): pull A40 FL `next`/`next_region` → C29 FL log_T → run.
- **AZ, GA** AL-style baseline Δ (small states, safe on MPS).
- **CA / TX** baselines — large; may need a CUDA card (§5 STOP) — flag before fan-out.
- Re-merge `BASELINES_HASH_MANIFEST.json` to include poi2vec CA/TX.

---

## 7 · TASK-NATIVE BASELINES — SOTA audit + native-E2E plan (2026-06-24)

### Why this section exists
The SC baselines (b2c/b2b/poi2vec/ctle) are all **next-POI / embedding** methods adapted to our tasks via a frozen-substrate → STL head. That undersells deep models (e.g. **CTLE-frozen cat = 17.75, BELOW the random floor**). For an honest "best-baseline vs Check2HGI" claim we add **task-native published baselines** run with their **authors-intended architecture/input** and only the **output head adapted** to next-cat / next-region (the established pattern in `flashback_e2e.py` / `b3_hmt_grn.py`).

### SOTA audit — are our baselines true papers for THESE tasks? (evidence: `docs/baselines/`, `docs/research/references.md`, ultracode wf `wf_31cfde91-f0d`)
- **next-CATEGORY — we DO have category-native papers:**
  - **MHA+PE** (Zeng 2019) — category-native; reproduced 5-fold/6-state (`research/baselines/mha_pe/`).
  - **POI-RGNN** (Capanema 2022, Ad Hoc Networks / PE-WASUN'21, *"Predict the Next Place's Category"*) — category-native; reproduced (`research/baselines/poi_rgnn/`). Caveat: paper numbers are global-Gowalla; only our state-level reproductions are comparable (`POI_RGNN_AUDIT.md`).
- **next-REGION — NO truly region-native published baseline exists** (the field does next-POI and adapts the output). Region coverage:
  - **HMT-GRN** (SIGIR'22) — **region-native** (hierarchical region+POI target). `scripts/baselines/b3_hmt_grn.py`.
  - **ReHDM** (IJCAI'25) — **region-aware SOTA** (quadkey-L10 region as *input*; target adapted POI→region). `research/baselines/rehdm/` (`train.py` faithful). [Chosen 2nd region baseline — zero new impl.]
  - (DRRGNN TKDD'22 *"next activity region"* is the only strictly region-target-native candidate found — **deferred**, no public impl.)
  - STAN (WWW'21) is next-POI-native, adapted — supportive, not a region-native claim.

### Decided AL native-baseline roster (seed 0 × 5 folds)
| task | native baselines |
|---|---|
| next-category | **MHA+PE**, **POI-RGNN** (category-native) |
| next-region | **HMT-GRN** (region-native) + **ReHDM** (region-aware SOTA) |
| E2E/substrate probes | CTLE-E2E, POI2Vec-E2E, Flashback (native arch + adapted reg/cat heads) |

### Methodology (the honest comparison)
- Native baselines run their **own architecture/input**, output head adapted to next-cat (macro-F1) / next-region (`top10_acc_indist`), on **our board folds** (StratifiedGroupKFold, groups=userid, y=next_category, `--seed 0 --folds 5`).
- **Comparand = our FULL solution (MTL champion / E2E Check2HGI)**, NOT the frozen-substrate STL ceiling (E2E-vs-frozen would be apples-to-oranges).
- **Row-base alignment (REQUIRED):** the E2E natives default to canonical **stride-9** (`load_next_data(CHECK2HGI)`, 12,709 rows AL) but the board base is gated **stride-1** `check2hgi_dk_ovl` (96,326 rows). All native baselines must read the stride-1 base for comparability (the ultracode workflow adds a `--engine/--base` flag to Flashback/HMT-GRN; MHA+PE/POI-RGNN/ReHDM need the same check before tabling).

### ultracode workflow `wf_31cfde91-f0d` — DONE
Implemented + adversarially reviewed: `scripts/baselines/ctle_e2e.py` + `poi2vec_e2e.py` (native arch + reg/cat heads,
stride-1, leak-safe per-fold) and aligned `flashback_e2e.py`/`b3_hmt_grn.py` to the stride-1 base (`--engine/--base`
flag, default `check2hgi_dk_ovl`). All 4 verified to read the **96,326-row stride-1** base. (One review agent —
`review:align_existing` — died on an API drop; the alignment was verified manually: flashback smoke rows=96,326.)
Region research confirmed: **HMT-GRN is the only region-native POI baseline; no other is missing**; grid-native
crowd-flow models (ST-ResNet/STDN/UniMove) are dense-GPS/aggregate → apples-to-oranges for sparse check-ins.

### Native-E2E baseline RESULTS — AL, seed 0 × 5 folds, stride-1, vs Check2HGI (paused mid-run)
| model | cat macro-F1 | reg Acc@10 | status |
|---|---|---|---|
| **Check2HGI — MTL champion** (full solution) | **~63.6** | **~69.8** | reference (PR #33) |
| Check2HGI — STL ceiling | 55.72 | 61.94 (checkin) / 69.73 (region) | this lane |
| **HMT-GRN** (region-native E2E) | 20.43 | **62.37** | ✅ 5f |
| **CTLE-E2E** (native transformer, fine-tuned) | 21.24 | 42.74 | ✅ 5f |
| **Flashback** (IJCAI'20) | 18.06 | 26.65 | ✅ 5f |
| **POI2Vec-E2E** | 18.65 | 49.59 | ⏸ 1/5 folds (paused) |

**Findings (honest):** (1) **Category — Check2HGI crushes every native baseline by +35–45 pp**; even fully-fine-tuned
native models can't approach it. Decisive. (2) **Region — HMT-GRN (region-native) is the closest competitor (62.4 ≈ our
STL-checkin 61.9, below our MTL 69.8)**; region is where a region-native model competes, but Check2HGI's full solution
still leads. (3) **CTLE native-E2E (cat 21.2) >> CTLE frozen-SC (cat 17.75)** — confirms freezing undersells deep models;
native E2E is the honest treatment.

### PAUSED — resume here
- **Resume POI2Vec-E2E** AL (got 1/5; `poi2vec_e2e.py --state alabama --seed 0 --folds 5 --epochs 20`, CPU/geo-tree, slow).
- **Align + run MHA+PE, POI-RGNN (cat-native) + ReHDM (region-aware)** — they use `research/baselines/*/etl.py`
  (canonical stride-9); apply the same `--engine/--base` stride-1 alignment the workflow gave Flashback/HMT-GRN,
  then run AL seed 0 × 5f. (Inline mechanical change, or a small ultracode pass.)
- Then fan out the native-E2E set to **AZ/GA/FL** (small/mid) and flag **CA/TX** (may need CUDA).
- ⚠ MPS: keep ≤2–3 concurrent under `ram_watchdog.sh` (4+ → OOM-reboot zone).
