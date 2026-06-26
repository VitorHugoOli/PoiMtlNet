# HANDOFF — A40 · remaining GPU work for the MobiWac close · self-contained · 2026-06-25

> **You are the A40. Read this file first.** It is the master queue for the GPU-blocked items in
> [`../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md`](../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md).
> The two **CPU-only** items (Tbl-1 stats + the small-state region **TOST**) are **already DONE** on the
> Mac (PR #49, branch `closing-data/tbl1-overlap-and-region-tost`) — do not redo them. Your plate is what remains:
> ✅ **W6 (PR #48) + Blocker 2 FL CTLE-E2E (PR #50) DONE.** Your plate is now **Blocker 3 — finish CA + TX**
> (HGI cat-STL under overlap; AL/AZ/FL landed in #50, margins +27.6…+39.6; §3). The `HGI_DK_OVL` engine + builder
> are on main — just run CA then TX (`run_hgi_ovl_cat_cell.sh`, build→train→delete, disk is tight). **Blocker 1**
> (FL CTLE-SC 5f) is the **H100's** job (PR #47 landed 2/5; W3 closed at AL/AZ/Istanbul) — §5 here is a fallback only.
>
> **House rules (non-negotiable, same board as everyone):** seed 0 × 5 folds (n=5); gated stride-1 overlap,
> engine `check2hgi_dk_ovl`, MIN_SEQ=10; **fp32** (Ampere bf16 grad-NaN) — set the AMP gate and verify healthy late
> best-epochs; leak-clean per-fold train-only (**NEVER `--folds 1`** for a substrate build — it leaked 81.8% of val
> users); user-disjoint `StratifiedGroupKFold`. Scope = AL/AZ/FL/CA/TX + Istanbul; **GE (Georgia) is out of paper
> scope**. Commit per cell + a one-line finding; **never cite a void fp16/bf16 JSON**. Work on a branch + PR; the
> orchestrator audits and records the verdict — do not merge to main.

---

## 0 · Setup (once)
```bash
cd <A40 repo>; git checkout main && git pull        # MUST include ae8...(--freeze-reg-stream) + PR #49 merged
export PYTHONPATH=src
export DISABLE_AMP=1 MTL_DISABLE_AMP=1               # PR #43 fp16 gate -> true fp32, no autocast (board protocol)
export MTL_STRICT=1 MTL_COMPILE_DYNAMIC=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCHINDUCTOR_CACHE_DIR=$HOME/.inductor_cache_board
```

## Priority order (UPDATED 2026-06-26 — W6 + Blockers 2 & 3 DONE; STAN re-impl landed PR #53)
1. ✅ **W6 encoder-isolation probe — DONE (PR #48): trunk, not transfer.** §1 = record; no re-run.
2. ✅ **Blocker 2 — FL CTLE-E2E — DONE (PR #50)** (FL 29.69/33.45; phantom 29.65 retired). §2 = record; no re-run.
3. ✅ **Blocker 3 — HGI cat-STL under overlap — DONE (CA/TX landed PR #52: CA +37.95 / TX +37.47).** All 5 Gowalla
   states complete (AL/AZ/FL #50 + CA/TX #52); Tbl 2 is one windowing. §3 = record; no re-run.
4. **Blocker 4 — region externals — THE LIVE A40 WORK.** ⚠ **Re-scoped (2026-06-26):** the STAN headline is now
   **FAITHFUL-from-raw** (audited v5, PR #53: AL 60.72 / AZ 49.86 / Istanbul 61.86, all converged + below our joint).
   **STAN-`stl_hgi` is NO LONGER a baseline** — it landed in #52 (AL 70.35 / AZ 59.66 / FL 76.82) and is now a
   **future-headroom signal** (substrate lifts STAN above our MTL@AL), kept OUT of the paper. **Remaining A40 work:**
   **(a) FINISH FL faithful-STAN** (in-flight, fold-0 v6 ckpt Acc@10 0.7307 → complete 5f; CA/TX infeasible-at-scale,
   footnoted); **(b) ReHDM-faithful** (AL/AZ done per refooting; FL/CA/TX as possible, Istanbul via the FSQ→mahalle
   adapter or footnote). See §4 + `../../../articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md`.
5. **Blocker 1 fallback** — FL CTLE-SC 5f, only if the H100 never reaches 5f (it has 2/5; W3 closed at AL/AZ/Istanbul). See §5.

---

## 1 · W6 probe — ✅ DONE (PR #48): TRUNK, not transfer
Result: probe cat (region frozen) AL 63.50 / AZ 63.67 / FL 79.79 ≈ full-MTL cat (±0.3 pp), ≫ STL ceiling
(+4.6…+7.6) → the joint category win is the shared trunk, not region→category transfer. Recorded in
`W6_ENCODER_ISOLATION.md` + RESULTS_BOARD §1c. **No re-run.** (Run-spec below kept for the record.)

Reproduce (record only): `STATES="alabama arizona florida" bash scripts/run_freeze_reg_probe.sh` (seed 0 × 5f,
fp32; the first-fold `[per-head-LR]` line must show the reg group at 0 trainable params). Full verdict + per-state
JSONs in [`W6_ENCODER_ISOLATION.md`](W6_ENCODER_ISOLATION.md) + RESULTS_BOARD §1c.

---

## 2 · Blocker 2 — FL CTLE-E2E (the unrun half of the FL CTLE gap)
> ✅ **DONE (PR #50 merged).** FL CTLE-E2E (A40, seed 0 × 5f, leak-clean, fp32): **cat 29.69 (final) / 33.45
> (best-ep), reg 61.44**; AL 21.14 / 23.94. The seeded re-run **reproduces the prior unbacked 29.65 to ±0.04**
> (it was a real result missing its artifact, not a fabrication — phantom retired). Even at best-epoch, CTLE-E2E
> (FL 33.4) ≪ Check2HGI cat (FL 73–75) → "even at its best, CTLE ≪ ours." Recorded in RESULTS_BOARD §4 + `ctle.md`.
> No re-run. (Run-spec below kept for the record.)

The native-end-to-end half of the FL representation block (Blocker 1 covers the *frozen-SC* half, which the H100
got to **2/5** in PR #47).

**Why it's its own blocker (and small).** CTLE-E2E is CTLE in its *best* (fine-tuned) form — the honest "even at
its strongest CTLE is well below ours" point for the §6.1 representation block. It is a single cat-head E2E run
(7-class head → cheap, ~30 min–1 h, no wide-region accumulation). **Not load-bearing**: W3 (CTLE leak-clean) is
already closed at AL/AZ/Istanbul, so FL CTLE is *corroborating*; but it removes the last phantom path at the
headline state and completes the FL representation block.

**Run (seed 0 × 5f, FL, fp32):**
```bash
python scripts/baselines/ctle_e2e.py --state florida --seed 0 --folds 5
#  -> results/ctle_e2e_b1/florida/ctle_e2e_seed0.json   (AL template: results/ctle_e2e_b1/alabama/ctle_e2e_seed0.json = 21.24)
```
**Sanity:** native CTLE transformer fine-tuned end-to-end (NOT the frozen SC); healthy late best-epochs, fp32
(`DISABLE_AMP=1` from §0), no ~ep12 NaN. **Acceptance:** FL CTLE-E2E cat on disk; **Check2HGI (≫) > CTLE-E2E**;
present as the E2E rung beside the frozen CTLE-SC ladder ("even in its best E2E form CTLE is well below ours" —
never "we crushed CTLE"). **Record** the number in `docs/baselines/next_category/ctle.md` (the `e2e` note) +
RESULTS_BOARD §4; strike any lingering phantom 29.65. If you also have the H100's FL CTLE-SC 2/5, finishing it to
5f here (Blocker 1, §4) is the natural companion.

---

## 3 · Blocker 3 — Tbl 2 substrate contrast on ONE windowing (HGI category-STL under overlap)

> ✅ **STATUS — DONE (PR #52 merged 2026-06-26).** All 5 Gowalla states complete. HGI-overlap cat-STL:
> AL 26.56 / AZ 29.50 / FL 35.53 (#50) + **CA 32.31 / TX 32.48 (#52)** → substrate margin **+29.31 / +27.63 /
> +39.62 / +37.95 / +37.47** vs the board Check2HGI ceiling (RESULTS_BOARD §4). The PAPER_PLAN Tbl-2 "non-overlap"
> caveat is dropped (one windowing for the whole paper). **No re-run.** The §3.1–§3.5 builder/gate details below are
> kept as the reproduction record.

**Why.** Part 2 is on the overlap board; the Part-1 substrate table (Tbl 2: Check2HGI vs HGI category macro-F1 +
the per-visit share) is still on the **non-overlap** base. A reviewer asks "why two windowings?". The **Check2HGI**
arm under overlap is already the board STL cat ceiling (`RESULTS_BOARD §1`: AL 55.87 / AZ 57.13 / FL 75.15 /
CA 70.26 / TX 69.95). **Only the HGI arm under overlap is missing.**

### 3.1 Prereq — canonical CA/TX HGI embeddings (build on the A40, CPU)
HGI is **not bit-reproducible across machines**, so the whole 6-state HGI set must be one canonical A40 build
(AL/AZ/FL/GE already on the A40). CA/TX local copies are NOT canonical — rebuild on the A40:
```bash
mkdir -p /tmp/catx_hgi
setsid bash scripts/closing_data/build_catx_hgi.sh California > /tmp/catx_hgi/california.log 2>&1 < /dev/null &
setsid bash scripts/closing_data/build_catx_hgi.sh Texas      > /tmp/catx_hgi/texas.log      2>&1 < /dev/null &
#  -> output/hgi/{california,texas}/{embeddings,region_embeddings}.parquet  (HGI trains on CPU, GPU stays free)
```
Verify each: `[verify] region_embeddings ... nan=False std>0` and `embeddings.parquet exists=True` in the log.
(AL/AZ/FL HGI embeddings already on disk; Istanbul HGI — see 3.4.)

### 3.2 ⚠ OPEN BUILDER STEP — HGI inputs under the OVERLAP windowing
There is currently **no `hgi_dk_ovl` engine and no HGI-overlap input builder** (`build_overlap_probe_engine.py` is
Check2HGI-only; `setup_hgi_inputs.py` builds **non-overlap** POI-level HGI inputs). You must produce HGI category
inputs on the **same overlap windows** the board uses. The windows are **substrate-independent** — the per-user POI
sequences are identical to Check2HGI's; only the embedding lookup differs (HGI POI vector vs Check2HGI check-in
vector). Recommended build (a ~20-line script mirroring `setup_hgi_inputs.py`, but overlap):
- **Reuse** the frozen overlap sequences `output/check2hgi_dk_ovl/<state>/temp/sequences_next.parquet` (already
  stride-1 / MIN_SEQ=10 / emit_tail=False), OR call `generate_next_input_from_poi(state, EmbeddingEngine.HGI,
  stride=1, min_sequence_length=10, emit_tail=False)` (the builder accepts these — see
  `src/data/inputs/builders.py:256-268`).
- Map each `poi_k` through the **HGI** POI embedding lookup (`output/hgi/<state>/embeddings.parquet`).
- Write to a new engine dir, e.g. `output/hgi_dk_ovl/<state>/input/next.parquet`, and register an `HGI_DK_OVL`
  enum member in `src/configs/paths.py` (mirror how `CHECK2HGI_DK_OVL` routes), so `train.py --engine hgi_dk_ovl`
  resolves it.
- **Correctness gate (MUST pass before training):** the HGI-overlap and Check2HGI-overlap category inputs are
  windowed identically, so the row counts MUST match exactly:
  `len(output/hgi_dk_ovl/<state>/input/next.parquet) == len(output/check2hgi_dk_ovl/<state>/input/next.parquet)`
  (FL = 1,274,418; full Tbl-1 column in `STATS_T1.md`). If they differ, the windowing desynced — stop.

### 3.3 Run — HGI category STL ceiling under overlap (mirror of board Cell 1)
The Check2HGI cat ceiling was produced by (`scripts/closing_data/h100_state_cells.sh` Cell 1):
```bash
python scripts/train.py --task next --state <state> --engine <hgi-overlap-engine> \
    --model next_gru --folds 5 --epochs 50 --seed 0
#  score: python scripts/closing_data/score_stl_cat_ceiling.py <rundir> --tag <state>_hgi_ovl_cat
#  (reads metrics/fold*_next_val.csv, macro-F1 at f1-best epoch, fold-mean — cat is fp16-robust, A40 fine)
```
Run for **AL, AZ, FL, CA, TX, Istanbul**. Smoke first on AL (`--folds 1 --epochs 2` is fine for a *smoke*, but the
real ceiling is 5f/50ep). Cat is the 7-class head → cheap, no wide-region accumulation.

### 3.4 POI-pooled Check2HGI cat-STL under overlap (the per-visit-context share) — optional if time-pressed
For the Tbl-2 per-visit share (canonical check-in-level vs POI-mean-pooled), run the same Cell-1 recipe with the
**`check2hgi_pooled`** engine (the C4 POI-mean-pooled counterfactual, already an enum member). If time-pressed, the
per-visit *share* is windowing-robust enough to **footnote** as the non-overlap CH19 value — but the clean version
is to re-score. Istanbul HGI: build only if its row-count gate (3.2) passes on the Istanbul substrate; else footnote.

### 3.5 Acceptance + outputs
Tbl 2 = Check2HGI-cat-STL (board) vs **HGI-cat-STL (new, overlap)** at the 6 states, on ONE windowing; expect the
substrate margin to land at the same **~+15…+29** magnitude as the non-overlap measurement (sanity: if HGI cat ≈
Check2HGI cat, the substrate didn't swap — stop). Commit a small result JSON per state under
`docs/results/closing_data/baseline_compare/` (or a `tbl2_*` sidecar), update **Tbl 2 in `PAPER_PLAN.md`** (drop the
"non-overlap" caveat), and add a one-line finding to `RESULTS_BOARD.md`. (Fig 4 embedding-geometry is
windowing-robust → **no** re-score.)

---

## 4 · Blocker 4 — Region externals (faithful STAN ✅ AL/AZ/Istanbul; FL in-flight, then ReHDM)

> Full brief + acceptance gates:
> [`../../../articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md`](../../../articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md).

**Why.** Table 3's region externals must be on the right footing vs the board, and run FAITHFULLY (STAN's own
embeddings from raw — feeding it a pretrained embedding, `stl_hgi`, is non-standard and is now relegated to a
future-headroom signal, NOT a baseline). The earlier faithful-STAN v4 (AL 34.46 / AZ 38.96, below Markov) was an
under-training collapse artifact — **superseded.** HMT-GRN-style is the primary matched region external; faithful STAN
is secondary; ReHDM is the own-protocol reference.

### 4.1 Phase 1 — FAITHFUL STAN — ✅ DONE for AL/AZ/Istanbul (PR #53); FL in-flight
> ✅ **The re-implementation landed (PR #53).** Audited **v5**: all 6 faithfulness fixes (STAN-native prefix-expansion
> sequences, restored matching layer + interval embedding, constant-LR convergence; two-agent audit + GO review;
> ~85× optimized, audit≈compiled within 0.1 pp). **Converged** (best-epochs 5–12) and **clears the Markov floor +
> stays below our joint**: **AL 60.72 / AZ 49.86 / Istanbul 61.86** (reg Acc@10, seed 0 × 5f). v4 superseded.
>
> **Remaining A40 work — FINISH FL faithful-STAN:** the run is in-flight (fold-0 v6 ckpt Acc@10 0.7307). Complete the
> 5 folds, score, commit `docs/results/baselines/faithful_stan_florida_5f_200ep_v5_*.json`, fill the Table-3 FL STAN
> cell (currently `--‡` in-flight). **CA/TX faithful-STAN is infeasible at scale → footnote `†`** (HMT-GRN + Markov
> carry CA/TX). Recipe + flags: `research/baselines/stan/README_FAITHFUL_STAN.md` (constant-LR, prefix-expansion,
> `--compile`; fp32 on the A40 — bf16 backward grad-NaN risk at large C, so FL stays fp32).

### 4.2 Phase 2 — ReHDM-faithful (after Phase 1)
ReHDM in its **faithful** form (own architecture + raw inputs + own protocol). **Order: AL/AZ/Istanbul in parallel
first, then FL/CA/TX as possible.** AL/AZ/FL faithful already exist (66.06 / 54.65 / 65.68) — reuse/re-confirm.
Istanbul faithful is NEW and needs an **FSQ→mahalle region adapter** (else footnote not-available; do NOT use
`stl_check2hgi`). CA/TX faithful is heavy (~75–120 h/state) — as possible, else footnote infeasible-at-scale. ReHDM
is a **published-method reference under its own protocol**, gap-to-ceiling, never a paired cell.

### 4.3 Outputs
Commit per state under `docs/baselines/next_region/` + `docs/results/...`; update Table 3
(`articles/[mobiwac]/src/tables/tbl3_results.tex`, the STAN `‡` cells), `comparison.md`/`stan.md`/`rehdm.md`,
`RESULTS_BOARD.md §4`, and `PAPER_PLAN.md §5.4/§7`. Istanbul STAN comes from the M4 run, not here.

---

## 5 · Blocker 1 (FL CTLE-SC) — H100's job; A40 FALLBACK ONLY

Do **not** start this unless the orchestrator says the H100 stalled. If re-tasked to the A40, the FL run-spec is
the four commands below (device-agnostic; keep the fp32 AMP gate from §0). The two missing JSONs:
```bash
# base + per-fold log_T (idempotent guard: skip the re-window if input/{next,next_region}.parquet already fresh)
python scripts/mtl_improvement/build_overlap_probe_engine.py florida 1
python scripts/compute_region_transition.py --state florida --engine check2hgi_dk_ovl --per-fold --seed 0 --n-splits 5
# FL Check2HGI-SC comparand (the reference for the CTLE delta)
python scripts/closing_data/comparand_check2hgi_sc.py --state florida --folds 5 --heads cat reg
# FL CTLE-SC (frozen) 5f
python scripts/closing_data/mac_baseline_compare.py --state florida --baseline ctle --cells-root output --folds 5 --heads cat reg
#  -> docs/results/closing_data/baseline_compare/florida_ctle.json
# FL CTLE-E2E (CTLE's native fine-tuned form) 5f
python scripts/baselines/ctle_e2e.py --state florida --seed 0 --folds 5
#  -> results/ctle_e2e_b1/florida/ctle_e2e_seed0.json
```
Gates: CTLE-SC cat ≈ 17–20 vs Check2HGI-SC cat ≈ 75 (Δcat large, expected); min_seq=10
desync check `len(check2hgi_ctle/florida/input/next.parquet) == len(check2hgi_dk_ovl/florida/input/next.parquet)`.
Then strike the phantom fold-0/never-run numbers (27.98/73.00 SC, 29.65 E2E) in `RESULTS_BOARD.md §4`,
`H100_FL_BASELINES_FINDINGS.md`, `docs/baselines/next_category/ctle.md`. **Presentation:** CTLE as a *ladder*
(one-hot / skip-gram / POI2Vec / CTLE-SC frozen → our head) with CTLE-E2E beside it; **never "we crushed CTLE."**

---

## 6 · Pre-flight checklist (every reg/joint cell)
- [ ] `git pull` includes the freeze-reg-stream commit + PR #49; smoke passes (no reg-collate crash).
- [ ] AMP gate exported (`DISABLE_AMP=1 MTL_DISABLE_AMP=1`); best-epochs land **late** (no ep≤12 NaN collapse).
- [ ] `--folds 5` (NEVER `--folds 1` for a substrate/SC build); seed 0; user-disjoint folds.
- [ ] log_T fresh (mtime ≥ `next_region.parquet`) for any `--per-fold-transition-dir` cell; rebuild if stale.
- [ ] Blocker-3 row-count identity gate passes (HGI-overlap == Check2HGI-overlap row counts) before training.
- [ ] Commit per cell + a one-line finding; never cite a void fp16/bf16 JSON; branch + PR, no merge to main.

## 7 · Do NOT
Redo the Mac CPU work (Tbl-1 stats / TOST — PR #49). Build GE (out of scope) unless a dependency forces it. Run the
FL CTLE cells unless the H100 stalled and the orchestrator re-tasks you. Merge to main.
