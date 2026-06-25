# HANDOFF — A40 · remaining GPU work for the MobiWac close · self-contained · 2026-06-25

> **You are the A40. Read this file first.** It is the master queue for the GPU-blocked items in
> [`../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md`](../../../articles/[mobiwac]/CLOSE_BLOCKERS_HANDOFF.md).
> The two **CPU-only** items (Blocker 3 Tbl-1 stats + the small-state region **TOST**) are **already DONE** on the
> Mac (PR #49, branch `closing-data/tbl1-overlap-and-region-tost`) — do not redo them. Your plate is what remains:
> ✅ **W6 probe DONE (PR #48): trunk, not transfer** (probe cat ≈ full-MTL, ≫ ceiling; §6.2 BACKED). Your plate is
> now **Blocker 2** (HGI category re-score under overlap, incl. the canonical CA/TX HGI build) — the last item
> needing new GPU runs. **Blocker 1** (FL CTLE) is the **H100's** job (PR #47 landed 2/5; W3 already closed at
> AL/AZ/Istanbul) — commands here are a fallback only.
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

## Priority order (UPDATED 2026-06-25 — W6 DONE)
1. ✅ **W6 encoder-isolation probe — DONE (PR #48): trunk, not transfer.** §1 below is the record; no re-run.
2. **Blocker 2 (NOW #1, the continuing work)** — HGI category STL under overlap. Needs the canonical CA/TX HGI
   build first (CPU-bound HGI training → kick off early; the GPU stays free meanwhile). See §2.
3. **Blocker 1 fallback** — only if the H100's FL CTLE never reaches 5f (it has 2/5; W3 already closed at AL/AZ/Istanbul).

---

## 1 · W6 probe — ✅ DONE (PR #48): TRUNK, not transfer
Result: probe cat (region frozen) AL 63.50 / AZ 63.67 / FL 79.79 ≈ full-MTL cat (±0.3 pp), ≫ STL ceiling
(+4.6…+7.6) → the joint category win is the shared trunk, not region→category transfer. Recorded in
`W6_ENCODER_ISOLATION.md` + RESULTS_BOARD §1c. **No re-run.** (Run-spec below kept for the record.)

Run-spec is complete and self-contained in **[`HANDOFF_A40_W6_PROBE.md`](HANDOFF_A40_W6_PROBE.md)** (the summary is
also in [`BASELINE_A40.md`](BASELINE_A40.md)). One line:
```bash
MODE=smoke bash scripts/run_freeze_reg_probe.sh                          # AL 1f×2ep sanity
STATES="alabama arizona florida" bash scripts/run_freeze_reg_probe.sh    # the real run, seed 0 × 5f, fp32
```
Sanity gate: the first-fold `[per-head-LR]` line MUST show the reg group(s) at **0 trainable params** (else it
aborts with a RuntimeError — do not paper over it). Read **cat macro-F1 only**; reg is meaningless here. Verdict
table + outputs (`W6_ENCODER_ISOLATION.md` + a result JSON per state + a RESULTS_BOARD §1c line) are in that doc.

---

## 2 · Blocker 2 — Tbl 2 substrate contrast on ONE windowing (HGI category-STL under overlap)

**Why.** Part 2 is on the overlap board; the Part-1 substrate table (Tbl 2: Check2HGI vs HGI category macro-F1 +
the per-visit share) is still on the **non-overlap** base. A reviewer asks "why two windowings?". The **Check2HGI**
arm under overlap is already the board STL cat ceiling (`RESULTS_BOARD §1`: AL 55.87 / AZ 57.13 / FL 75.15 /
CA 70.26 / TX 69.95). **Only the HGI arm under overlap is missing.**

### 2.1 Prereq — canonical CA/TX HGI embeddings (build on the A40, CPU)
HGI is **not bit-reproducible across machines**, so the whole 6-state HGI set must be one canonical A40 build
(AL/AZ/FL/GE already on the A40). CA/TX local copies are NOT canonical — rebuild on the A40:
```bash
mkdir -p /tmp/catx_hgi
setsid bash scripts/closing_data/build_catx_hgi.sh California > /tmp/catx_hgi/california.log 2>&1 < /dev/null &
setsid bash scripts/closing_data/build_catx_hgi.sh Texas      > /tmp/catx_hgi/texas.log      2>&1 < /dev/null &
#  -> output/hgi/{california,texas}/{embeddings,region_embeddings}.parquet  (HGI trains on CPU, GPU stays free)
```
Verify each: `[verify] region_embeddings ... nan=False std>0` and `embeddings.parquet exists=True` in the log.
(AL/AZ/FL HGI embeddings already on disk; Istanbul HGI — see 2.4.)

### 2.2 ⚠ OPEN BUILDER STEP — HGI inputs under the OVERLAP windowing
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

### 2.3 Run — HGI category STL ceiling under overlap (mirror of board Cell 1)
The Check2HGI cat ceiling was produced by (`scripts/closing_data/h100_state_cells.sh` Cell 1):
```bash
python scripts/train.py --task next --state <state> --engine <hgi-overlap-engine> \
    --model next_gru --folds 5 --epochs 50 --seed 0
#  score: python scripts/closing_data/score_stl_cat_ceiling.py <rundir> --tag <state>_hgi_ovl_cat
#  (reads metrics/fold*_next_val.csv, macro-F1 at f1-best epoch, fold-mean — cat is fp16-robust, A40 fine)
```
Run for **AL, AZ, FL, CA, TX, Istanbul**. Smoke first on AL (`--folds 1 --epochs 2` is fine for a *smoke*, but the
real ceiling is 5f/50ep). Cat is the 7-class head → cheap, no wide-region accumulation.

### 2.4 POI-pooled Check2HGI cat-STL under overlap (the per-visit-context share) — optional if time-pressed
For the Tbl-2 per-visit share (canonical check-in-level vs POI-mean-pooled), run the same Cell-1 recipe with the
**`check2hgi_pooled`** engine (the C4 POI-mean-pooled counterfactual, already an enum member). If time-pressed, the
per-visit *share* is windowing-robust enough to **footnote** as the non-overlap CH19 value — but the clean version
is to re-score. Istanbul HGI: build only if its row-count gate (2.2) passes on the Istanbul substrate; else footnote.

### 2.5 Acceptance + outputs
Tbl 2 = Check2HGI-cat-STL (board) vs **HGI-cat-STL (new, overlap)** at the 6 states, on ONE windowing; expect the
substrate margin to land at the same **~+15…+29** magnitude as the non-overlap measurement (sanity: if HGI cat ≈
Check2HGI cat, the substrate didn't swap — stop). Commit a small result JSON per state under
`docs/results/closing_data/baseline_compare/` (or a `tbl2_*` sidecar), update **Tbl 2 in `PAPER_PLAN.md`** (drop the
"non-overlap" caveat), and add a one-line finding to `RESULTS_BOARD.md`. (Fig 4 embedding-geometry is
windowing-robust → **no** re-score.)

---

## 3 · Blocker 1 (FL CTLE) — H100's job; A40 FALLBACK ONLY

Do **not** start this unless the orchestrator says the H100 stalled. If re-tasked to the A40, the FL run-spec is
`BASELINE_H100.md §2` verbatim (device-agnostic; keep the fp32 AMP gate). The two missing JSONs:
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
Gates (`BASELINE_H100.md §3`): CTLE-SC cat ≈ 17–20 vs Check2HGI-SC cat ≈ 75 (Δcat large, expected); min_seq=10
desync check `len(check2hgi_ctle/florida/input/next.parquet) == len(check2hgi_dk_ovl/florida/input/next.parquet)`.
Then strike the phantom fold-0/never-run numbers (27.98/73.00 SC, 29.65 E2E) in `BASELINE_H100.md`,
`H100_FL_BASELINES_FINDINGS.md`, `docs/baselines/next_category/ctle.md`. **Presentation:** CTLE as a *ladder*
(one-hot / skip-gram / POI2Vec / CTLE-SC frozen → our head) with CTLE-E2E beside it; **never "we crushed CTLE."**

---

## 4 · Pre-flight checklist (every reg/joint cell)
- [ ] `git pull` includes the freeze-reg-stream commit + PR #49; smoke passes (no reg-collate crash).
- [ ] AMP gate exported (`DISABLE_AMP=1 MTL_DISABLE_AMP=1`); best-epochs land **late** (no ep≤12 NaN collapse).
- [ ] `--folds 5` (NEVER `--folds 1` for a substrate/SC build); seed 0; user-disjoint folds.
- [ ] log_T fresh (mtime ≥ `next_region.parquet`) for any `--per-fold-transition-dir` cell; rebuild if stale.
- [ ] Blocker-2 row-count identity gate passes (HGI-overlap == Check2HGI-overlap row counts) before training.
- [ ] Commit per cell + a one-line finding; never cite a void fp16/bf16 JSON; branch + PR, no merge to main.

## 5 · Do NOT
Redo the Mac CPU work (Blocker 3 / TOST — PR #49). Build GE (out of scope) unless a dependency forces it. Run the
FL CTLE cells unless the H100 stalled and the orchestrator re-tasks you. Merge to main.
