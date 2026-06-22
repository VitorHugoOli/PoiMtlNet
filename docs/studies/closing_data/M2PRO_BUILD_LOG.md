# M2 Pro lane — baseline-embedding build log (`study/board-m2pro`)

> Lane scope: build the four LIGHT substrate-column baseline embeddings (B2c one-hot64,
> B2b skip-gram, POI2Vec faithful, CTLE) on the **gated stride-1 overlap** base, **train-only
> per fold**, states × {0,1,7,100} × 5 folds. Device-tolerant INPUTS the CUDA board consumes;
> the matched-head COMPARISON is **CUDA-only** (device-class rule). See `HANDOFF_BOARD_M2PRO.md`.
>
> Machine: M4 Pro running the M2 Pro lane. **24 GB RAM**, output SSD **~83 GB free at start (91% full)**.

## 1 · Stride-1 gating gate — PASSED (and it is a no-op, not a code change)
The handoff says "just pass `--stride 1`". Verified correct:
- `_resolve_emit_tail(None, stride=1)` ⇒ `emit_tail=False` (the M1 skew gate) automatically.
- `min_sequence_length` stays the global **5**, but this is a **proven no-op at stride-1**: with
  `emit_tail=False` a full window needs ≥ `SLIDE_WINDOW+1 = 10` check-ins to emit *any* row, so
  all sub-10-checkin users are dropped by **geometry**, identical to the board base's `min_seq=10`
  (`build_overlap_probe_engine.py` `BOARD_MIN_SEQ=10`).
- **Empirical (alabama):** stride-1 gated rows are **96,326** at both `min_seq=5` and `min_seq=10`
  (`Δ=0`). B2c/B2b/POI2Vec/CTLE all emit 96,326 next/next_region at `--stride 1`. ✔ row-aligned.

## 2 · Three gaps vs the handoff (found + resolved)
1. **POI2Vec §3c command is incomplete.** It reads the frozen check2hgi substrate through the
   *same* scratch `OUTPUT_DIR` it writes to, so a bare `OUTPUT_DIR=/tmp/bl_poi2vec …` fails
   (`FileNotFoundError: …/check2hgi/<state>/input/next.parquet`). Fix = the **smoke-script staging**
   per cell: `cp` frozen `temp/sequences_next.parquet`, `input/next.parquet`, `embeddings.parquet`
   into the scratch `check2hgi/<state>/`; symlink `region_embeddings.parquet` + `temp/checkin_graph.pt`.
   (B2b/CTLE write to **namespaced** dirs so they don't need staging; POI2Vec uses the CHECK2HGI
   probe-engine hack → needs it.)
2. **RAM-headroom guard too aggressive for 24 GB.** `_guard_cpu_resident_ram` wants 16 GB headroom;
   the tiny fold-split read (≤0.1 GB) trips it under concurrent builds. Pass `MTL_RAM_HEADROOM_GB=2`.
3. **CTLE had no `--stride` flag — REAL DEFECT (code fix in this PR).** `build_ctle_substrate.py`
   called `generate_next_input_from_checkins(state, CTLE_ENGINE)` with no stride → silent **stride-9
   non-overlap** (12,709 rows), so it would NOT row-align with the board base on CUDA. The §3d handoff
   assumption ("CTLE inherits stride-1 from the engine's windowing") was wrong — there is no windowing
   to inherit. **Fix:** added `--stride` (default `None`), threaded to
   `generate_next_input_from_checkins(..., stride=args.stride)`.
   **Verified AL `--stride 1`:** `embeddings.parquet` stays **113,846** (contextual substrate is
   windowing-independent), `next`/`next_region` → **96,326** aligned, cols `userid,placeid,category,datetime`.
   ⚠ The pre-freeze "CTLE adversarially audited" pass never exercised stride-1 (that is P3) — this is a
   new defect surfaced by the board build, not a config slip.

## 3 · Per-cell timings (alabama, smallest state; CPU/MPS, fp32)
| Baseline | time/cell | cells (6 states × 4 seeds × 5 folds) | notes |
|---|---|---|---|
| B2c one-hot64 | ~17s | **6** (per-state, fold-independent) | trivial seeded random projection |
| B2b skip-gram | ~27s | 120 | namespaced dir, leak assert fires |
| POI2Vec faithful | ~140s | 120 | needs staged scratch + headroom override |
| CTLE | ~52s (AL, 10ep, CPU) | 120 | heaviest; per-fold leak-clean; scales with #trajs/vocab |
Large states (CA ≈ 2.9M next rows, TX similar) are ~10–25× the AL wall-time per cell.

## 4 · DISK is the binding constraint → keep **embeddings only**, regen next/next_region on CUDA
Measured (CA, 2.9M-row stride-1 cell): `next.parquet` **4.8 GB** + `next_region.parquet` **4.1 GB**
vs `embeddings.parquet` **536 MB**. Keeping next/next_region for ~360 heavy cells is physically
impossible on an 83 GB-free disk (CA alone ≈ 180 GB).
- **The `embeddings.parquet` IS the deliverable substrate.** `next`/`next_region`/`sequences` are
  **deterministic regens** (fixed fold split + `--stride 1`), so the CUDA board regenerates them.
- **Per-cell policy:** build next/next_region (fires the row-align + val⊥train leak asserts) →
  keep `embeddings.parquet` + region symlink + `*_build_provenance.json` + leak/fold markers →
  **delete** next/next_region/sequences. Avoids accumulation; CA build still spikes +~9 GB transiently.
- **CUDA regen recipe** (per consumed cell): point `train.py --engine <baseline>` at the cell's
  `embeddings.parquet`, run `generate_next_input_from_checkins(state, ENGINE, stride=1)` +
  `build_next_region_for(state, ENGINE)`; reuse `--per-fold-transition-dir output/check2hgi/<state>`.
- **⚠ OPEN for user/orchestrator:** even embeddings-only for CA+TX × 3 heavy baselines ≈ 60–70 GB,
  perilously close to 83 GB free. Needs a disk decision (free space / external transfer target /
  scope) before the large-state heavy fan-out — see the lane status note.

## 5 · B2c one-hot64 build status (stride-1, per-state)
| state | next/next_region rows | embeddings | next.parquet | status |
|---|---|---|---|---|
| alabama | 96,326 | 16 MB | 115 MB | ✔ |
| arizona | 200,895 | 34 MB | 252 MB | ✔ |
| california | 2,925,466 | 536 MB | 4.8 GB | ✔ |
| florida | 1,274,418 | ~170 MB | (trimmed) | ✔ |
| georgia | 349,191 | ~50 MB | (trimmed) | ✔ |
| texas | 3,830,414 | ~700 MB | (trimmed) | ✔ |
**B2c COMPLETE for all 6 states** (next/next_region trimmed to embeddings-only).
region/poi embeddings symlinked from frozen check2hgi; `log_T` reused via
`--per-fold-transition-dir output/check2hgi/<state>` at the (CUDA) train step.

## 6 · Fan-out driver + launch
`scripts/closing_data/m2pro_baseline_fanout.py` drives B2b/POI2Vec/CTLE per
`(state,seed,fold)` (staging for POI2Vec/CTLE, `--stride 1`, `MTL_RAM_HEADROOM_GB=2`,
row-align asserts, embeddings-only keep to `output/board_baselines/<baseline>/<state>/
s<seed>_f<fold>/`, disk-floor=18 GB stop, resumable). Validated on 3 AL cells
(b2b 25s / poi2vec 138s / ctle 63s) — all 113,846-row distinct substrates.

**LAUNCHED (background, serial):** small/mid states **AL, AZ, GA, FL** × {0,1,7,100} ×
5f × 3 baselines = **240 cells** (`/tmp/fanout_small.log`). Serial to avoid the RAM
contention that killed a concurrent TX build on the 24 GB box.

## 7 · Hardening after three infra incidents (2026-06-22) + driver knobs
The fan-out exposed three independent failure modes on this 24 GB Mac; each fixed:
1. **External SSD dropped off the bus twice under sustained write load** (Errno 5,
   then full unmount). Fix: **`--work-dir`** routes ALL heavy transient writes
   (b2b native via `--read-output-dir`, POI2Vec/CTLE scratch, logs) to the **internal**
   disk; only the small final embeddings are written to the external handoff. Disk-floor
   now guards **both** disks. (Commit `0ce0aa10`.)
2. **CPU oversubscription** — at `--workers 6` each torch build grabbed all 12 cores
   (load avg ~22, ctle cell 176s→1438s). Fix: **`--threads`** pins OMP/MKL/OpenBLAS/torch
   so `workers*threads ≈ cores`. (Commit `466dd228`.)
3. **RAM exhaustion → box crash** — 6 concurrent builds × GB-scale data (poi2vec CBOW
   examples / ctle ~6.6 GB) blew 24 GB. Fix: **`--ram-floor-gb`** gate (a build waits
   before launch until N GB free) + memory-safe defaults **3 workers × 4 threads**.
   (Commit `211bdc87`.) Plus **`PYTHONUNBUFFERED`** so long builds don't *look* hung
   (their epoch prints were block-buffered to the log). (Commit … unbuffered.)
All commits are pushed; build artefacts are regennable, so nothing was lost across
the incidents — only wall-time.

## 8 · DEVICE SPLIT (user decision 2026-06-22): Mac = light, CUDA = heavy CTLE
**Measured wall:** CTLE has **no GPU path on this Mac** (its code only knows cuda/cpu;
MPS unsafe for its RNG) and uses **~1 core effectively** on CPU → **~110 min + 6.6 GB
RAM per Florida cell**; large-state CTLE (FL+CA+TX × 20) ≈ **~68 h babysat**, RAM/crash-
prone. On a GPU it's minutes. So per the handoff's intended split:

**MAC OWNS (light / feasible) — building here:**
- **B2c one-hot64** — ✅ all 6 states.
- **b2b skip-gram** — small states ✅ (80/80); **CA/TX to build here** (b2b is light;
  next-build churn goes to the internal work-dir).
- **POI2Vec** — small states (AL/AZ/GA) building; **FL** to build here (medium).
- **CTLE** — small states ✅ (AL/AZ/GA 60/80).

**→ HAND TO A40 (user decision 2026-06-22 — Mac gives up CTLE entirely):**
- **CTLE for FL, CA, TX** — all seeds×folds. Build on the A40 GPU via
  `build_ctle_substrate.py --stride 1 --device cuda` on the gated-overlap base
  (the `--stride` + `--device` fixes are committed); leak-clean per fold; reuse the
  frozen check2hgi fold split. (AL/AZ/GA CTLE are already built here = valid
  device-tolerant inputs; A40 only needs FL/CA/TX — or rebuild all 6 for uniformity.)
- **POI2Vec for CA, TX** — heavy CBOW build; candidate for A40 too (assess on Mac first).

### MPS finding (why CTLE is NOT worth keeping on the Mac)
Audited + made the builders MPS-correct (CTLE `--device` auto-MPS; **on-device loss
accumulation, no per-batch `.item()` sync** — matches canonical `mtl_cv.py:818`;
data confirmed on-device). But MPS does **not** rescue large-state CTLE:
- **Single GPU → cannot parallelize cells**; the CPU path runs 3–4 in parallel.
- Small transformer (4-layer, bs 256) → MPS per-op overhead eats the gain
  (**AL CTLE: 40 s MPS ≈ 52 s CPU**; clean FL ≈ ~28–32 min/cell, ~2–3× per-cell only).
- Net **MPS-serial ≈ CPU-parallel** in throughput → ~50–60 h either way. So large-state
  CTLE goes to the A40 (real datacenter GPU); the Mac's many-small-cell batches stay
  **CPU-parallel** (better throughput than MPS-serial). MPS only wins for a *single* big cell.
- Unified-memory caveat: MPS shares the 24 GB system RAM (the `psutil` RAM-floor gate
  captures it); never run ad-hoc MPS jobs alongside the managed CPU fan-out (that drove
  RAM to 23 % and tripped a build guard).

**MAC NOW: finish POI2Vec (AL/AZ/GA/FL, CPU-parallel, running) + b2b CA/TX (light).**
**STILL PARKED:** AL-ownership fold-1 MPS fit check (opt-in; AL v14 substrate present).
