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

**→ HAND TO A40 (user decision 2026-06-22 — anything >1 h on the Mac goes to A40):**
- **CTLE for FL, CA, TX** — all seeds×folds. `build_ctle_substrate.py --stride 1 --device cuda`
  on the gated-overlap base (the `--stride` + `--device` fixes are committed); leak-clean per
  fold; reuse the frozen check2hgi fold split. (AL/AZ/GA CTLE already built here = valid
  device-tolerant inputs; A40 only needs FL/CA/TX — or rebuild all 6 for uniformity.)
- **POI2Vec for CA, TX** — measured **~66 min/cell (CA) / ~85 min/cell (TX)** on the Mac
  (>1 h PER CELL) → A40. `build_poi2vec_substrate.py <state> --seed S --fold F --n-splits 5
  --epochs 30 --embed-dim 64 --user-dim 64 --theta 0.05 --route-count 4 --context-window 9
  --loss-form mixture --stride 1 --device cuda` (per-cell scratch OUTPUT_DIR + frozen-stage,
  see §2 gap-1 / the smoke script). On-device-accumulation fix is committed (helps CUDA too).
- **b2b skip-gram for CA, TX** — ~12 min (CA) / ~15 min (TX) per cell, ~4–5 h each batch
  (>1 h) → A40. `build_b2b_skipgram_substrate.py --state <st> --seed S --fold F --n-splits 5
  --epochs 5 --dim 64 --stride 1 --device cuda` (namespaced dir; no staging needed).

**MAC keeps (all <1 h/cell):** POI2Vec **AL/AZ/GA/FL** (running, CPU-parallel) — AL ~140s,
AZ ~256s, GA ~8min, FL ~29min per cell.

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

**MAC SCOPE: POI2Vec AL/AZ/GA/FL — ✅ COMPLETE (see §10).** Everything else
(CTLE FL/CA/TX, POI2Vec CA/TX, b2b CA/TX) is >1 h on the Mac → A40.
**STILL PARKED:** AL-ownership fold-1 MPS fit check (opt-in; AL v14 substrate present).

---

## 9 · A40 lane pickup (2026-06-22) — the heavy cells handed off in §8

A40 box (`study/board-m2pro` worktree at `/home/vitor.oliveira/PoiMtlNet-board-m2pro`,
running against the MAIN repo `/home/vitor.oliveira/PoiMtlNet`). Picking up the §8 hand-off:
**CTLE FL/CA/TX + POI2Vec CA/TX + b2b CA/TX**, full `{0,1,7,100}` × 5 folds (140 cells),
`--device cuda`, gated stride-1, train-only per fold.

### 9a · Feasibility (verified before launch)
- **GPU — fits.** A40 46 GB; the other lane's TX champion-G MTL run (board-a40 Task 2,
  PID 595258) holds ~23 GB and is on fold 4/5. The baseline builds are small. Per the A40
  "GPU cells strictly serial" rule, the CUDA fan-out **waits for that run to finish** rather
  than confound its timed insurance cell. torch pinned `2.11.0+cu128` ✔.
- **DISK — the §4 blocker, resolved by routing.** `/home` was at 97 % / 12 GB free; `/dados`
  (2.4 TB) is **not accessible** to this user (group `dados`, permission denied). Decision
  (user, 2026-06-22): **free `/home` + build on `/tmp` (80 GB on `/`) + migrate finished
  embeddings back to `/home` as space frees.**
  - Freed **16.1 GB** on `/home` → 28 GB: deleted the regennable `check2hgi_dk_ovl/california`
    `next.parquet`+`next_region.parquet` (CA is the A100's lane, unused on this box; breadcrumb
    `RECLAIMED_README.txt` + `build_overlap_probe_engine.py california 1` to regen).
  - Build buffer: `--work-dir /tmp/board_work` (transient next/next_region spikes, ~9 GB CA /
    ~21 GB TX per cell, trimmed per cell) + `--handoff-dir /tmp/board_baselines` (kept
    embeddings). `/home` is only READ (frozen substrate via `--output-root`), never filled by
    a build.

### 9b · Driver changes (committed this PR)
The fan-out driver was CPU-only and pinned to the checkout's own `output/`. Added `--device`
(cuda, threaded to all three builders), `--output-root` (read frozen substrate + write
deliverables to the MAIN repo `output/` while running from the worktree), `--handoff-dir`
(route embeddings to `/tmp`, migrate later). Fixed a pre-existing `--dry-run` NameError
(`fg`→`fg_ext`). M2 Pro CPU path unchanged (defaults preserve prior behavior). The worktree
`.venv` is a gitignored symlink to the main repo venv.

**Validated:** `--help`, full-{0,1,7,100} CTLE dry-run (60 cells, main-repo output read), one
real AL b2b CPU cell end-to-end (28 s → embeddings + provenance + leak marker in `/tmp` →
migrated to repo `output/` → removed as a throwaway plumbing test).

### 9c · Launch command (run once the GPU frees)
```bash
cd /home/vitor.oliveira/PoiMtlNet-board-m2pro
PYTHONPATH=src .venv/bin/python scripts/closing_data/m2pro_baseline_fanout.py \
    --baselines ctle poi2vec b2b \
    --states florida california texas \   # CTLE wants FL+CA+TX; b2b/poi2vec only CA/TX (filter per §8)
    --seeds 0 1 7 100 --folds 0 1 2 3 4 \
    --device cuda \
    --output-root /home/vitor.oliveira/PoiMtlNet/output \
    --handoff-dir /tmp/board_baselines --work-dir /tmp/board_work \
    --workers 1 --threads 24 --disk-floor-gb 15
```
> NOTE: b2b/poi2vec are CA/TX only (FL stays on the Mac per §8); run CTLE (FL/CA/TX) and
> b2b/poi2vec (CA/TX) as separate invocations so the state lists stay correct.

### 9d · Open notes for the orchestrator (comparison step, CUDA-only)
- **min_seq=5 vs board's 10.** All baseline builds carry `min_sequence_length=5` (global
  default); §1 proved this a **no-op at stride-1** (geometry → identical rows to min_seq=10).
  Row-alignment holds. If the matched-head comparison runs `train.py` with `MTL_STRICT=1`, the
  `_warn_if_ungated_overlap` guard reads the engine sidecar's `min_sequence_length` — the
  baseline engines will report 5. This is the established lane convention (B2c + small-state
  builds are all min_seq=5); flagged so the comparison either accepts 5-at-stride-1 or restamps.
  **⚠ Caveat (verify at comparison):** the equivalence is established by equal row *count* vs the
  v11 `check2hgi` substrate; the row-*set* identity against the actual board base
  `check2hgi_dk_ovl` (design-k re-windowed, NOT v11) is **inferred, not verified** here. The CUDA
  comparison reads both sides' rows and will hard-test it — if a baseline cell's row set diverges
  from `dk_ovl`'s, the paired Δ is invalid for that cell. (Geometry says they match; confirm.)
- **Deliverables persistence.** Embeddings accumulate on `/tmp` (non-persistent). Migration to
  `/home` is best-effort as space frees; full `{0,1,7,100}` deliverables ≈ 78 GB and exceed
  `/home`'s post-reclaim 28 GB. The durable fix is `/dados` access — surfaced to the user.

---

## 10 · MAC LANE COMPLETE (2026-06-22) — 226 cells; all-internal FL finish
**Mac lane done** (`M2PRO_MANIFEST.md`, 226 cells): B2c 6/6 · b2b 80/80 (AL/AZ/GA/FL) ·
CTLE 60 (AL/AZ/GA) · POI2Vec 80/80 (AL/AZ/GA/FL).

**FL POI2Vec finished via all-internal execution** (the external SSD dropped off the bus a
**3rd** time under FL load). Pattern: copy the frozen FL substrate → internal
(`/private/tmp/fl_src`, one-time ~1.1 GB SSD read), run with `--output-root /private/tmp/fl_src
--work-dir /private/tmp/board_work --handoff-dir /private/tmp/board_baselines` (reads + scratch
+ handoff ALL internal → **zero per-cell SSD I/O**), then move the embeddings internal→SSD
(the single SSD write). SSD touched exactly twice (one read, one write) — the failing drive
can't be triggered by the build load.

**Finding — internal storage is a RELIABILITY win, NOT a speed win for POI2Vec.** Measured FL
POI2Vec ≈ **~60 min/cell on internal ≈ ~51 min on the SSD** (3-way). POI2Vec is compute-bound
on the overlap-area `phi` step (O(POIs); 76,544 POIs at FL) + CBOW build — NOT I/O-bound (the
30 training epochs are ~5 min total, ~10 s/epoch). Consequence for the A40-routed cells:
- **CTLE** (transformer pretrain, compute-bound) and **POI2Vec CA/TX** (more POIs → slower
  `phi`) stay >1 h/cell → **A40** — internal storage would not rescue them.
- Only **b2b CA/TX** is I/O-bound (the stride-1 next-build) and would benefit from internal,
  but it's trivial on the A40 GPU and already queued there.

**STILL ON A40:** CTLE FL/CA/TX · POI2Vec CA/TX · b2b CA/TX.
**STILL PARKED:** AL-ownership fold-1 MPS fit check (opt-in).

---

## 10 · A40 lane EXECUTION RESULTS (2026-06-22 → 23)

The §9 plan executed. Outcome by baseline (all on the gated stride-1 overlap base, train-only
per fold, `--device cuda`, → main-repo `output/board_baselines/`; manifest regenerated → **326 cells**):

| baseline | A40 scope | status |
|---|---|---|
| **CTLE** | FL/CA/TX × {0,1,7,100} × 5f = 60 | ✅ **60/60** (0 fails) |
| **b2b** | CA/TX × {0,1,7,100} × 5f = 40 | ✅ **40/40** (0 fails) |
| **POI2Vec** | CA/TX — **REDUCED to 10/state** (seeds {0,1} × 5f = 20), per user decision | ⏳ in progress |

Combined with the Mac lane (AL/AZ/GA/FL), the board now has **b2b + CTLE complete for all 6 states**
and **b2c for all 6**; POI2Vec CA/TX is the only remainder.

### What it took
- **GPU/disk feasibility, shared box.** CTLE large-state runs are GPU-heavy (~32 GB/cell) → kept
  **serial**; b2b is light → ran **parallel** alongside. The box is **shared** (users `felipe.sousa`,
  `lucas.lana` ran 8–26 GB Jupyter/`main.py` jobs intermittently) — handled with pause/resume +
  GPU-near-ceiling watchdogs; no OOM hit our cells or theirs. Disk: built on `/tmp` then migrated to
  `/home`; later (another user freed ~67 GB of the shared `/home`) wrote POI2Vec direct to `/home`.
- **POI2Vec was INFEASIBLE until a perf fix.** `build_poi_routes` (phi) was an O(n_poi·n_leaf) pure-
  Python loop = 169K × 65K ≈ 11B calls ≈ **~3 h/cell** at CA scale (TX worse), recomputed per cell —
  8 cells sat 90 min in phi with the GPU idle. **Vectorized it (numpy, byte-identical — verified
  leaf_ids+phi bit-exact incl. NaN/edge/tie cases), committed `7a6f30e3`** → phi now **seconds**.
  POI2Vec then relaunched and bumped to **workers=10** (phi is no longer the bound; CBOW training
  ~30–40 min/cell + next-build ~20 min/cell now dominate).
- **Scope reduced to 10/state** (seeds {0,1}) for POI2Vec per user decision (vs the board's
  {0,1,7,100}); manifest "expected" stays 20, so POI2Vec CA/TX will read **10/20** = the deliberate
  reduced scope, not a shortfall.

### Further POI2Vec optimization — evaluated, deferred (see `POI2VEC_OPTIMIZATION_EVAL.md`)
The remaining quality-safe optimizations (per-state tree/phi/edge **cache**; **next-build**
vectorize/parallelize) save <1 % on this workers=10 reduced run (post-phi the cell is ~95 %
training+next-build, and at workers=10 the next-build is already overlapped) at real risk to the
SHARED `core.py`. **Deferred to the full-board launch** (hundreds of cells → tens of hours of
next-build) each gated by a byte-equivalence check. The quality-RISK options (batch↑/fp16/fewer-epochs)
are rejected — they change the trained `poi_embed`.

### Commits (A40 lane on this PR)
driver `--device/--output-root/--handoff-dir` · `SUBSTRATE_VERSION_MAP.md` (+§4b no-v11-contamination)
· POI2Vec phi vectorization `7a6f30e3` · `POI2VEC_OPTIMIZATION_EVAL.md` · manifest cross-box merge +
regen (326 cells).
