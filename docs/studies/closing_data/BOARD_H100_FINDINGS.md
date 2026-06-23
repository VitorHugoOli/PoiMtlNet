# Board lane H100 — findings & resume notes

> Branch `study/board-h100`, PR #32. Companion to `BOARD_H100_STATUS.md` (setup/decision).

## ⚠ INTERRUPTED 2026-06-22 ~05:45 UTC — GPU studio stopped
The Lightning Studio's **H100 was stopped/restarted** (SIGKILL / exit 137); the shell is now on a CPU-only
fallback VM (`torch.cuda.is_available()=False`, 4 CPU / 14 GB RAM, no `nvidia-smi`). **The persistent
filesystem survived** — rebuilt v14 substrates (FL+CA), gated-overlap engines (FL+CA), seeded log_T, and the
FL STL-ceiling results are all intact. **Resume requires restarting the H100 studio** (see Resume below).
> Operational note: stopping this studio detaches the GPU but keeps `/teamspace/studios/this_studio`.

## ⭐ CURRENT STATE (2026-06-22, frozen v14 substrate — supersedes the rebuilt-substrate results below)
Board switched to the **frozen v14 substrate** (verified == V14_HASH_MANIFEST for all 6 states). FL+CA cells
re-run on frozen. Studio restarted 3× (~every 5h); FL Task 1 finished + committed before the 3rd restart; CA
recovered via p1 checkpoint resume.

**HANDOFF_BOARD_A100 — Task 1 (FL) ✅ COMPLETE** (frozen v14, seed 0, 5f, H100 compiled+tf32):
| FL | STL ceiling | MTL | Δ | vs A40 (same frozen substrate) |
|---|---|---|---|---|
| cat macro-F1 | 75.147 | 78.517 | **+3.37** (beats) | MTL −0.49 |
| reg FULL top10 | 76.712 | 75.422 | **−1.29** (within δ_reg=2pp) | MTL −0.08; ceiling 76.7123 vs 76.7138 (−0.0015) |

Device-internal Δreg −1.29 reproduces the A40's −1.21 (systematic ~−1.2 gated-overlap reg gap). Reg
cross-arch reproduction is ~exact; cat MTL −0.49 is fold-std + volatile cat best-epoch ([8,8,43,20,9]).
Result JSON: `docs/results/closing_data/h100/florida_s0_board.json`.

**Task 2 (CA) — IN PROGRESS:**
- STL reg ceiling ✅ **63.4848 ± 0.31** (full top10, 8501 regions; per-fold 63.25/63.45/63.38/63.33/64.02).
  Resumed cleanly from a 3-fold checkpoint after the restart. `docs/results/P1/region_head_california_..._s0.json`.
- CA champion-G MTL **COLLAPSES at ep30** (both heads → ~3% floor) → the −5.23 pp "Δreg breach" it produced is
  a **collapse ARTIFACT — VOID, do not cite.** **It is NOT a tf32 issue:** fp32 collapses at the identical ep30
  (precision refuted). Deterministic NaN event at ep30, CA-scale; under audit. Full writeup:
  [`CA_MTL_DIVERGENCE.md`](CA_MTL_DIVERGENCE.md). CA Δreg blocked pending a real fix (LR / NaN-guard / anomaly
  diagnosis). CA reg ceiling 63.48 stands.
- ⚠ **FL caveat:** FL Task 1 hit the same instability in 1/5 folds at ep50 (late, benign; scorer used the
  pre-collapse peak) → FL reg 75.42 slightly underestimated; FL MTL re-run recommended once a fix exists.
- **STL ceilings are CLEAN** (single-task heads stable) — no re-run needed.

---

## Results so far (seed 0, 5 folds, gated-overlap `check2hgi_dk_ovl`, compiled+tf32) — REBUILT-substrate, superseded

### FL — A40 (frozen substrate) vs H100 (rebuilt substrate)
| FL cell | A40 frozen | H100 rebuilt | Δ |
|---|---|---|---|
| **STL reg ceiling** (full top10, fp32) | 76.7138 | **76.7674** | **+0.05** ✅ |
| **STL cat ceiling** (macro-F1) | *(not run by A40)* | **75.46** | ref ~75.2 ✅ |
| MTL cat macro-F1 | 79.0083 | ⏳ died fold1/ep25 (best C77.84) | re-run |
| MTL reg full top10 | 75.5000 | ⏳ died fold1/ep25 (best N75.43) | re-run |

**Key validation:** despite the rebuilt substrate being byte-different from the frozen artifact
(`V14_REBUILD_H100_PROVENANCE.json`), the FL STL reg ceiling reproduces the A40's to **+0.05 pp** — the
own-states rebuild is functionally equivalent at the metric level. A40 FL device-internal Δreg (MTL−STL) =
**−1.21 pp** (within δ_reg=2pp); the H100 MTL re-run will confirm its own Δ.

## Lessons — parallelism on a GPU-rich / host-RAM-tight box (H100 80GB GPU, 108GB host)
1. **GPU memory was never the limit** (cells ~17 GB, util ~15%, transfer-bound). But the A40's "serial" rule
   still matters here for a different reason:
2. **GPU contention pushes datasets CPU-resident.** `folds._dataset_device` auto-fit decides at fold start
   using *free* VRAM; when cells share the GPU, the loser keeps its dataset on CPU → per-batch transfer
   (slower). For peak speed, run the big cells SERIALLY so each gets a GPU-resident dataset.
3. **The p1 STL-reg val-metric is the OOM hazard, not the model.** It accumulates the full val logit
   `[N_val,C]` and `torch.cat`s it (~2× = ~37 GB GPU at CA's C=8501). Options: GPU path (needs the cell ~alone
   on the GPU) or CPU path (`P1_CHUNK_VAL_METRIC=1`, frees GPU but spikes ~20 GB host — bad on this RAM-tight
   box). CA OOM'd both ways while sharing with two FL cells. **On THIS box: run CA STL-reg with the GPU val
   path (`P1_S2_AUTO_BUDGET_GB=30`) but give it the GPU to itself or pair only with one light cell.**
4. Cherry-picked the A40's p1 fix (`61a7e0fd`) onto this branch (CPU val-metric option). MTL cells use S2
   (`MTL_CHUNK_VAL_METRIC=1`) and are GPU-friendly/host-light.

## RESUME (when the H100 studio is back)
Env persists (torch 2.11.0+cu128 + pt211 pyg). Re-export the board env, then **run cells SERIALLY**:
1. **FL MTL** (re-run; died fold1): `scripts/closing_data/h100_fl_cells.sh` already ran Cells 1–2 (results on
   disk) — just re-run Cell 3, or the standalone champion-G command in `logs/h100_fl_mtl_s0.log`'s header.
   Score with `h100_score_matched.py <rundir> --seed 0` → FL Δcat / Δreg vs the ceilings above.
2. **CA**: `scripts/closing_data/h100_ca_reg.sh` (Cell A GPU-val + Cell B MTL) — run when the GPU is free of
   other big cells. CA STL-reg val-cat is ~37 GB; keep it alone or with one light cell.
3. Commit MTL + matched-score JSONs (C28); update the FL table; flag PR #32 for audit.

---

# SESSION 2026-06-23 (PM) — precision gate, small-state cells, Istanbul Phase V, reg-prior leak fix

> Branch `study/board-h100`, 16 commits (this session). All seed-0×5f unless noted; engine `check2hgi_dk_ovl`
> (gated stride-1 overlap, frozen v14 substrate) for Gowalla board states; `check2hgi` for Istanbul.

## 1 · Precision gate (the fp16-collapse fix, CA_MTL_DIVERGENCE follow-through)
- **AL gate (bf16 vs fp32, fp32-scored):** reg bf16 69.6873 / fp32 **69.8067** (Δ −0.12); cat bf16 63.5810 /
  fp32 63.5591 (Δ +0.02). fp32 hits the §1 anchor exactly. **User decision: small/mid states → fp32**; FL runs
  its own fp32-vs-bf16 gate.
- **FL gate (running):** partial folds 1-2 — fp32 reg 77.54 / bf16 77.56; fp32 cat 79.68 / bf16 79.91. **Both
  precisions beat BOTH FL ceilings** (reg 77.5 > 76.71, cat 79.8 > 75.15) — the fp16-artifact reg-gap REVERSAL
  confirmed. **bf16 ≈ fp32 on quality AND speed** — these overlap runs are **data-loading/launch-bound** (GPU
  util ~8-25%), so bf16 buys ~0 wall-clock; precision only matters for CA (fp16 overflow). Full 5f table pending.

## 2 · Cells completed (fp32, gated overlap)
| state | MTL cat | STL cat ceiling | Δcat | MTL reg FULL@10 | STL reg ceiling | Δreg |
|---|---|---|---|---|---|---|
| **AL** | 63.5591 | 55.8704 | **+7.69 (beats)** | 69.8067 | 69.99 (fresh p1 = doc 69.98) | **−0.18 (matches)** |
| **AZ** | 63.3875 | 57.1305 | **+6.26 (beats)** | 59.3360 | 59.40 | **−0.06 (matches)** |
| **CA** | (MTL §4 pending) | **70.2573** (NEW, fills gap) | — | — | 63.4848 (reuse) | — |
| **FL** | gate running | 75.147 (reuse) | — | gate running | 76.7123 (reuse) | — |
Pattern (both small states): MTL **beats the cat ceiling, matches the reg ceiling** under correct precision.
AL reg ceiling rebuilt as a fresh 5f artifact (69.99 ≈ the documented 69.98 → scalar now backed on disk).

## 3 · Istanbul Phase V — champion-G transfers to a non-US corpus (frozen substrate, paper-grade)
Built frozen GCN-500ep substrate on the existing 520-mahalle graph (index-aligned, `phase_v_substrate.py`);
4 seeds (0/1/7/100). **cat 60.16±0.07 (beats STL cat 52.10 +8.06), reg 69.79±0.06 (matches STL reg 70.37,
−0.58), +17.27 over the Markov-1 floor 52.52.** Tiny cross-seed variance → highly reproducible. Region def =
mahalle (proper admin, PRIMARY); H3 (2585) staged as secondary. TX v14 substrate sha256-verified vs manifest
(no download needed). See `PHASE_V_ISTANBUL_S0.md`.

## 4 · ⚠ Reg-prior split-provenance leak (CONCERNS C29) — found, FIXED, **zero impact** (two advisors)
`compute_region_transition.py` (+ `compute_region_colocation.py`) built the per-fold log_T/log_C split on the
**canonical CHECK2HGI** engine while the board trains on **dk_ovl** (stride-1, MIN_SEQ-filtered, ~220 vs ~322
val-users/fold) → the prior's fold partition mismatched → ~80% of dk_ovl val-users leaked into the prior.
**INERT in every actual run** (verified bit-identical + by two independent advisors): the board reg head runs
**prior-OFF** (`freeze_alpha=True alpha_init=0.0`) and `--log-t-kd-weight 0.0`, so log_T is ×0; Istanbul's prior
IS active but built on its **matched** engine. **Fixes shipped:** both builders engine-aware (`--engine` threads
split+sequences+output, stamps `engine` provenance, with a substrate-engine probe-fallback); a trainer-side
guard in `mtl_cv.py` fails loud on a prior-engine mismatch **when the prior is active** (α-prior OR any
log_T/C-KD weight>0 — the v12 default `--log-t-kd-weight 0.2` makes the KD route the sneaky one). Double-gated →
never fires on the board (prior-OFF + legacy payloads).

## 5 · Operational learnings (carry forward)
- **Resume driver (`resume_fl_fold.sh`)** for the ~1-2h studio restarts: `--only-fold` is **0-INDEXED**
  (canonical fold F = `--only-fold F-1`, loads `fold{F}.pt`) AND re-indexes its single output CSV to `fold1` —
  copy `fold1_*`→`fold{F}_*` into the master. Isolated `RESULTS_ROOT` per arm (no rundir race). Idempotent.
- **Host-RAM headroom guard is real — never override it.** The overlap datasets (FL 1.27M / CA 2.92M / TX
  3.83M rows) need ~16-22GB host RAM each; `MTL_RAM_HEADROOM_GB=10` passed the preflight then **OOM-killed**
  (took the newest proc — protected FL). Big-dataset cat ceilings (CA/TX) can't co-exist with the FL gate
  through its fold transitions without RAM pressure; run them when the gate frees RAM, or alone.
- **bf16≈fp32 speed**: these runs are CPU/launch-bound, so `--compile`/`MTL_COMPILE_DYNAMIC=1` (distinct
  `TORCHINDUCTOR_CACHE_DIR` per co-scheduled proc) help the compute path that isn't the bottleneck.
- **The env is conda `cloudspace` (`python`), NOT `.venv`** (the handoff's `.venv/bin/python` is wrong here).
- **Board recipe needs `--canon none` + explicit `--no-{reg,cat}-class-weights`**: bare `train.py` auto-injects
  canon-v16 which trips the wrong-substrate guard under `MTL_STRICT=1` (the board runs champion-G on dk_ovl).

## Remaining
FL gate full-5f table; **CA MTL §4** (the heavy/restart-risky cell — run alone, must clear ep30 under bf16/fp32);
TX cat ceiling (running); A40 lane = TX MTL + TX cat (staged). Multi-seed {1,7,100} for Gowalla states is out of
scope (1-seed reduced board).
