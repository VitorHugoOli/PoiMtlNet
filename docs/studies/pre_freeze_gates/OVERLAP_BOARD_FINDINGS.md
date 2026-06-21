# Gated-overlap board prep — consolidated findings (2026-06-20/21)

> Single landing page for the overlap/gate/perf work done while preparing the gated stride-1 board.
> Detailed docs: `WINDOWING_AUDIT.md` (gate), `OOM_MEMORY_FIX.md` (memory/Q1), `SPEED_LEVERS.md` (compile/Q3),
> `DEFAULTS_AND_GUARDS.md` (board values + guards). All code is on PR #29 (byte-identical + perf).

## The board windowing (now the enforced default)
**stride-1 OVERLAP, GATED (`emit_tail=False`), MIN_SEQ=10.** Built via
`build_overlap_probe_engine.py <state> 1` (auto-gates at stride==1, defaults min_seq=10). The non-overlap
frozen v11/v14 substrate stays at the global `core.py` MIN_SEQ=5 (§0.1 reproduction — never flip globally).
A train-time guard (`folds._warn_if_ungated_overlap`) WARNs / `MTL_STRICT=1` fails if an overlap engine is
ungated or min_seq≠10 — so a stale ungated build can't silently train (it bit us once: AL left ungated →
phantom −2.5pp, see `ref_ovl_emit_tail_stale`).

## M1 — the tail-gate (why gated)
At stride-1, ~window_size OOB tail windows/user all target the user's LAST POI on near-all-padding
histories — a label-distribution skew (NOT a leak; orthogonal to `STRIDE1_LEAK_REAUDIT`). The gate drops them.
Footprint **shrinks with state size**: AL 10.9% of windows / 15.1% last-POI skew → TX 5.2% / 7.5%. So the
gate helps most at small states and is ~neutral at huge (can't hurt). StratifiedGroupKFold does NOT mitigate
the skew (it balances category across folds, not the pooled target distribution).

## Q1 — CA/TX dataset-on-GPU (CLOSED)
CA/TX overlap MTL is viable on the A40 via the **default auto-fit** (large-state dataset kept CPU-resident,
TX GPU ~6 GB, ~160 s/epoch). **Never set `MTL_DATASET_GPU=1` for CA/TX** — it forces ~31 GB of redundant
per-fold copies onto the GPU → OOM (no CPU fallback). The host-RAM OOM (eager 5-fold + dead `FoldData.x`,
~126 GB) was fixed by lazy per-fold construction + dropping `FoldData.x` (byte-identical). Full TX MTL ≈ 11 h
(intrinsic to 8.5× overlap). See `OOM_MEMORY_FIX.md`.

## Q2 — does the champion thesis hold under the gated board windowing?
**FL board-grade (gated, min_seq=10, seed-42, 5-fold, matched-metric):**

| task | champion-G MTL | STL ceiling | Δ |
|---|---|---|---|
| cat macro-F1 | 78.32 ± 0.93 | 75.20 ± 0.76 | **+3.12** ✅ beats (historical +2.6…+4.1) |
| reg top10 (full) | 75.52 ± 1.07 | 76.64 | **−1.12** ⚠ |

- **cat clearly holds** (+3.12, in range). Absolutes are higher than non-overlap (overlap lifts both heads).
- **reg is a FLAG:** historical (non-overlap, multi-seed) reg Δ was −0.35 ("matches"); under gated overlap at
  **seed-42** it's −1.12 (~0.8 pp more negative). Both sides ran at seed-42 (delta robust to seed bias), but
  it's single-seed/single-state. **Open: does the gate/overlap genuinely cost ~0.8 pp of reg-match, or is
  seed-42 unlucky? → needs multi-seed {0,1,7,100} fold-paired (board T3).** A seed-0 confirm is in progress.
- Wall times (uncompiled): STL-cat 23m50, STL-reg 39m41, MTL 3h11m (≈38 m/fold), total 4h15.

## Q3 — compile/tf32 (CLOSED)
- **Quality: neutral** (STL −0.05/+0.02; AL flat; SPEED_LEVERS A/B +0.05 pp). compile/tf32 = fp-ordering noise.
- **The "32-min compiled warmup" was a MISDIAGNOSIS** — it was torch.compile exceeding
  `cache_size_limit=8` → silent **eager fallback** (un-compiled, slow). Fixed by three levers:
  1. `MTL_COMPILE_DYNAMIC=1` (one symbolic-shape graph), 2. shared persistent `TORCHINDUCTOR_CACHE_DIR`,
  3. `cache_size_limit` 8→64 (default-on when compiling).
- **Measured (FL MTL fold-1, 8ep):** uncompiled 366 s → compile+dynamic fresh-cache 372 s (13 recompiles,
  break-even) → **cache-reuse 318 s, 0 recompiles (~13% faster, 0 warmup)**. The `requires_grad` (train↔eval)
  recompile is inherent but one-time (13 → 0 on reuse).
- **Board compiled recipe:** every compiled cell runs `--compile --tf32 MTL_COMPILE_DYNAMIC=1` + ONE shared
  `TORCHINDUCTOR_CACHE_DIR`. First cell ≈ break-even; all later cells ~13–15% faster, quality-neutral.
  Apply UNIFORMLY (p1 STL-reg ceiling supports `--compile`/`--tf32` so the comparison stays fair — never mix).

## Open
1. **reg −1.12 multi-seed check** — FL seed-0 (compiled) in progress; full board T3 settles it.
2. **PR #29** — all byte-identical + perf code (lazy-fold, S2-auto, M1 gate, guards, min_seq default, compile
   levers). Awaiting merge.
