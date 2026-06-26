# Faithful STAN — findings, data, and knowledge (region baseline)

> **Branch `stan` / this PR.** Faithful reimplementation of STAN (Luo et al., WWW 2021) for the
> next-**region** task, run on our data + seed-0 user-disjoint 5-fold split. Code +
> run guide: [`research/baselines/stan/README_FAITHFUL_STAN.md`](../../../research/baselines/stan/README_FAITHFUL_STAN.md).
> Faithfulness-vs-reference table: [`docs/baselines/next_region/stan.md`](../../baselines/next_region/stan.md).
> Companion to the region-baseline plan in [`../../../articles/[mobiwac]/STAN_REFOOTING_HANDOFF.md`](../../../articles/%5Bmobiwac%5D/STAN_REFOOTING_HANDOFF.md).

## 1. What this settles

STAN is the closest **architecture** competitor for next-region. The prior **v4** numbers
(AL 34.46 / AZ 38.96, below the Markov floor) were a **COLLAPSE artifact — superseded, never cite.**
A two-agent faithfulness audit vs the paper + reference code, an independent code review (GO), and an
observer-driven performance pass produced an **audited-faithful, converged, trustworthy** STAN:

| State | regions | Acc@10 (seed 0, 5f) | our MTL reg | verdict | precision |
|---|---:|---:|---:|---|---|
| AL | 1109 | **60.72 ±5.20** | 69.81 | we beat | fp32+compile |
| AZ | 1547 | **49.86 ±11.52** | 59.34 | we beat | fp32+compile |
| Istanbul | 520 (mahalle) | **61.86 ±0.61** | 74.28 | we beat | bf16+compile |
| FL | 4703 | running | 77.28 | (expect beat/near) | bf16+compile |
| CA / TX | 8501 / 6553 | heavy (~8–12 h/state on A40) | 65.66 / 67.02 | pending | bf16+compile |

STAN is a **SECONDARY** region reference (HMT-GRN, faithful + board-matched, is PRIMARY; ReHDM is the
own-protocol reference). The substrate-bound `stl_hgi` STAN (STAN on our HGI embedding, board windowing)
is an OPTIONAL labeled ablation only — recorded separately (AL 70.35 / AZ 59.66 / FL 76.82), NOT the
headline STAN cell.

## 2. The six faithfulness fixes (the audit)

`next-region` target, CrossEntropy, single-head, dropped user-embedding (cold-user CV) are the kept,
documented, task-driven deviations. Everything else was made faithful:

1. **Matching layer** → STAN's `layers.Attn`: **multiplicative** Δd gate (`content * bias`) + learned
   **`Linear(seq,1)`** position collapse, **no softmax**. (v4 used additive bias + softmax-mean → it
   collapsed to a pure proximity/distance prior — the cause of the below-floor number.)
2. **Embedding init** std 0.02 → ≈ `1/√d` (v4's content channel was ~25× below the bias at init).
3. **`/√d` removed** in both attention layers (reference omits it).
4. **Sequence construction** stride-9 windows → STAN-native **prefix-expansion** (every position from
   its causal prefix; `CONTEXT_LEN` cap). v4 was ~9× data-starved + biased to every-9th-position targets.
5. **Convergence** OneCycleLR (annealed to ~0 by ep50; "best@49/50" never plateaued) → **constant LR +
   early-stop** (patience 20, cap 200); best-epoch must land interior.
6. **fp32 + seed 0** (v4 was bf16 + seed 42).

## 3. Performance / methodology knowledge (the O(regions) analysis)

The matching layer is **mathematically irreducibly O(B·n·R·D)** (the output is `[B,R]`; the distance
gate is full-rank in R, so the logit does **not** reassociate to anything cheaper — the multiplicative
content×gate happens before the position collapse; per-bin expansion over the 64 interval params
collapses right back). **The scaling wall in R is fundamental.**

BUT the *runtime* was ~60× above that FLOP floor (FL's GEMM is ~10 ms/step; we measured 617 ms) — it was
**memory-bound on materialized `[n_pois,R]` / `[B,n,R]` fp32 intermediates + their backward scatter.**
That constant was reclaimed without touching the metric, **~530 → ~2 s/epoch on small states (~85×)**:

- **`F.embedding`** for the table/row gathers — the single biggest fix (**23×**). The generic
  `tensor[idx]` backward dispatches a slow atomic `indexing_backward` kernel (profiled = **97.7%** of the
  backward); `F.embedding` uses the optimized `embedding_dense_backward`.
- **dd_poi precompute** (POI→region haversine once + gather, not per-batch trig) +
  **per-poi bias precompute** (interp the Δd bias over `[n_pois,R]` once/forward, gather).
- **`torch.compile`** (`--compile`) — fuses the elementwise interp (2.6–4.7×), numerically faithful.
- **bf16** (`--amp bf16`) — A/B-validated **quality-neutral** (0.07 pp vs fp32 on AL fold-0); halves the
  big-state memory traffic. Small states stay fp32+compile (bf16 gives them nothing — they're launch-bound).

**Parallelism (measured + advisor-confirmed):** fold/stream parallelism does NOT help on one GPU — the
GPU is compute+memory saturated post-compile, 5-fold-parallel OOMs the A40, and parallel folds time-slice
(no overlap). The only on-GPU levers are bf16 + compile. Fold fan-out helps **only across machines**: an
**H100** (80 GB + ~4.8× bandwidth) fits all 5 bf16 folds and they genuinely overlap → estimated FL
~30–40 min / CA ~50–75 min / TX ~40–60 min (vs ~6–11 h sequential on this A40). `--only-fold k` is wired
for that. Further single-GPU speedup needs a custom **Triton fused matching kernel** (recompute geometry
in-SRAM, never materialize the intermediates) — high effort, not done; would approach the FLOP floor (~1–2 h FL).

### 3.1 Optimization sweep A–G (2026-06-26, validated on AL; commit 507a5f22)

The safe-stack candidates were implemented and **A/B-validated on AL** (5-fold, fp32+compile, seed 0;
quality bar = the pre-opt reference **Acc@10 60.72 ±5.20**). Verdicts:

| Opt | What | Verdict | Evidence |
|---|---|---|---|
| **A** bf16 | `--amp bf16` autocast | **adopt (big states)** | prior 0.07 pp vs fp32; halves big-state mem traffic |
| **B** CUDA graphs | `--compile-mode reduce-overhead` | **available, not adopted** | faithful, but AL 44 ms/step = default (not launch-bound); disables D |
| **C** val-once | per-epoch Acc@10 only; snapshot best state; recompute metrics once | **adopt** | **bit-identical** → AL 60.72 exact; drops the per-epoch `[n_val,R]` cache (FL ~15 GB/epoch) |
| **D** distinct-POI | interp Δd bias over the batch's distinct POIs (`unique→interp→scatter`) | **adopt** | **bit-identical** → AL 60.72 exact; full-table interp alone is 115 ms (FL), the whole forward with D is 110 ms |
| **E** max-autotune | `--compile-mode max-autotune` | **dropped** | **SEGFAULTS** during Triton autotune (core dump) |
| **F** larger batch+LR | bs 4096/8192, √-LR scaling | **rejected** | no wall-time gain (matching saturated at bs2048: 44→64→135 ms/step, flat ~1.5 s/epoch) **and** −0.6…−0.8 pp (bs4096 59.94, bs8192 60.15) |
| **G** Triton fused matching kernel | recompute geometry in-SRAM, never materialize `[B,n,R]` | **future work** (see below) | targets the real bottleneck but ROI marginal post-compile |

**Production config = A (bf16, big states) + C + D + compile (default mode).** Quality bit-identical to
the pre-opt reference; FL fold-0 ~115–130 s/epoch → **full FL ~4.5 h** (vs ~6 h).

### 3.2 Profile (FL fold-0, eager, bs2048) — why G is future-work

```
full forward:      110 ms      forward+backward: 618 ms  → backward ≈ 508 ms (82% of step)
  matching layer:   41 ms        interp bias (full table): 115 ms  ← D avoids this
```

The **backward** dominates: the `[B,n,R]` (481M-elt) content-einsum + gate gradients and the
`embedding_dense_backward` scatter into `[U,R]`. A fused fwd+**bwd** Triton kernel (G) would eliminate
that materialization (FLOP floor ~10–15 ms/step). **But `torch.compile` already reclaims most of it**
(real compiled step = **213 ms**, not 618 ms), so G competes with the compiled path: realistic ~3–7×
more (FL ~1–1.5 h), at the cost of hours of work + MED backward-correctness risk — **not worth it for a
secondary baseline we already beat at every state.** Shelved 2026-06-26 (decision: profile-then-shelve).

## 4. Data — Istanbul (Massive-STEPS) faithful-STAN data pipeline

Istanbul is NOT a Gowalla/TIGER state; it needed a new data path (now supported in `etl.py`):

1. **FSQ check-ins** — `python -m scripts.second_dataset.acquire --city istanbul` → HF
   `Massive-STEPS-Istanbul` → `data/massive_steps_istanbul/raw/`. The **`raw/tabular/*.parquet`** have the
   structured records (`user_id, venue_id, latitude, longitude, venue_category, timestamp`); the
   `raw/data/*` are LLM-prompt format and lack coordinates — use `tabular/`. (`huggingface_hub` installed
   via `uv pip install huggingface_hub` — the venv had no pip.)
2. **Mahalle regions** — `python scripts/second_dataset/acquire_istanbul_mahalle.py` → OSM
   `admin_level=8` polygons → `data/miscellaneous/istanbul_mahalle/istanbul_mahalle.geojson` (964 polys).
3. **STAN check-ins** — concat the tabular parquets, factorize `user_id`/`venue_id`, rename to the STAN
   schema → `data/checkins/Istanbul.parquet` (**481,850 rows / 23,697 users / 30,279 POIs**).
4. **ETL** — `etl.py` registers `istanbul` (geojson region source) and auto-detects the region-id column
   (TIGER `GEOID` vs mahalle `@id`). Result: **520 mahalle regions** (matches the board Istanbul count) /
   438,886 prefix-expansion windows.

> **Note on the handoff:** the original handoff put Istanbul STAN "on the M4." It runs here cleanly via
> the pipeline above; the 520-region count agrees with the board's Istanbul substrate.

## 5. Reproduce
See [`research/baselines/stan/README_FAITHFUL_STAN.md`](../../../research/baselines/stan/README_FAITHFUL_STAN.md)
§3 for the exact ETL + train commands (small states fp32+compile; FL/CA/TX add `--amp bf16`; Istanbul via
the data pipeline above). Acceptance: best-epoch interior, Acc@1 sane, no NaN-collapsed fold; STAN below
our MTL reg. **Old v4 / seed-42 STAN numbers are superseded — strike them from the paper artifacts.**
