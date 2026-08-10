# v18_2 execution plan — CA/TX on Modal, small states on the A40

> Written 2026-08-09 19:35. Numbers are MEASURED this session unless marked estimate.
> Under adversarial review before launch. Nothing runs until the user approves.

## 1 · Hardware, measured on one identical cell

The same alabama dedicated-cat cell (5 folds, fp32, identical data) was run on all three cards:

| GPU | macro-F1 | wall | speedup | $/h | **$ per A40-hour** |
|---|---:|---:|---:|---:|---:|
| A40 (local) | 30.7654 | 322 s | 1.00x | free | free |
| **A100-40GB** | 30.6790 | 133 s | **2.42x** | 2.10 | **$0.87** |
| H100 | 30.7049 | 96 s | 3.35x | 3.95 | $1.18 |

Fold SD is 1.0391, so the largest cross-hardware deviation is **0.083 SD**. Rented cells are
poolable with local ones. H100 is faster but 26 % dearer per unit of work: it needed 4.55x to
break even against the A100 and delivered 3.35x. **A100-40GB is the default; H100 is reserved for
the one latency-critical cell.**

## 2 · Lanes

| lane | account | dataset | staged | engine |
|---|---|---|---|---|
| 1 | `vho2009` (primary) | california | yes | **materialized, 7.2 GB** |
| 2 | `vitor-h-oliveira` (MODAL_2) | texas | yes | pending (~96 s) |

Staging is solved: `modal volume put` direct from the GPU host runs at ~50 MB/s (versus 0.44 MiB/s
through the confined path) and both datasets uploaded in **41 s, in parallel**. Volumes persist, so
every later job submits with `inputs=[]` in ~6 s and pays no transfer.

Total data moved was ~1.9 GB rather than 34 GB, via three compounding reductions: ship the seed
(`embeddings_insample.parquet`) instead of the engine; recompress snappy to zstd (CA 916 -> 739 MiB,
verified lossless with `table.equals()`); and extract only the two columns materialization reads
from `next.parquet` (8.8 GiB -> 1.2 MiB). The 7.2 GB engine is then rebuilt on Modal in 96 s.

## 3 · Phases

**A — california seed-0 joint (urgent).** H100, one job, lane 1.
`22550 s / 3.35 = 1.87 h`, cost `1.87 x 3.95 = $7.39`. ETA ~21:22 against the A40's 01:06.
The A40 run is killed **only after** the H100 job confirms it is training.

**B — small states, free.** Once CA joint leaves the A40, that card runs IST/AL/AZ/FL for seeds
1/7/100: `14.3 A40-h` serial, ~8 h at 2-wide. Costs nothing.

**C — CA and TX seeds 1/7/100.** A100-40GB, **3 concurrent jobs per lane**, one job per cell.

| lane | cells | container-h | cost | makespan (3-wide) | longest cell |
|---|---:|---:|---:|---:|---:|
| 1 california | 9 | 11.5 | $24.14 | **3.83 h** | 2.59 h |
| 2 texas | 9 | 11.6 | $24.41 | **3.87 h** | 2.57 h |

Makespan is a bin-packing result, not `total/3`. Here the two coincide because each of the three
bins receives exactly one joint, one cat and one reg cell — verified by LPT packing, all three bins
land at 3.83 h. A 9-wide fan-out would finish in 2.59 h for the same money, since Modal bills per
container-second; 3-wide is the user's deliberate choice of safety over speed.

## 4 · Why one job per cell

Job granularity is the checkpoint. The runs use `--no-checkpoints` (protocol), so resilience comes
from decomposition instead:

| decomposition | lost if a job dies |
|---|---:|
| one job per cell | 2.59 h = **$5.44** |
| one monolithic 3-seed job | 11.5 h = $24.14 |

No storage cost, no protocol deviation, and a dead cell is simply re-run. What the merge needs is
tiny: **475 bytes of score JSON** per cell out of a 5.6 MB rundir.

## 5 · Budget

| | |
|---|---:|
| already spent (lane 1) | $1.60 |
| phase A, H100 | $7.39 |
| phase C, both lanes | $48.55 |
| **total** | **$57.54** |
| free credit, 2 accounts | $60.00 |
| headroom | **$2.46 (4 %)** |

Four percent is too thin. Two mitigations: the user is **adding funds to `vho2009`**, and R$100
(~$19 = 9 extra A100-hours = 22 A40-hours of rescue capacity) is held in reserve. Launch proceeds in
waves, with actual spend read between waves rather than trusted from these estimates.

## 6 · Precision

fp32 everywhere (`MTL_DISABLE_AMP=1`). bf16 is forbidden at CA/TX scale on Ampere — a documented
backward-pass NaN at 8462 and 6530 region classes, where the collapse is silent because the
per-task best-epoch selector reports the pre-collapse peak. It is known-safe on Hopper, but these
cells pool with A40 fp32 numbers, so precision must not vary inside a pooled cell. TF32 is already
enabled on every command (`--tf32`) and is not a change.

## 7 · Open risks

- **Concurrency grant unverified.** Nothing confirms Modal permits 3 concurrent A100s per account
  on this tier. If it queues at 1, lane makespan becomes 11.5 h and phase C would finish ~08:50 —
  still inside the deadline, but the margin shrinks a lot.
- **Actual credit balance unknown.** The $1.60 is my own arithmetic from job walls, not Modal's
  ledger; the SDK exposes no balance query. The dashboard figure should replace it.
- **Texas engine not yet materialized** (~96 s, CPU tier).
- **Region results land outside the rundir**, at `docs/results/P1/region_head_<state>_...json`. A
  rented reg cell must return that path too or the merge silently drops it.
- **Wall estimates are A40 timings scaled by a single measured ratio.** They are good, not
  guaranteed; a cell that overruns eats headroom directly.

---

# CORRECTION 2026-08-10: the speedup figures above are WRONG

Everything in this document that rests on a 2.42x A100 speedup is invalid. The benchmark that
produced it omitted `--compile`, which `v18/run_wave.sh` passes on all three cells.

| | wall | compiled |
|---|---:|---|
| A40 reference | 322 s | **yes** |
| A100 gate | 133 s | **no** |
| H100 gate | 96 s | **no** |

So "2.42x" measured a compiled A40 against an uncompiled A100. The A100 was handicapped and still
won, which means the true hardware advantage is **larger** than what was recorded. Published
figures put H100 at roughly 2.4x A100 for training and A100 at roughly 2.5x A40; the numbers
above sat far below both, and that gap was the signal that something was wrong in software.

The same defect explains the California joint failure: the A40 finished in 4.85 h **with**
compile, the A100 failed to finish in 3.6 h **without** it, and the job was killed by its run
clock having written nothing to `./out/`. Cost: $9.87 for no data.

Two conclusions drawn from the broken benchmark are retracted:

1. that rented GPUs do not help — they do, and by more than was measured;
2. that the A40 should keep the large states on cost grounds — that rested on $/A40-hour ratios
   computed from the same bad number.

`scripts/run_lane.sh` has since been made flag-equivalent to the record (`--compile`,
`--embedding-dim 64`, the global `PYTHONPATH` and `PYTORCH_CUDA_ALLOC_CONF` exports, and a
persistent `TORCHINDUCTOR_CACHE_DIR`), verified by a mechanical per-cell diff.
`MTL_RAM_HEADROOM_GB` is deliberately not set: it guards a RAM-constrained A40 box and has
nothing to protect on an H100 container.

**No timing in this document should be reused. Re-measure with the corrected script before
planning any spend.** See `MODAL_MANUAL.md`.

---

# CORRECTION 2026-08-10 (second): every cost in this document is ~42 % low

Separately from the timing defect above, the **rates** are wrong too. §5's budget priced GPU
hours at the bare GPU list price. Modal bills **GPU + CPU + memory** per container-second, so an
A100-40GB at 8 CPU / 64 GB is **$2.988/h**, not $2.10 — and not the $2.73 that `MODAL_MANUAL.md`
§6 used to give either. The full price table, fetched 2026-08-10, is now in `MODAL_MANUAL.md` §6,
and `scripts/modal_lane.py` computes the rate before it submits so a plan is never costed from
memory again.

| | as planned | at the real rate |
|---|---:|---:|
| phase C, both lanes | $48.55 | **$68.72** |
| phase A, H100 1.87 h | $7.39 | $9.05 |
| total | $57.54 | **$79.37** |
| against $60 of credit | +$2.46 | **−$19.4** |

§7's other two open risks are **closed**, both in the plan's favour:

- **Concurrency was never a constraint.** The Starter plan allows 100 containers + **10 GPU
  concurrency**. 3-wide was never going to queue, and 9-wide would finish phase C in 2.59 h for
  identical money.
- **The balance is queryable.** `modal billing summary` / `modal billing report --show-resources`
  exist in client 1.5.3. Measured on lane 2 (`vitor-h-oliveira`) 2026-08-10: **$0.00 metered,
  $0.00 billed** — that account has never run a billable job, so its $30 monthly credit is
  intact. Free credits are **$30/month per workspace and they recur**; the plan treated
  $30 × 2 as a one-time $60.

One structural correction to §2's lane table: **lane 2 held texas only, and its seed was never
materialized into an engine** — a preflight on 2026-08-10 found `output/check2hgi_v18/texas/`
absent. Alabama was staged onto lane 2 the same day (627 MB, direct from nespedgpu, verified by
byte count) and ran there. Check `modal volume ls poimtl-v18-data /seed` and run
`modal_lane.py --preflight-only` before assuming a lane can run a state.
