# v18 — the v17 recipe on a leak-free substrate

> **Status: RUNNING** (started 2026-08-06, commit `f281a709`, host `nespedgpu`).
> Live state: [`status.json`](status.json) (machine-readable) · [`PROGRESS.md`](PROGRESS.md) (human).
> Charter: [`../V18_AGENT_PROMPT.md`](../V18_AGENT_PROMPT.md).

**v18 = the frozen v17 recipe, with the consecutive-visit leak fixed, plus elapsed-time node
features.** Not an architecture change. The check-in graph is **forward-only** in training and at
readout, and node features carry 4 elapsed-time columns (`in_channels = 15`).

## Read in this order

1. [`METHODOLOGY.md`](METHODOLOGY.md) — what v18 is, why forward-only, why elapsed time, what was
   excluded and on what evidence. **Start here.**
2. [`V18_RESULTS.md`](V18_RESULTS.md) — the tables. Per state: dedicated cat, dedicated reg, joint
   (diag-best **and** joint-best), Δ vs the v17 published values, current `n` stated in every table.
3. [`AUDIT.md`](AUDIT.md) — the §6 self-checks with their measured values, and anything that failed.
4. [`PROVENANCE.md`](PROVENANCE.md) — every rundir: state, seed, PID, path, recipe, commit SHA.
5. [`data/v18_results.json`](data/v18_results.json) + [`score_all.py`](score_all.py) — the
   machine-readable record and the reproducer that regenerates it from the rundirs. **Every number
   in every markdown table here is traceable to that JSON.**

## What to expect (so a real result is not mistaken for a bug)

- **Category falls a long way.** At Alabama the honest forward-only arm scores ~28.3 where v17
  reported ~56.9. The v17 number was inflated by the leak; the drop is the point of the exercise.
  A v18 category number landing **within 5 pp of its v17 value** means the forward-only path is
  broken — `run_wave.sh` raises a `[VERIFY]` flag automatically and it is investigated, not reported.
- **Region barely moves.** The region tower consumes region embeddings indexed by historical places,
  which are per-region dataset constants, so the leak cannot reach it. A move **> 2 pp** is likewise
  flagged and investigated.

## Scope guardrails

- **Do not** add place identity or region identity to the node features. Every variant was measured
  at Alabama and every one hurt (−0.41 to −1.86); only elapsed time gained (+0.83). See
  METHODOLOGY §3 for the mechanism and the one recorded unexplained ordering.
- **Do not** re-sweep encoder width or depth. 32/64/128 and 1/2/3/4 are settled at 64 and 2.
- **Do not** change the output width 64 — it is a hard constant in three places.
- **Do not** write into `output/check2hgi_dk_ovl/`, `output/check2hgi_design_k_resln_mae_l0_1/`, or
  any other existing engine directory. Those are the frozen v17 artifacts behind the published
  table. `build_study_repr.py` refuses to; do not work around it.
- **Do not** report a v18 number as beating or matching anything without the test §10 requires:
  paired superiority for "outperforms", TOST non-inferiority for "matches".

## Layout

```
run_phase0_build.sh    build + readout + materialize the v18 engine, per state (resumable)
run_wave.sh <SEED>     one wave: 6 states x 3 families at one seed (resumable, idempotent)
status_update.py       rewrites status.json + PROGRESS.md from the per-cell sidecars
smoke_alabama.sh       1-fold/2-epoch validation of the full training path
logs/                  driver + per-cell logs
```

Per-cell score sidecars live at `docs/results/closing_data/v18/<state>_s<seed>_<family>.json` and
are the resumability source of truth: a cell is done iff its sidecar exists.

## Engine

`EmbeddingEngine.CHECK2HGI_V18 = "check2hgi_v18"`, registered in **three** places in
`src/configs/paths.py` (the enum, `MTL_CHECK2HGI_ALLOWED_ENGINES`, and the separate `supported`
tuple inside `IoPaths.get_next_region` — a category-only registration trains fine and then fails at
the region tower). One engine per state at `output/check2hgi_v18/<state>/`, shared by all four
seeds; the representation does not depend on the downstream seed.
