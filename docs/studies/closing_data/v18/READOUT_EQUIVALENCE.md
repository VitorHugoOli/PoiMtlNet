# The per-window readout is an identity on a forward-only graph

> **Finding, 2026-08-06.** For a **forward-only** check-in encoder, the per-window
> `prefix_forward_only` readout reproduces the one-shot full-graph export to float32 round-off. The
> per-window rebuild — one forward pass per window, `O(n²)` in a user's history — is therefore
> redundant for v18, and the v18 engine can be materialized from `embeddings_insample.parquet` at
> `O(1)` forward passes, exactly as the v17 substrate is.
>
> **This does NOT generalize to bidirectional arms.** There the rebuild does real work. The guard is
> enforced in code, not by convention: `materialize_from_insample.py` hard-fails unless
> `build.json` reports `causal_graph.forward_only == true`.

## Why the cost appeared at all

The v17 substrate and the v18 substrate are produced by different instruments:

| path | forward passes | device | Florida cost |
|---|---:|---|---:|
| `checkin_emb` export (v17 substrate, and the builder's `embeddings_insample.parquet`) | **1**, whole graph | GPU | **133 s** |
| `infer_checkins.py --readout prefix_forward_only` | **one per window** (1,274,418) | **CPU-only** | ~3–5 h |

`infer_checkins.py` documents this itself (line 42): *"`prefix` is one forward pass per window and
`streaming` one per visit, so both are `O(n²)` in a user's history length; **no saving is claimed**.
On the two study datasets this is affordable because the per-user paths are short relative to the
graph."* The two study datasets were Alabama and Florida; California and Texas were never exercised.
The same file's `--max-users` help anticipates the wall: *"a large state needs a declared random
subsample."*

The script has **no CUDA support anywhere** — no `device` argument, no `.to()`. That is a reasonable
design for an instrument doing millions of tiny (≤ 9-node subgraph) passes, where kernel-launch
overhead would dominate; a naive GPU port would likely be slower, not faster. So the fix is not to
move the readout to the GPU — it is to notice the readout is unnecessary here.

## Why the identity holds

The per-window rebuild exists because on a **bidirectional** graph a visit's vector is genuinely
window-dependent: node `v` is convolved over `v+1`, so cutting the path at the window's last
observed visit changes `v`. That is precisely the leak the strict readout is built to remove.

**v18 trains forward-only.** Edges survive only where `src < tgt`, so messages flow strictly
past → future and a visit's representation is a function of its own prefix alone. Truncating the
graph at the window's end removes only nodes that never sent it a message. The rebuild recomputes,
at `O(n_windows)` cost, exactly what the single full-graph pass already produced.

The one place this could have failed is **degree normalization**: truncation removes the outgoing
edge of the window's last node (slot 8), which would change its degree and hence its normalization.
The measurement rules that out — slot 8 is no worse than slot 0.

## Evidence

`materialize_from_insample.py --validate-against-npz` compares the assembled `[n_windows, 9, 64]`
tensor against the real per-window npz over **every window**, not a sample. Embeddings have scale
‖emb‖ ≈ 1.05 mean.

| state | windows compared | max abs diff | slot 8 max | mean abs diff |
|---|---:|---:|---:|---:|
| alabama | 96,326 | 2.384e-06 | 2.384e-06 | 1.407e-07 |
| arizona | 200,895 | 2.861e-06 | 2.861e-06 | 1.463e-07 |
| istanbul | 271,666 | 3.099e-06 | 2.861e-06 | 1.474e-07 |
| florida | 1,274,418 | *pending — the readout was left running as a large-scale check* | | |

Mean residual is float32 epsilon; the tolerance gate is `1e-4`, ~30× above the worst observed value.
The slow growth with dataset size (2.4 → 3.1e-06) is accumulated round-off from a different op
ordering, not a semantic drift.

An earlier, independent embedding-level check (before the materializer existed) compared npz slots
against the per-visit export directly and found the same magnitudes, with no slot-8 outlier.

## What this changes

- **Phase 0 for FL/CA/TX**: from ~17 h of single-threaded CPU readout to seconds of indexing.
  Florida's engine materialized in **19 s** for 1,274,418 windows.
- **Nothing about the science.** The vectors are the same to float32 precision, so every downstream
  number is unchanged. v18 remains the v17 recipe on a forward-only graph with elapsed-time features.

## Reproducing

```bash
# fast path (forward-only arms only; hard-fails otherwise)
python scripts/integrity_v2/materialize_from_insample.py \
  --state <state> --study-run results/check2hgi_v18/<state>/V18 \
  --dest-engine check2hgi_v18 \
  [--validate-against-npz results/check2hgi_v18/<state>/V18/win_matched.npz]

# reference path (required for any bidirectional arm)
python scripts/integrity_v2/infer_checkins.py --state <state> --checkpoint <ckpt> \
  --readout prefix_forward_only --out <npz> --self-test
python scripts/integrity_v2/materialize_engine.py --state <state> --arm-npz <npz> \
  --dest-engine <engine>
```

`docs/studies/closing_data/v18/run_phase0_fast.sh` runs the fast path end to end and passes
`--validate-against-npz` automatically whenever an npz happens to exist for that state.
