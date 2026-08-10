# v18_2 — the rented-GPU lane

What lives here, and what to read first.

| file | purpose |
|---|---|
| **`MODAL_MANUAL.md`** | **read this before submitting anything to Modal.** Start at §0.1 — it tells you which entry point your harness can actually use. Upload, harvest, preflight, cost, and the failure each rule came from. |
| `scripts/run_lane.sh` | the ONLY sanctioned way to run a cell. Flag-equivalent to `v18/run_wave.sh` (all three cells re-diffed 2026-08-10, post-audit). |
| `scripts/run_wave_a40.sh` | seed-major driver for the four small states on the local A40. |
| `scripts/preflight.py` | authoritative per-cell input list. Run it before every launch. |
| **`scripts/modal_lane.py`** | **Claude Code / plain-shell entry point.** Sandbox-based: live stdout, harvest onto the Volume, local archive, verified teardown, cost printed before submit. Works on either account. |
| `scripts/run_modal_cell.py` | Claude Science entry point — needs the `host.compute` API. Archives locally, then sweeps sandboxes. |
| `scripts/watch_modal.py` | watch a running lane from outside the container (reads the committed heartbeat). |
| `scripts/monitor.sh` | local-A40 monitor: duplicate drivers, write progress, GPU throttle, disk/RAM. |

**Two harnesses, two entry points.** `run_modal_cell.py` is written against the Claude Science
`host.compute` API. A Claude Code agent does not have it and should use `modal_lane.py`, which
needs only `pip install modal` and a token. `MODAL_MANUAL.md` §0.1 has the decision table.
| `EXECUTION_PLAN.md` | the two-lane plan. **Carries a correction: its speedup numbers are invalid.** |
| `METHODOLOGY.md` | what v18 is and why each piece is in it. |
| `FINAL_SETTINGS.md` | the approved hyperparameter sheet (author-approved 2026-08-09). |
| `AUDIT.md` | charter self-checks with measured values. |
| `PRECISION_CAVEAT.md` | why a bare `export` of a precision variable corrupts later cells. |

Deleted 2026-08-10 as superseded: `data/` (a stale sidecar copy that misled a review into
reporting three missing seed-0 cells when only one was missing), `PLAN.md` and `STAGING.md`
(both folded into `EXECUTION_PLAN.md` and `MODAL_MANUAL.md`).

## Instrumentation

A lane reports on itself. With `HARVEST=1` it writes `out/heartbeat.jsonl` (one sample per minute:
GPU, throttle, live cells, `folds_done`) and copies the tail of any failed cell's log into
`out/logs/`. Both land in the harvested directory, so they come back even when a job is killed by
its run clock. Prefer this to an external monitor, which cannot see inside a running sandbox.

## The state of the numbers

`v18_2` inherits the `v18` protocol unchanged: fp32, `--compile`, `--tf32`, 5 folds, 50 epochs,
`logit-adjust-tau=0.5` on the category heads only. Cells produced here pool with `v18` cells
**only** if that protocol is identical, which is why `run_lane.sh` is diffed against the record
rather than maintained by hand.

The rented-hardware timings recorded on 2026-08-09 are void; they compared a compiled A40 against
uncompiled rented GPUs. Re-measure before planning spend.
