# Handoff — what a Claude Code session changed on 2026-08-10, and what it means for Claude Science

Written for the other agent working this study. Read the three **coordination items** first; the
rest is detail you can take at your own pace.

---

## Coordination items — these need your attention

### 1. Lane 1's scripts bundle is STALE. Your runs still use the old `run_lane.sh`.

The fixes below are in the repo **and** in `/scripts/v18_2_scripts.tgz` on **lane 2's** volume
(`vitor-h-oliveira`). Lane 1 (`vho2009`) has its own volume with its own bundle, and I did not
touch it. Until you re-stage, `run_modal_cell.py` keeps shipping the previous code — including
the `harvest_watch` hang (item A below), which is live on exactly your path.

Re-stage from a session that holds the lane-1 token:

```bash
tar czf /tmp/v18_2_scripts.tgz -C docs/studies/closing_data/v18_2 scripts
modal volume put poimtl-v18-data /tmp/v18_2_scripts.tgz /scripts/v18_2_scripts.tgz --force
modal volume ls poimtl-v18-data /scripts --json      # verify the byte count, per §1
```

### 2. Nothing I changed alters the recipe — verified, not assumed.

The per-cell flag diff against `v18/run_wave.sh` was re-run after every edit:

| cell | verdict |
|---|---|
| REG | identical |
| JOINT | identical |
| CAT | identical **plus `--task-a-input-type checkin`** — the argparse default (`train.py:541`), inert, and already present before I touched anything |

`ENVCOMMON` is unchanged. No learning rate, batch size, head, τ, weight, scheduler, precision or
selector moved. Cells produced before and after these edits are protocol-comparable.

### 3. The one change with numerical exposure is OFF by default.

Giving the cat/reg cells a persistent `TORCHINDUCTOR_CACHE_DIR` would save the `torch.compile`
warm-up on every rented cell — and would also change how cells are produced relative to ones you
have already banked with a cold cache. Compiled numbers are within-fold-σ rather than
bit-reproducible (`CLAUDE.md`), so it is not a protocol change, but it is not mine to make for
your lane. It is gated behind **`INDUCTOR_SHARE_CELLS=1`, default off**. Unset, cat/reg behave
exactly as before. The joint cell is unaffected either way — the record pins its cache dir.

---

## What was fixed in `run_lane.sh`

All six were found by auditing against `v18/run_wave.sh`; each is a divergence a flag diff cannot
see. **A is the one that would have cost you money.**

| | defect | consequence |
|---|---|---|
| **A** | `HW=$(harvest_watch ...)` **hung forever** when `HARVEST=1` | The backgrounded loop inherited the command-substitution pipe, which never reached EOF. The joint cell would have burned its **entire run clock at 0 % GPU, before training started**. `HARVEST=0` short-circuits before the `&`, so the A40 never saw it — it was live only on the Modal path. Reproduced in isolation, then fixed. |
| B | rundir resolved by newest **mtime** | The record anchors to the launched pid at all 22 of its call sites and says so explicitly. Under the one-job-per-seed packing §12 recommends, containers share one Volume, and the heartbeat's `Volume.commit()` makes their rundirs mutually visible — "newest" is then a race that scores another seed's rundir and writes a sidecar attesting to it. Now `rundir_for <prefix> <pid>`, with a loud WARN if the anchor misses. |
| C | a **failed cell still wrote a sidecar** | The skip-guard then swallowed the relaunch — the failure §5 warns about, created by the script itself. Now: check `rc`, save the log tail, write no sidecar. |
| D | reg seed glob `*reg_s${SEED}*` matched **`s100` when `SEED=1`** | `ls -t … \| head -1` returns the seed-100 file whenever it is newer, so a seed-1 sidecar would carry seed-100 numbers. Reproduced. Now the exact filename the ablation writes is constructed, as the record does. |
| E | `PY` defaulted to `$REPO/.venv/bin/python`, **which does not exist on the volume** | Confirmed empirically by the preflight: `venv present: NO`. Every rented cell depended on the submit wrapper passing `PY=`. Now resolved in-script with a PATH fallback. |
| F | the **lane log was harvested only from runs that failed** | `$HARVEST_OUT/logs` is created by `save_log()`, which only runs on failure, so the final `cp` silently no-op'd on a clean run. Now `mkdir -p` first, and sidecars are copied home too. |

Added, to close gaps with the record:

- the **joint cell now scores in-container** (`a40_score_matched.py` + `score_joint_best.py`).
  Your `values_from()` already globbed `*a40_matched_score.json`; it will now actually find one.
- **sidecars carry values, commit, protocol block and recipe**, matching
  `run_wave.sh::sidecar_write` field for field, so a rented cell is as mergeable as a local one.
  Two additions beyond the record: **`lane_host`** and **`train_pid`**. §10 needs the machine
  recorded per cell, and until now "where did this number come from" had to be reconstructed from
  a pid and a directory size.
- the charter **§6.6 v17 sanity guard** runs on the joint result.
- `python -u` restored on reg — a buffered log loses its tail when the run clock kills the box.
- **`CELLS`** (default `cat,reg,joint`). Your `--cells` flag previously gated only the preflight
  while the script ran all three anyway; asking for two cells silently bought three, which on a
  joint cell is hours of GPU. Your default behaviour is unchanged.

Every path was smoke-tested against a fake harness before use: clean run, failing cell,
`PARALLEL=1`, resume/skip, and the seed-1/seed-100 collision.

## What was added

- **`scripts/modal_lane.py`** — a Modal entry point for agents without `host.compute`. Does not
  touch your path. Sandbox-based, so stdout streams live; harvests onto the Volume instead of
  `./out/`; archives locally with metadata; terminates in a `finally` and then verifies.
- **`MODAL_MANUAL.md` §0.1** — a decision table so an agent finds its own entry point instead of
  discovering mid-turn that `submit_job` does not exist for it.

## What was NOT touched

- **nespedgpu**: read-only. I listed files, read sidecars, and ran `modal volume put`. No writes,
  and the running v18 wave (seed 1, texas reg at the time) was left alone.
- **Lane 1**, its volume, its bundle, its image, its results.
- **The board**: `docs/results/closing_data/v18/` on the GPU host is untouched. The seed-100 cell
  is archived locally only — see below.
- `v18/run_wave.sh`, `score_all.py`, and everything else under `v18/`.

## The seed-100 cell is archived but NOT merged

`alabama s100` (cat 30.7296, reg 70.0727) is at
`docs/results/closing_data/v18_2/modal_runs/alabama_s100_lane_20260810_021901/`. It is deliberately
not promoted, because two things must be decided first:

1. **`v18/score_all.py` hardcodes `SIDE = docs/results/closing_data/v18`.** Sidecars written to
   the `v18_2` directory are invisible to it — a rented cell simply reads as missing. Either run
   with `OUT=docs/results/closing_data/v18`, or teach the scorer both roots.
2. Promoting means copying `alabama_s100_cat.json`, `alabama_s100_reg.json` and the P1 file onto
   the GPU host's board. That is a board write and belongs to whoever owns the merge.

## Two stale references worth fixing

- `replication_gate.md` cites `data/alabama_s0_cat.json` for its 30.7654 comparand; `data/` was
  deleted on 2026-08-10, so the gate cannot be run as written.
- `v18/run_regen.sh:62` counts sidecars at `$BASE/../../results/…`, which resolves to
  `docs/studies/results/…` and does not exist — the end-of-wave line always logs `0/18`.

## Concurrent editing

Everything under `v18_2/` is untracked, so there is no merge machinery to catch a collision. I
edited `MODAL_MANUAL.md` (§0.1, §4, §6, §9, §10, §13, checklist), `README.md`, `scripts/README.md`,
`EXECUTION_PLAN.md` and `scripts/run_lane.sh`. If you were editing the same files in parallel,
reconcile by section rather than by file.
