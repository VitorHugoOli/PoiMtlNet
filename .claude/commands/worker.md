---
description: Study worker — execute planned tests in a phase (validate → launch → import → analyze).
argument-hint: "<phase> [count|all] [--dry-run]"
---

# /worker — phase execution loop

Takes a phase ID (e.g. `P0`, `P1`, `P2`) and drives the execution side of the
study coordinator workflow. Counterpart to `/coordinator`.

## Arguments

- `<phase>` — required. One of `P0`..`P5`.
- `[count|all]` — optional. Max number of planned tests to run this invocation (default: `1`). Use `all` to drain every remaining planned test in the phase.
- `--dry-run` — optional. Print what would be launched without running `scripts/train.py`.

Examples:
- `/worker P1` — run the next one planned test in P1.
- `/worker P1 5` — run up to 5 planned tests.
- `/worker P2 all --dry-run` — list what would happen for the whole phase.

## What you (the assistant) MUST do when this command fires

1. **Parse `$ARGUMENTS`.** Validate phase is one of P0..P5; refuse otherwise.
2. **Preflight once.** Run `python scripts/study/study.py status` and verify:
   - state.json exists (else tell user to run `/study init`).
   - target phase is not `completed` (refuse unless user forces a re-run).
3. **Main loop, repeat up to `count` times (break early on any failure/surprise):**
   a. `python scripts/study/study.py next --phase <phase>`  — launches the next planned test. Captures `test_id`, `config.state`, `config.engine`, `run_dir` in the printed output and in state.json.
   b. On success, read the just-completed test's `run_dir` from `state.json` (or from the train script output). Run `python scripts/study/study.py import --run-dir <run_dir> --phase <phase> --test-id <test_id> --claims <claim_ids>`.
   c. Run `python scripts/study/study.py analyze --phase <phase> --test-id <test_id>`.
   d. If verdict is `surprising`, `refuted`, or `corrupt`: **STOP the loop, do not launch more tests**, and report the issue to the user including the observed vs expected numbers. Recommend invoking `/coordinator <phase>` for deeper analysis.
   e. If the test `failed` (training crashed): report exit code + last lines of log and STOP.
4. **At the end**, run `/study status` once more and summarize:
   - how many tests ran
   - how many passed / partial / surprising / failed
   - next recommended action (either run `/worker <phase>` again or `/coordinator <phase>` to analyze)

## Invariants

- **Never edit state.json by hand.** Only go through the CLI.
- **Never skip analyze.** A test that's launched but not analyzed leaves the phase in an inconsistent state.
- **Gradient-surgery tests** (`cagrad`, `aligned_mtl`, `pcgrad`) have `--gradient-accumulation-steps 1` injected automatically; do not override.
- If a planned test is missing a `config.state` / `config.engine`, skip it with a warning (operator must enroll it properly).
- Respect `--dry-run`: in dry-run mode, call `/study next --dry-run` and do NOT import/analyze.

## Failure handling

| Situation | Action |
|-----------|--------|
| `state.json` absent | Tell user: `run /study init first` and stop |
| Phase already `completed` | Ask user whether to reopen; do not auto-rerun |
| No planned tests in the phase | Report that phase's queue is empty; suggest `/coordinator <phase>` |
| Preflight input missing | Report which file is missing and suggest `/study validate --state <s> --engine <e>` |
| Training crash | Stop loop; report exit code; do not advance phase |
| Surprising verdict | Stop loop; print expected vs observed; hand off to `/coordinator` |

## Invocation pattern for the assistant

Do this yourself, in sequence, for each test in the loop:

```
python scripts/study/study.py status
python scripts/study/study.py next --phase $PHASE
# parse last_launched_test_id + run_dir from state.json
python scripts/study/study.py import --run-dir <run_dir> --phase $PHASE --test-id <test_id> --claims <C...>
python scripts/study/study.py analyze --phase $PHASE --test-id <test_id>
```

Always use the repo venv python: `.venv/bin/python scripts/study/study.py ...`.

Keep user-facing output tight: one line per test (`[P1_AL_dsk42_al_seed42] launched → imported → verdict=matches_hypothesis (joint=0.52)`), plus a final summary.
