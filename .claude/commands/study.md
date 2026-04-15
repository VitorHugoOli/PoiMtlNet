---
description: Study coordinator — status, next test, import, validate, analyze, claim lookup.
argument-hint: "<subcommand> [args...]"
---

# /study — ablation study coordinator

Thin wrapper over `scripts/study/study.py`. Use this to orchestrate the
claim-driven ablation described in `docs/studies/fusion/`.

## What you (the assistant) must do when this command fires

1. Parse `$ARGUMENTS` — the first token is the **subcommand**, the rest are passed through.
2. Run the CLI via Bash: `python scripts/study/study.py <subcommand> [args]` (use the repo venv: `.venv/bin/python`).
3. Report the output to the user verbatim (plus a one-line summary if useful).
4. If a subcommand is missing a required argument, print the usage block below and stop. Do NOT guess arguments.
5. Never edit `docs/studies/fusion/state.json` by hand — always go through the CLI so writes stay atomic.

## Subcommands

| Form | What it does |
|------|---|
| `/study init [--force]` | Write initial `docs/studies/fusion/state.json` skeleton (P0 running, P1–P5 planned). Refuses to overwrite without `--force`. |
| `/study status` | Summarize current phase, per-phase test counts, and open issues. |
| `/study validate --state <s> --engine <e> [--cross <e2> <e3>...]` | Run parquet integrity checks; writes JSON to `docs/studies/fusion/results/P0/integrity/`. |
| `/study next [--phase <P>] [--test-id <id>] [--dry-run]` | Launch the next `planned` test in the current (or specified) phase. Updates state.json atomically. |
| `/study import --run-dir <path> --phase <P> --test-id <id> [--claims C01 C02] [--overwrite]` | Archive a training run (from this or another machine) into `docs/studies/fusion/results/<phase>/<test_id>/` and update state. |
| `/study analyze --phase <P> --test-id <id> [--tolerance 0.03]` | Compare observed vs expected ranges, set verdict, open an issue if `surprising` or `corrupt`. |
| `/study claim <C-id>` | List tests that reference a claim, with their status and joint F1. |
| `/study advance [--force]` | Move `current_phase` forward once the current phase has no `planned`/`running` tests (or force it). |



## Conventions

- Test IDs follow `P{n}_{STATE}_{short-config}_{seed}` (e.g. `P1_AL_dsk42_al_seed42`).
- Claim IDs come from `docs/studies/fusion/CLAIMS_AND_HYPOTHESES.md` (C01..Cnn).
- `expected` ranges are set when a test is enrolled into state.json from a phase doc.
- Gradient-surgery losses (`cagrad`, `aligned_mtl`, `pcgrad`) automatically get
  `--gradient-accumulation-steps 1` injected by `launch_test.py`.

## Typical workflows

### First time in a fresh clone
```
/study init
/study validate --state alabama --engine dgi --cross hgi fusion
/study validate --state alabama --engine fusion
/study status
```

### Run the next planned test and record the result
```
/study next              # launches training
/study import --run-dir results/fusion/alabama/mtlnet_lr... --phase P1 --test-id P1_AL_...
/study analyze --phase P1 --test-id P1_AL_...
```

### Import a result from another machine
```
/study import --run-dir /path/to/results/... --phase P1 --test-id P1_FL_champ_seed42 \
  --claims C01 C11
```

## Invocation

```bash
python scripts/study/study.py $ARGUMENTS
```

Run from the repo root (`/Volumes/Vitor's SSD/ingred`) and prefer the repo
venv python: `.venv/bin/python scripts/study/study.py ...`.
