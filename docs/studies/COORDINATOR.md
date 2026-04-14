# Coordinator Agent — Specification

The coordinator is a meta-process that orchestrates the study: it knows the plan, tracks state, executes tests, validates results, analyzes outcomes, and raises alarms. It can be invoked manually or run semi-autonomously.

It is not a new product. It's a **workflow** implemented via:
1. A **state file** (`docs/studies/state.json`) — the single source of truth on current progress.
2. A **set of actions** the coordinator can take (listed below).
3. A **Claude Code Skill** (`/study`) that wraps the actions (to be built in Phase 0).
4. **Integrity checks** (see `coordinator/integrity_checks.md`) run before and after each action.

---

## Why a coordinator (and not just a linear script)

- Experiments take hours. Training can fail mid-run. We need crash-resilience.
- Some results will be surprising. The coordinator must recognize this, pause, and request human (or agent) judgment before blindly proceeding.
- Multiple machines run in parallel. Results come back out-of-order. The coordinator merges them with integrity checks.
- New hypotheses emerge as we observe data. The coordinator must support appending claims to the catalog mid-study.

---

## State machine

```
    ┌──────────┐        ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  planned │───────▶│  running │───────▶│ validated│───────▶│ analyzed │
    └──────────┘        └────┬─────┘        └────┬─────┘        └────┬─────┘
                             │                    │                    │
                             ▼                    ▼                    ▼
                        ┌─────────┐          ┌─────────┐          ┌──────────┐
                        │  failed │          │ corrupt │          │surprising│
                        └─────────┘          └─────────┘          └──────────┘
                             │                    │                    │
                             └────────────────────┴────────────────────┘
                                               │
                                               ▼
                                        (coordinator pauses,
                                         raises issue for human/agent review)
```

**States per test:**
- `planned` — listed in a phase doc but not yet started.
- `running` — a training job is live.
- `completed` — training finished (may still have bad data).
- `validated` — integrity checks passed (right shape, right seed, right state, etc.).
- `analyzed` — results compared to expectations; claims updated.
- `archived` — summary.json + metadata.json copied to `docs/studies/results/`.
- `failed` — crashed or returned bad exit code.
- `corrupt` — completed but data integrity failed.
- `surprising` — analyzed but results fall outside expected range; requires review.

---

## Actions the coordinator performs

### Preflight (before any test)

1. **Read state.json** to know current phase + pending tests.
2. **Check embeddings exist** for the target state(s).
3. **Validate input files** (see `coordinator/integrity_checks.md`):
   - Category parquet: rows × (128 + 2), label distribution within tolerance
   - Next parquet: rows × (1152 + 2), label distribution within tolerance
   - Check point-of-embedding integrity (no NaN, reasonable norms)
4. **Verify git state is clean** (or record current commit).
5. **Verify required code dependencies** (loss registry has target optimizer, arch registry has target model).

### Execution

6. **Launch training** with the exact CLI from the phase doc, using seed 42 (or list of seeds for replicas).
7. **Record in state.json** under the current test: status `running`, start_time, pid, command, git_commit.

### Post-hoc (after training exits)

8. **Read the results directory** produced by `scripts/train.py`.
9. **Validate**:
   - `full_summary.json` exists and has all expected keys
   - Per-fold metrics exist for all folds
   - joint_score, cat F1, next F1 within plausible range (e.g., [0, 1])
   - No NaN or Inf in metrics
   - Check manifest matches: same seed, same config, same dataset hash
10. **Archive**:
    - Copy `full_summary.json` and a slim metadata.json to `docs/studies/results/<phase>/<test_id>/`.
    - Metadata includes: test_id, phase, claim_ids, config, timestamp, git_commit, wall_clock, status.

### Analysis

11. **Analyze** by comparing the result to the test's expected outcome (from the claim). Outcomes:
    - **Matches hypothesis (direction + magnitude):** mark claim `confirmed`; move on.
    - **Matches direction but magnitude off:** mark claim `partial`; flag for follow-up.
    - **Directly contradicts:** mark claim `refuted`; **pause and escalate**.
    - **Unexpected:** flag as `surprising` and append a new candidate claim to the catalog.

12. **Update CLAIMS_AND_HYPOTHESES.md** with the new status and evidence pointer.

13. **Update state.json** — test moved from `running` → `analyzed` (or `failed`, `corrupt`, `surprising`).

### Decision

14. Based on state.json and the phase doc's "next action" rules, pick the next test or decide the phase is complete.

15. If the phase is complete, check the phase's decision gate (e.g., "proceed to P2 only if MTL > single-task with p < 0.05"). If gate fails, escalate.

---

## Actions the human (or an Agent) must take

The coordinator never does these alone:

- **Phase transitions:** reviewing the phase summary and deciding whether to continue.
- **Reframing the study:** if a Tier A claim is refuted, the paper story may need to change; a human decides.
- **Adding new claims:** if analysis suggests a new hypothesis worth testing, a human approves its addition to the catalog and the phase plan.
- **Running on other machines:** the coordinator on this machine doesn't dispatch remote jobs — it only validates imports from other machines.
- **Interpreting edge cases:** any `surprising` or `corrupt` flag should ping a human.

---

## state.json schema

See `coordinator/state_schema.md` for the full JSON schema. Example top-level shape:

```json
{
  "study_version": "1.0",
  "current_phase": "P1",
  "last_update": "2026-04-14T09:00:00Z",
  "phases": {
    "P0": { "status": "completed", "started": "...", "finished": "..." },
    "P1": {
      "status": "running",
      "started": "...",
      "tests": {
        "P1_AL_screen_dsk42_al_seed42": {
          "status": "analyzed",
          "claim_ids": ["C01", "C02", "C04"],
          "command": "python scripts/train.py ...",
          "git_commit": "abc123",
          "results_dir": "docs/studies/results/P1/AL_dsk42_al_seed42/",
          "expected": { "joint_range": [0.45, 0.60] },
          "observed": { "joint": 0.52, "cat_f1": 0.75, "next_f1": 0.29 },
          "verdict": "matches_hypothesis",
          "notes": ""
        },
        "...": "..."
      }
    }
  },
  "open_issues": [
    { "test_id": "...", "flag": "surprising", "description": "..." }
  ]
}
```

---

## Invoking the coordinator

### Manual (current, before the Skill is built)

```bash
# Describe current state
cat docs/studies/state.json | jq .current_phase

# Run next pending test (by looking at the phase doc, then launching)
# [manual for now; Skill in P0 will automate]
```

### Via Skill (to be built in Phase 0)

```
/study status            # show current phase, pending tests, open issues
/study next              # run the next pending test in the current phase
/study import <run_dir>  # archive a result from another machine
/study validate P1       # run integrity checks on all tests in a phase
/study analyze <test_id> # re-run analysis on a specific test
/study claim <C-id>      # show status of a claim with evidence pointers
/study advance           # move to next phase if current phase's gate passes
```

### Via subagent (for autonomous exploration)

The `Agent` tool can be invoked with a prompt like:
> "Using docs/studies/state.json and CLAIMS_AND_HYPOTHESES.md, run the next 3 pending tests in phase P1, validate each, update state, and summarize. Stop and report if any test is marked `surprising` or `refuted`."

---

## Critical thinking loop (what makes this more than a script)

After every test that completes:

1. **Does the result match the hypothesis?**
2. **If yes:** is the effect size consistent with what we'd need for a paper claim?
3. **If no:** is this noise, a bug, or a real refutation? Consider:
   - Compare to pre-bug data (if available)
   - Check integrity thoroughly
   - Run with a second seed before concluding
4. **Does the result suggest new questions?** E.g., if MTL ≈ single-task, maybe single-task is undertrained — test with more epochs? Or maybe our MTL has a bug — check gradient flow?
5. **Should the plan change?** A truly surprising result might mean the rest of the phase is moot. Don't waste compute on tests whose answers no longer matter.

The coordinator surfaces these questions to a human. It doesn't answer them unilaterally.

---

## Failure modes to explicitly handle

| Mode | Detection | Response |
|------|-----------|----------|
| Training crashes mid-run | exit code ≠ 0 | mark `failed`; keep the partial log; retry once with same seed |
| Training returns NaN/Inf F1 | check in validation step | mark `corrupt`; do not archive; flag to human |
| Results dir has wrong dataset hash | integrity check | mark `corrupt`; something upstream changed |
| Two tests with same ID exist | state.json check before running | reject the duplicate; warn user |
| Test result is within noise of control | analysis step | mark `partial`; propose follow-up with more seeds |
| Test contradicts a confirmed claim | analysis step | mark `surprising`; pause the phase |
| Imported result from other machine has different seed than planned | import validation | reject; request correct seed |
| Embedding files missing | preflight | mark test `blocked`; surface to user |

---

## How this coordinator is different from a plain experiment runner

A plain runner just executes commands and reports exit codes. Our coordinator:
- Owns the experimental plan (knows what should be run next)
- Owns the hypothesis catalog (knows what each test is supposed to prove)
- Validates beyond "did the command return 0" (checks data integrity and result plausibility)
- Updates the scientific record (marks claims `confirmed`/`refuted`, adds new claims)
- Knows when to stop (phase gates, surprising results)
- Is auditable (every decision is a log entry in state.json)

---

## Build checklist for Phase 0

- [ ] `state.json` initial template + schema doc
- [ ] Integrity-check script (`scripts/study/validate_inputs.py`)
- [ ] Result-archive script (`scripts/study/archive_result.py`)
- [ ] `/study` Skill in Claude Code (`commands/study/`) — lightweight wrapper, delegates to scripts
- [ ] First integration test: run one P1 test end-to-end through the coordinator
- [ ] Claim-update helper (`scripts/study/update_claim.py`) — updates CLAIMS_AND_HYPOTHESES.md in-place
