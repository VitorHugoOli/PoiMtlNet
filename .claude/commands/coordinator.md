---
description: Study coordinator — analyze a phase, update claims, check phase-gate, recommend next action.
argument-hint: "<phase> [--dry-run]"
---

# /coordinator — phase analysis & decision

Takes a phase ID (e.g. `P1`) and performs the **thinking** side of the
workflow: synthesize results across all tests, cross-check against claims,
flag surprises, and decide whether the phase can advance.

Counterpart to `/worker` (which does the execution). Use `/coordinator` after
`/worker` finishes a batch, or whenever state.json has new analyzed tests.

## Arguments

- `<phase>` — required. One of `P0`..`P5`.
- `--dry-run` — optional. Do analysis + print recommendations but don't mutate `CLAIMS_AND_HYPOTHESES.md` or call `/study advance`.

Examples:
- `/coordinator P1` — analyze all P1 tests and produce a phase summary.
- `/coordinator P2 --dry-run` — preview the gate check without updating claims.

## What you (the assistant) MUST do when this command fires

1. **Parse `$ARGUMENTS`.** Validate phase is one of P0..P5.
2. **Collect state.** Run:
   - `python scripts/study/study.py status` — overall snapshot.
   - Read `state.json` directly (path: `$STUDY_DIR/state.json`, default `docs/studies/fusion/state.json`) for full test details in the target phase.
   - Read the corresponding `phases/<phase>_*.md` inside the study dir — it defines the phase gate and expected tests.
   - Read `CLAIMS_AND_HYPOTHESES.md` in the study dir — for claim definitions touched by the phase.
3. **Per-test sanity sweep.** For each test in the phase:
   - If `status ∈ {completed, analyzed, archived}` but `verdict` is absent: run `/study analyze --phase <phase> --test-id <id>`.
   - If `status == running` and `started_at` > 24 h ago: flag as possibly stuck.
   - If `status == failed`: note failure reason.
4. **Claim rollup.** For each claim referenced by any test in this phase:
   - Count matches vs partials vs refutations across tests.
   - Decide candidate status: `confirmed` / `partial` / `refuted` / `pending`.
   - Propose an edit to `CLAIMS_AND_HYPOTHESES.md` (show diff). Apply only if not `--dry-run`.
5. **Surprise review.** For each `surprising` test:
   - Compare observed vs expected, reason about it (variance? confound? genuine refutation?).
   - Suggest a follow-up: second seed, fixed confound, new claim, or accept.
   - **Do not auto-resolve** — pause and ask the user for confirmation.
6. **Gate check.** Using the phase doc's gate criteria:
   - List which conditions are met / unmet.
   - If all met: recommend `/study advance`.
   - If some missing: list the specific tests or evidence still needed.
7. **Write a phase summary.** Append / update `results/<phase>/SUMMARY.md` inside the study dir with:
   - Run counts and wall-clock budget used
   - Claim status table
   - Open surprises + recommended follow-ups
   - Gate check result
8. **End with a crisp recommendation.** One of:
   - "`/worker <phase>` — still N planned tests outstanding"
   - "`/study advance` — phase gate passes"
   - "Pause for user: surprising result on `<test_id>` needs judgment"

## What you MUST NOT do

- Never edit state.json directly — use `/study`.
- Never mark a claim `confirmed` / `refuted` on evidence from a single test without cross-checking against other tests in the phase and prior claims.
- Never silently overwrite `CLAIMS_AND_HYPOTHESES.md` — present the proposed change first unless `--dry-run` is absent AND the change is an obvious additive status update.
- Never invoke `/study advance` without explicit gate evidence.

## Critical-thinking loop (apply per analyzed test)

1. Does the observed value fall inside expected range? (handled by `analyze_test.py`)
2. If outside — is the deviation directional (as hypothesized) or reversed?
3. Is variance plausible? Compare `std` across folds; if std is a large fraction of the mean, mark `unreliable` and request another seed.
4. Does one test contradict a claim that other tests in this or a prior phase confirmed? Flag as **conflict** and pause.
5. Does the result open a new question not in `CLAIMS_AND_HYPOTHESES.md`? If yes, propose a new claim `C22+` with: statement, test plan, predicted outcome. Ask the user to approve before appending.

## Output format to user

Keep it scannable. Suggested template:

```
# Coordinator — Phase P1 (2026-04-18T09:30Z)

Tests: 12 total → 9 archived, 2 analyzed, 1 surprising

Claims touched:
  C01  fusion > HGI     CONFIRMED   evidence: P1_AL_champ, P1_AZ_champ
  C02  grad-surgery > eq PARTIAL    evidence mixed at matched batch
  C05  expert > FiLM    CONFIRMED
  C06  MTL > single-task PENDING    (waits on P2)

Surprises:
  P1_AL_mmoe4_eq — joint=0.57 vs expected [0.45, 0.52]
    → likely real; propose re-run with seed=123 to verify

Gate check: FAIL (C02 still partial; 3 planned AZ runs outstanding)
Recommendation: /worker P1 3   # drain AZ replicas, then re-run /coordinator P1
```

## Invocation pattern

Run steps yourself via Bash (`.venv/bin/python scripts/study/study.py ...`) and Read. Never delegate the synthesis — you must reason about the numbers.

If the target phase is empty or brand-new, respond: "Phase `<phase>` has no analyzed tests yet — run `/worker <phase>` first."
