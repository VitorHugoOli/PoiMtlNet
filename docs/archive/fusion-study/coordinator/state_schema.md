# state.json Schema

The state file is the single source of truth on study progress. It lives at `docs/studies/state.json` and is updated by every coordinator action.

---

## Top-level structure

```json
{
  "study_version": "1.0",
  "study_started": "2026-04-14T00:00:00Z",
  "last_update": "2026-04-14T12:34:56Z",
  "current_phase": "P1",
  "open_issues": [ ... ],
  "phases": {
    "P0": { ... },
    "P1": { ... },
    "P2": { ... },
    "P3": { ... },
    "P4": { ... },
    "P5": { ... }
  }
}
```

---

## Phase structure

Each phase entry:

```json
{
  "status": "planned | running | completed | gated | failed",
  "started": "2026-04-14T09:00:00Z",
  "finished": null,
  "expected_tests": 230,
  "completed_tests": 87,
  "failed_tests": 2,
  "surprising_tests": 1,
  "tests": {
    "<test_id>": { ... }
  },
  "summary": "docs/studies/results/P1/SUMMARY.md",
  "gate_check": {
    "required": ["C01 in {confirmed, partial}", "P1c has winner"],
    "passed": false,
    "notes": "Waiting on AZ confirmation"
  }
}
```

---

## Test structure

Each test entry:

```json
{
  "test_id": "P1_AL_screen_dsk42_al_seed42",
  "phase": "P1",
  "stage": "screen",
  "claim_ids": ["C02", "C03", "C05"],
  "status": "planned | running | completed | validated | analyzed | archived | failed | corrupt | surprising",
  "config": {
    "state": "alabama",
    "engine": "fusion",
    "embedding_dim": 128,
    "model_name": "mtlnet_dselectk",
    "model_params": {"num_experts": 4, "num_selectors": 2, "temperature": 0.5},
    "mtl_loss": "aligned_mtl",
    "mtl_loss_params": {},
    "category_head": null,
    "next_head": null,
    "seed": 42,
    "folds": 1,
    "epochs": 10,
    "batch_size": 4096,
    "gradient_accumulation_steps": 1,
    "lr": 1e-4
  },
  "command": "python scripts/train.py --task mtl --state alabama ...",
  "started_at": "2026-04-14T10:00:00Z",
  "finished_at": "2026-04-14T10:01:05Z",
  "wall_clock_seconds": 65,
  "git_commit": "abc123def456",
  "run_dir": "results/fusion/alabama/mtlnet_lr1.0e-04_bs4096_ep10_20260414_1000",
  "results_archive": "docs/studies/results/P1/AL_dsk42_al_seed42/",
  "expected": {
    "joint_range": [0.45, 0.55],
    "cat_f1_range": [0.60, 0.75],
    "next_f1_range": [0.20, 0.32],
    "rationale": "Prior similar config produced joint ~0.50; allow 10% variance."
  },
  "observed": {
    "joint_score": 0.518,
    "joint_std": 0.000,
    "cat_f1": 0.720,
    "next_f1": 0.270,
    "cat_std": 0.000,
    "next_std": 0.000
  },
  "verdict": "matches_hypothesis | partial_match | refutes | surprising | unreliable",
  "notes": "",
  "integrity": {
    "preflight_passed": true,
    "postflight_passed": true,
    "warnings": []
  }
}
```

---

## open_issues structure

For surprises, corruptions, conflicts:

```json
[
  {
    "issue_id": "ISS-001",
    "test_id": "P1_AL_screen_base_ca_seed42",
    "type": "surprising | corrupt | conflict | blocker",
    "raised_at": "2026-04-14T11:00:00Z",
    "description": "Base arch + CAGrad joint=0.55 (expected 0.45-0.50). Needs investigation.",
    "status": "open | acknowledged | resolved | wont_fix",
    "resolution": null
  }
]
```

---

## Example: mid-study snapshot

```json
{
  "study_version": "1.0",
  "study_started": "2026-04-14T00:00:00Z",
  "last_update": "2026-04-14T14:30:00Z",
  "current_phase": "P1",
  "open_issues": [
    {
      "issue_id": "ISS-001",
      "test_id": "P1_AL_screen_mmoe4_eq_seed42",
      "type": "surprising",
      "raised_at": "2026-04-14T11:30:00Z",
      "description": "MMoE+equal_weight better than CGC22+equal_weight, reversing prior finding.",
      "status": "open"
    }
  ],
  "phases": {
    "P0": {
      "status": "completed",
      "started": "2026-04-14T00:00:00Z",
      "finished": "2026-04-14T08:00:00Z",
      "summary": "docs/studies/results/P0/SUMMARY.md"
    },
    "P1": {
      "status": "running",
      "started": "2026-04-14T08:30:00Z",
      "expected_tests": 230,
      "completed_tests": 115,
      "failed_tests": 0,
      "surprising_tests": 1,
      "tests": {
        "P1_AL_screen_base_eq_seed42": { "status": "analyzed", "verdict": "matches_hypothesis", ... },
        "P1_AL_screen_dsk42_al_seed42": { "status": "analyzed", "verdict": "matches_hypothesis", ... },
        "P1_AL_screen_mmoe4_eq_seed42": { "status": "surprising", "verdict": "surprising", ... }
      },
      "gate_check": {
        "required": ["all screens complete", "no open surprising issues"],
        "passed": false
      }
    },
    "P2": { "status": "planned" },
    "P3": { "status": "planned" },
    "P4": { "status": "planned" },
    "P5": { "status": "planned" }
  }
}
```

---

## Atomicity

The state file is a single JSON document. Concurrent writes would corrupt it. Conventions:

1. Only the coordinator process writes to state.json.
2. Writes are atomic: read → modify in memory → write to a tmp file → rename over the original.
3. Before each write, the coordinator re-reads state.json to merge any external edits (e.g., manual human notes).

---

## Migration

When the schema evolves (e.g., adding a new field):

1. Bump `study_version`.
2. Write a migration script in `scripts/study/migrate_state.py`.
3. Document the change here with the old and new shape.

---

## What is NOT in state.json

- Full training logs (too large)
- Per-fold metric tables (live in run_dir)
- Raw model weights (gitignored)

The state file is the **index** into results, not the results themselves.
