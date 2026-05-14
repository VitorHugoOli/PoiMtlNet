# Coordinator Agent — Check2HGI Study Specification

The coordinator workflow is **shared** with the fusion study — see `docs/studies/fusion/COORDINATOR.md` for the full specification (state machine, actions, preflight/execution/postflight/analysis phases, claim-update protocol, abort conditions, operational guarantees).

This document captures only the **check2HGI-specific bindings** of the shared workflow.

---

## Study binding

The shared coordinator commands (`/coordinator`, `/worker`, `/study`) pick up the study via the `STUDY_DIR` environment variable. For this study, set:

```bash
export STUDY_DIR=docs/studies/check2hgi
```

Without the override, commands default to `docs/studies/fusion/` and will operate on the wrong study.

---

## State file

- **Path:** `docs/studies/check2hgi/state.json`
- **Schema:** same as fusion — see `docs/studies/check2hgi/coordinator/state_schema.md` (which points at the fusion spec).

---

## Integrity checks

**Path:** `docs/studies/check2hgi/coordinator/integrity_checks.md`.

Overrides vs fusion's:

- **Schema shapes** differ (next-POI label is a placeid → poi_idx lookup; next-region label is a region_idx; no category labels).
- **Class-distribution checks** are per-engine (HGI / Check2HGI) and per-task (next_poi / next_region).
- **Scale-ratio checks** do NOT apply (no fusion half-L2 ratio).
- **PI.5 embedding value sanity** applies identically.

---

## Phase binding

| Phase | Doc |
|---|---|
| P0 | `docs/studies/check2hgi/phases/P0_preparation.md` |
| P1 | `docs/studies/check2hgi/phases/P1_single_task_baselines.md` |
| P2 | `docs/studies/check2hgi/phases/P2_mtl_headline.md` |
| P3 | `docs/studies/check2hgi/phases/P3_dual_stream.md` |
| P4 | `docs/studies/check2hgi/phases/P4_cross_attention.md` |
| P5 | `docs/studies/check2hgi/phases/P5_ablations.md` |

---

## Claim catalog

**Path:** `docs/studies/check2hgi/CLAIMS_AND_HYPOTHESES.md`.

Uses `CH##` prefix to avoid collisions with fusion's `C##`. CH-claims from this study do not appear in the fusion state.json and vice versa.

---

## Decision gates specific to this study

- **P0 exit:** requires CH14 (fclass-shortcut audit) resolved. If shortcut is present in Check2HGI, P1/P2 test plans must be amended before proceeding.
- **P4 entry gate:** P3 must show ≥ 2pp Acc@10 lift from dual-stream input on Florida, otherwise P4 is documented as future work and skipped.
- **Branch merge gate:** all Tier-A (CH01, CH02, CH03) and ≥ 2 of 3 Tier-C region-input claims (CH06, CH07, CH11) must have evidence pointers.

---

## Escalation & human-in-the-loop

Same protocol as fusion's COORDINATOR.md:

- Any test with status `surprising` pauses the coordinator until a human acknowledges.
- Any `corrupt` result requires regeneration — the coordinator does not silently retry.
- Cross-study claim conflicts (e.g., if a Check2HGI result contradicts a fusion result at the shared-code level) are escalated to the user.

---

## Skills (branch-scoped)

```bash
STUDY_DIR=docs/studies/check2hgi /coordinator P0    # analyze P0, check gate
STUDY_DIR=docs/studies/check2hgi /worker P1         # run next planned test
STUDY_DIR=docs/studies/check2hgi /study status      # current phase + test statuses
STUDY_DIR=docs/studies/check2hgi /study import <path>  # archive a test result
STUDY_DIR=docs/studies/check2hgi /study advance     # phase gate → flip to next
```
