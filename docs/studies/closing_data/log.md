# closing-data — study log (append-only, newest at the bottom)

## 2026-06-12 — Study SCAFFOLDED (not launched)

**Phase**: pre-launch scaffold, created at the close of `mtl_improvement` (design agent + user).

**What exists**
- `AGENT_PROMPT.md` (mission, hard rules C25–C28, read-first) + `PLAN.md` (DRAFT v0 phases:
  P0 pre-freeze gates → P1 cross-study re-eval sweep → P2 recipe FREEZE barrier → P3 CA/TX majors →
  P4 final tables/doc sync).
- Inherited from `mtl_improvement` (CLOSED 2026-06-12, see its `FINAL_SYNTHESIS.md`):
  the **G0.1 aligned-pairing pre-freeze gate** (spec: `docs/results/mtl_improvement/X_SERIES_FINDINGS.md §X1`),
  the **CA/TX majors spec** (mtl_improvement INDEX `#T6-1`), and the scale checks to fold in
  (HSM at 8.5k/6.5k; `next_conv_attn` FL-only cat lever).

**Decision**
- DRAFT only — launch requires user sign-off on `PLAN.md` (incl. the 3 open questions at its foot).

**Next**
- User: review/refine `PLAN.md`; decide launch timing (after remaining studies settle).

---

## 2026-06-12 — RE-SCOPED (pre-launch, user): full experimental base for the NEW paper, not just CA/TX completeness

**Phase**: pre-launch draft refinement (design agent + user).

**What changed (PLAN.md v0 → v1)**
- The study is the **experimental engine for the NEW paper** (story defined in a follow-up effort;
  may not reuse every BRACIS experiment — but the base is regenerated here, story-agnostic).
- Scope expanded from "CA/TX majors" to: **STL baselines RE-RUN + MTL champion + every relevant
  BRACIS-suite experiment, at ALL states × 4 seeds {0,1,7,100} × 5 folds (n=20/cell)** under the
  frozen recipe.
- New **P1b — BRACIS suite inventory → `RUN_MATRIX.md`** (walks `TABLES_FIGURES.md` T1–T5 +
  `RESULTS_TABLE.md §0.1–0.6` + the baselines strategy; per cell: RE-RUN / REUSE / STORY-DEPENDENT
  with exact run specs). The matrix is what Phase 3 executes and what the user signs off with the
  freeze.
- P2 freeze now ALSO pins the **substrate identity** (v14 vs canonical — user decision, drives
  M0/M1 scope) and the n=20 protocol for every cell.
- P3 restructured: M0 missing artifacts (CA/TX builds + 4-seed log_T everywhere + baseline-engine
  artifacts) → M1 STL baselines re-run (ceilings, composite, external engines) → M2 champion all
  states → M3 remaining suite cells → M4 one full-board matched-metric re-score with per-cell
  provenance (the new paper's source of truth).
- Open questions updated (substrate identity; whether the external-baseline T5 set is kept at the
  full n=20 protocol).

**Next**
- User: sign off PLAN.md v1 (4 open questions at its foot); follow-up effort defines the new
  paper's story; launch when the remaining studies are settled.

---

## 2026-06-12 — Pre-launch questions RESOLVED (user) → PLAN v2; machine allocation defined

**Phase**: pre-launch draft refinement (design agent + user).

**Decisions (user)**
1. Studies settled — working assumption YES; P1a verifies (`merge_design` first).
2. **Substrate = v14** or whichever newer blessed base exists at launch (re-check
   `CANONICAL_VERSIONS.md` then). Single-substrate board.
3. **External baselines: RE-RUN under the NEW regime** at the full n=20 protocol — no reuse of
   BRACIS-era lighter-protocol numbers. P1b dispositions per-engine existence/relevance only.
4. No timeline pressure. **Execution split across 3 machines**: H100 (**6 h TOTAL** — the metered
   burst, default = CA/TX v14 builds, job list must be timed on the A40 before spending), A40
   (unmetered workhorse — the run board), M4 Pro 32GB (prep/scoring/small-state lane, MPS caveats
   per `docs/infra/`). Coordination: per-machine manifests merged at M4; C28 rundir rules.

**Remaining launch gate**: user sign-off on `RUN_MATRIX.md` (P1b) together with the P2 freeze.

---

## How to add an entry

```markdown
## YYYY-MM-DD — Short title

**Phase**: P?.? in flight / completed / paused.
**What ran** / **Result** (numbers, per-state) / **Decision** / **Blocker** / **Next**
```
Rules: append at the bottom; never edit historic entries; if a gate promotes, STOP and record the
user decision before proceeding.
