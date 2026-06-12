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

## How to add an entry

```markdown
## YYYY-MM-DD — Short title

**Phase**: P?.? in flight / completed / paused.
**What ran** / **Result** (numbers, per-state) / **Decision** / **Blocker** / **Next**
```
Rules: append at the bottom; never edit historic entries; if a gate promotes, STOP and record the
user decision before proceeding.
