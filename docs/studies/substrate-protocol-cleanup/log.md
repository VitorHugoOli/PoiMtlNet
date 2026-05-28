# substrate-protocol-cleanup — Progress Log

Append-only chronological log. Every agent working on this track adds entries here.
Dates are absolute (e.g. `2026-05-28`), never relative ("today", "yesterday").

Sections at the bottom of each entry:
- **Decision** if you changed direction.
- **Blocker** if you got stuck (and what unblocked you, in a later entry).
- **Findings** if results landed.
- **Next** what the next agent should pick up.

---

## 2026-05-28 — Study launched

**Phase**: Design complete; no experiments run yet.

**What happened**

- Folder `docs/studies/substrate-protocol-cleanup/` created alongside `mtl-protocol-fix/` and `mtl_improvement/`.
- `AGENT_PROMPT.md`, `INDEX.md`, `considerations.md`, and this `log.md` landed.
- Future-works re-routing applied in `docs/future_works/README.md` and the affected memos (`mtl_architecture_revisit.md`, `substrate_adaptive_mtl_balancing.md`, `head_window_batch_audit.md`, `reg_head_architecture_sweep.md`, `composite_two_substrate_engine.md`).

**Scope captured (from user 2026-05-28 conversation)**

- Substrate + small MTL fixes only. Anything on the architectural axis goes to `mtl_improvement` (branch `mtl-improve`).
- Small states (AL, AZ) only for main sweeps; FL/CA/TX as 1-fold pilots only.
- §4.1 confirmed as **variant A** (3 internally-consistent MTL snapshots routed by task at deploy). Variant C (mixed-epoch heads + backbone) is explicitly rejected as incoherent.
- §4.7 confirmed as MTL retrain at small states (Designs B, J have STL numbers only — MTL+F1 was never run).
- New study folder (not Phase 4 of `mtl-protocol-fix`) to preserve the v6-final closure provenance of the parent study.

**Decision** — Tier order is D (no GPU, anytime) → A (cheap multi-seed promotion) → B (substrate cross-study) → C (protocol coherence). D may run in parallel with A.

**Decision** — Variant A for §4.1, not variant C. Documented in `INDEX.md` §C1.

**Decision** — Decision-gate hardness: every Tier ends with a gate that either promotes (move to multi-seed) or archives (write the null result). No "let's try one more thing" without re-opening the design.

**Blocker (resolved)** — `analyze_t64_selectors.py` cannot zero-retrain Designs B/J because their per-epoch val CSVs are STL-only, not MTL. Tier B requires real MTL training at AL/AZ (~8 GPU-h per design). Captured in `INDEX.md` §B framing.

**Advisor pass (2026-05-28)** — Ran a general-purpose advisor agent across all studies (mtl-protocol-fix, mtl_improvement, merge_design, hgi_category_injection, canonical_improvement) + CONCERNS + CLAIMS_AND_HYPOTHESES + future_works re-routing. Findings applied to the study:

1. **Tier B4 added: Lever 5 (KL distill) absorbed** — orphan rescue. `merge_design` is closed; Lever 5 has no other live owner; ~3 GPU-h at AL+AZ. Independent of architectural champion. Added to `INDEX.md` §B4 and `considerations.md` §"Why Levers 4 and 5 are in this study, Lever 6 is not".
2. **Tier C3 added: P4 K/V capacity-stealing test** — P4 frozen-cat froze cat parameters but cat activations still flow through cross-attention K/V. Near-zero-compute pilot (`--zero-cat-kv` flag, ~1 day code + 4 GPU-h total) closes or surfaces a residual mechanism for `mtl_improvement` to target. Added to `INDEX.md` §C3 and `considerations.md` §"The P4 frozen-cat residual hole".
3. **D1 ↔ T0.2 explicit handoff** — `mtl_improvement` T0.2 plans the same mask audit on a separate branch. Codified a first-to-claim protocol in `INDEX.md` §"D1 ↔ T0.2 handoff" so both studies share one artefact.
4. **Branch-coordination protocol** — added explicit rebase cadence for `BestTracker` (C1) and freeze-logic (C2) collision risks with `mtl_improvement`. In `INDEX.md` §"Branch-coordination protocol with mtl_improvement".
5. **Variant C-prime** — acknowledged as a deferred re-open trigger in `considerations.md`. Not in scope; runs only if variant A promotes AND deploy storage becomes binding.
6. **Cross-references hardened** — added C18 (encoder-swap leak-probe MONITORED) to "Open concerns this study touches"; added explicit "NOT IN SCOPE" cross-reference to `hgi_category_injection` FL/CA/TX re-open; added project-headline §4.2 composite cross-reference so Tier A's smaller +2-5 pp is not mistaken for the project's strongest reg lift; D1 prior-art now cites F50_T4_C4_LEAK_DIAGNOSIS.md.

Net effect: study scope grew by Tier B4 + Tier C3 (~7 GPU-h, ~1 day code), branch-collision risk explicitly mitigated, advisor's 5 critical gaps closed.

**Next**

1. Implementing agent should read in order: this log, `INDEX.md` (Tier by Tier), `considerations.md`, `docs/studies/mtl-protocol-fix/DEFERRED_WORK.md`, `docs/results/mtl_protocol_fix/phase3_summary.md`, `docs/CONCERNS.md` C15/C21/C22/C23.
2. **Recommended start:** Tier D (window/mask audit, 1 day no GPU) in parallel with Tier A pre-flight (verify C22 stale log_T at AL/AZ for the 4 seeds {0,1,7,100}).
3. Use `TaskCreate` to break down each Tier into validate → code-change → unit-test → re-evaluate → analyse sub-tasks.
4. After each Tier completes, write the verdict into `INDEX.md` + a finding doc under `docs/findings/` if promoted.
5. After the full study completes, run a final advisor pass on the whole `INDEX.md` + `log.md` before declaring done.

---

## (template — copy and date for next entry)

## YYYY-MM-DD — <Short title>

**Phase**: <Tier A / B / C / D / Final synthesis>

**What happened**

- <bullet>
- <bullet>

**Decision** (only if changed direction):
- <what changed and why>

**Blocker** (only if stuck):
- <what's blocked>
- <what you tried>

**Findings** (only if results landed):
- per state, three-frontier numbers (best joint / best disjoint / STL)
- statistical significance / Wilcoxon p / fold-by-fold deltas
- **Verdict**: <"promoted" | "null" | "falsified" | "inconclusive at n=5">

**Next**:
- <experiment ID> next, or <decision needed>

---
