# Agent Onboarding Prompt — mtl-protocol-fix study

> ⚠ **STUDY CLOSED 2026-05-20 v6 final.** This document is preserved for historical record. If you arrive here looking for active work, redirect to the next-tier study: **[`docs/future_works/mtl_architecture_revisit.md`](../../../future_works/mtl_architecture_revisit.md)** (highest-EV per P4 frozen-cat mechanism finding).
>
> Closure verdict: [`docs/results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md`](../../../results/mtl_protocol_fix/phase1_phase2_verdict_v6_final.md). Read [`log.md`](log.md) end-to-end for the full execution trail (Phase 0 → 2 P5 stale log_T audit → Phase 3 residual-gap brief).

---

> **Paste this prompt (or its core block) at the start of every fresh Claude Code session that picks up this research track.** It onboards you with everything you need before touching code.

---

## Your role

You are the **implementing agent** for the `mtl-protocol-fix` study. The predecessor study (`canonical_improvement`, closed 2026-05-19) exhausted the substrate axis on next-reg (Tier 1-6, 26 mechanism families, ceiling ±0.8 pp). The load-bearing finding from that study was **C21**: the production `joint_canonical_b9` selector at `src/training/runners/mtl_cv.py:679` throws away ~10.7 pp of reg-top10 capacity from the canonical Check2HGI shipping recipe itself at FL, with no substrate change involved.

**You execute the protocol fix and characterise the residual MTL-vs-STL reg gap that survives it.** You are NOT locked into the design — if results redirect you, propose and pursue the new path. Document the redirection in `log.md` first.

## Required reading (in this order, before any code change)

| # | File | Purpose |
|---|------|---------|
| 1 | `docs/studies/archive/mtl-protocol-fix/log.md` | Most recent progress and decisions |
| 2 | `docs/studies/archive/mtl-protocol-fix/INDEX.html` | Full experiment design + three-frontier protocol + falsified-history |
| 3 | `docs/studies/archive/mtl-protocol-fix/considerations.md` | The four conceptual questions the study must answer |
| 4 | `docs/CONCERNS.md` C21 | Selector-bug full diagnosis |
| 5 | `docs/CLAIMS_AND_HYPOTHESES.md` CH23-A/B | Paper-facing claim status |
| 6 | `docs/studies/archive/canonical_improvement/log.md` 2026-05-19 final entry | What the predecessor study closed; falsified-history binding for substrate work |
| 7 | `docs/studies/mtl-exploration/FUTUREWORK_substrate_aware_mtl_balancing.md` | Predecessor F1/F2/F3 memo (F2/F3 split into future_works) |
| 8 | `docs/NORTH_STAR.md` | Canonical training recipe (B9 MTL) + selector limitation banner |
| 9 | `src/training/runners/mtl_cv.py:679` and `:710` | The bug site + the already-coded alternative selector |
| 10 | `scripts/canonical_improvement/analyze_t64_selectors.py` | Re-selection tool (zero retraining) |

## Hard rules (do not break)

1. **No new substrate work.** Tier 1-6 of canonical_improvement closed the substrate axis at ±0.8 pp. Do not re-open Check2HGI internals (boundaries, encoders, edge types, POI features) under this study. Substrate falsified-history applies.

2. **No retraining for Phase 1 numbers.** F1 selector fix is a one-line code change at `mtl_cv.py:679`. AL/AZ/FL/CA/TX re-evaluation uses existing per-fold val CSVs in `results/check2hgi/{state}/mtlnet_*/`. **If a Phase-1 number requires retraining, you have made a mistake — stop and check.**

3. **Three-frontier reporting MANDATORY.** Every MTL number must be reported under: (a) `joint_canonical_b9` legacy, (b) `joint_geom_simple` (the new principled selector), (c) per-task disjoint best (substrate-capacity ceiling). STL numbers report at per-task best only.

4. **No paper-canon §0.1 multi-seed re-evaluation in this study.** That is deferred to `docs/future_works/paper_canon_reevaluation.md` (sequenced AFTER `mtl_architecture_revisit.md`). Phase 1 uses single-seed=42 only; Phase 3 may extend selected cells to multi-seed but only if the gap-characterisation requires it.

5. **No MTL architecture revisit.** `mtl_architecture_revisit.md` is the next-tier study. Stay focused on protocol/selector axis here.

6. **Statistical claims need n ≥ 10.** This study's primary outputs are single-seed n=5 (paired Wilcoxon p ≥ 0.0312 ceiling) at first; promote selected cells to multi-seed (n=20) only if Phase 3 requires statistical power.

7. **Pre-flight gate (Phase 0)**: reproduce the matched-protocol numbers in `docs/results/canonical_improvement/T6_4_dual_selector_final.json` (shipping FL ep=50 single-seed=42 n=5) under the F1-fixed code path. If the gate fails, the F1 fix has unintended side effects — debug before any Phase 1 work.

8. **Falsified history off-limits.** Substrate Tier 1-6 closures hold; the F1/F2/F3 split locks F2 (loss balancing) and F3 (substrate × protocol 2×2) into future-work, not this study.

## Required workflow

1. **Create the dedicated git worktree first**. Branch `mtl-protocol-fix` from `main`. Do not contaminate other ongoing work.

   ```bash
   git fetch origin
   git worktree add ../worktree-mtl-protocol-fix -b mtl-protocol-fix main
   cd ../worktree-mtl-protocol-fix
   ```

2. **Use `TaskCreate` / `TaskUpdate` to break down every experiment** into sub-tasks: `code-change → unit-test → validate → re-evaluate → analyze`. Mark `in_progress` before starting, `completed` only when:
   - Results are filled in the matching `<div class="results-placeholder">` in `INDEX.html`.
   - A `log.md` entry is written.
   Never leave a task `in_progress` overnight without an explicit blocker note.

3. **After each Phase completes, call advisor** with that Phase's results. Capture advisor feedback in `log.md`. Apply revisions before moving to the next Phase.

4. **After the full study completes (or at any major redirect), spawn a mandatory advisor sub-agent** to evaluate the F1 fix correctness, the three-frontier evaluation protocol, the residual-gap characterisation. Capture feedback in `log.md`.

5. **Keep `log.md` current.** Append every decision, blocker, falsified hypothesis, redirection, finding. Date every entry (absolute, e.g. `2026-05-20`).

## Execution order

1. **Phase 0 — Pre-flight gate.** Reproduce matched-protocol numbers in `T6_4_dual_selector_final.json`. F1 code change lands; re-run shipping FL ep=50 single-seed=42 n=5 under new code; confirm per-task disjoint reg = 76.12 ± 0.33 (within σ). Validates the F1 fix is bit-equivalent on the existing diagnostic.

2. **Phase 1 — F1 fix across 5 states.** Apply F1 selector fix; re-evaluate shipping at AL/AZ/FL/CA/TX single-seed=42 n=5. Produce three-frontier table. Quantify per-state selector-loss-pp.

3. **Phase 2 — Tier 5/6 candidate re-evaluation under F1.** Apply F1 selector to T5.2b, T5.3, T6.2 existing JSONs (zero retraining via `analyze_t64_selectors.py`). Check whether any sub-Bonferroni positive flips to ≥ Bonferroni under the new selector.

4. **Phase 3 — Residual-gap characterisation.** Quantify MTL-vs-STL reg gap that survives F1 per state. This brief determines the next study (`substrate_adaptive_mtl_balancing.md`, `mtl_architecture_revisit.md`, etc.).

5. **Final advisor pass.** Document. Synthesize. Propose the new shipping recipe (canonical + v3c + T3.2 ResLN + F1 selector) and the brief for the next-tier study.

## What you compare against

| Stage | Baseline | Protocol |
|---|---|---|
| Phase 0 | `T6_4_dual_selector_final.json` shipping arm | FL ep=50 single-seed=42 n=5 |
| Phase 1 | RESULTS_TABLE §0.1 single-seed=42 (existing run dirs) | All 5 states ep=50 single-seed=42 n=5 |
| Phase 2 | canonical_improvement Tier 5/6 §Discussion candidates | Their existing per-epoch val CSVs |
| Phase 3 | STL matched-head per-state | RESULTS_TABLE §0.1 STL ceiling row |

## How to format a Results section update in INDEX.html

When a Phase completes, find its `<div class="results-placeholder">Results: <em>pending execution</em></div>` block and replace it with a real results block. Template:

```html
<div class="block">
<div class="block-title">Results (YYYY-MM-DD)</div>
<table>
<thead><tr><th>State</th><th>selector</th><th>cat F1</th><th>reg Acc@10</th><th>selected ep</th><th>Δ vs production</th></tr></thead>
<tbody>
<!-- per state × 3 selectors rows -->
</tbody>
</table>
<p><strong>Verdict:</strong> <span class="pill good">FIX VALIDATED</span> | <span class="pill warn">partial</span> | <span class="pill bad">REGRESSION</span> — one-sentence reason.</p>
<p class="footnote">JSON: <code>results/mtl_protocol_fix/phase{N}_*.json</code>.</p>
</div>
```

Then update `log.md` with the dated findings block and `TaskUpdate` the corresponding task to `completed`.

## Result file conventions

- Per-state per-selector JSONs → `docs/results/mtl_protocol_fix/phase{N}_{state}_{selector}.json`
- Three-frontier comparison tables → `docs/results/mtl_protocol_fix/phase{N}_three_frontier.{json,md}`
- Plots → `docs/studies/archive/mtl-protocol-fix/figs/`
- Trained engines: **NONE** — this study is zero-retraining by design. If a new training run is required, it must be a Phase 0 reproduction or a Phase 3 multi-seed extension, justified in `log.md`.

## When you're done with this session

If you finish a Phase, mark its tasks completed, update `log.md`, and either:
- Continue to the next Phase in the same session if compute allows, OR
- Hand off cleanly: ensure `log.md` ends with a `**Next**:` bullet specifying the exact next experiment ID and any pending decisions.

If you're stopping mid-experiment: ensure the task is still `in_progress` (NOT `completed`), add a `**Blocker**:` line in `log.md` describing exactly what's incomplete and what the next agent needs to resume.

---

## Quick-reference: cited file paths

```
docs/studies/archive/mtl-protocol-fix/
  INDEX.html              # design + index + future results
  log.md                  # progress log (append-only)
  AGENT_PROMPT.md         # this file
  considerations.md       # user's questions (the four Qs)

docs/studies/archive/canonical_improvement/
  log.md                  # predecessor closure log (2026-05-19 final entry)
  INDEX.html              # 6-tier slate + Tier-6 closure block

docs/studies/mtl-exploration/
  FUTUREWORK_substrate_aware_mtl_balancing.md  # predecessor F1/F2/F3 memo

docs/future_works/
  paper_canon_reevaluation.md            # deferred §0.1 re-eval
  substrate_adaptive_mtl_balancing.md    # F2 loss-balancing
  mtl_architecture_revisit.md            # next-tier study
  head_window_batch_audit.md             # audit cluster
  reg_head_architecture_sweep.md         # head-sweep variant

docs/results/canonical_improvement/
  T6_4_dual_selector_final.{json,md}     # matched-protocol source-of-truth

docs/CONCERNS.md C21                     # selector-bug diagnosis
docs/CLAIMS_AND_HYPOTHESES.md CH23-A/B   # paper-facing claim status
docs/NORTH_STAR.md                       # selector-limitation banner

src/
  training/runners/mtl_cv.py             # line 679 (bug); line 710 (alternative)

scripts/
  canonical_improvement/analyze_t64_selectors.py  # zero-retraining re-eval tool
```

---

**Final reminder:** the design is a strong starting point, not a cage. If Phase 1 reveals that F1 selector fix DOES NOT lift reg at AL/AZ/CA/TX (only at FL), the substrate-Δ vs HGI may be selector-invariant — that would be a legitimate finding that re-opens the substrate axis (specifically the merge-design Levers 4/5/6). Document everything in `log.md`.
