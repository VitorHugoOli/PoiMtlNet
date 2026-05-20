# docs/future_works/ — Forward-looking notes

Notes that capture work-to-do-later. Not in `docs/studies/` (which holds *active* research tracks) and not in `docs/archive/` (which holds *closed* work). This is the staging ground for things deferred to future iterations.

## Convention

Each file is a self-contained memo with at minimum:
- **Date drafted**
- **What's deferred and why** (clear scope)
- **Concrete acceptance criterion** if it ever gets picked up (what does "done" look like?)
- **Pointers to live docs** the work would touch (NORTH_STAR.md, CLAIMS_AND_HYPOTHESES.md, articles/[BRACIS]_*, etc.)

When a future-work item is picked up:
- If small: do the work + delete the memo (or `git mv` to `docs/archive/future-works-done-YYYY-MM-DD/`).
- If large enough to be its own study: spawn a `docs/studies/<name>/` folder for it; remove the memo here.

## Current items

| Memo | What's deferred | Why deferred |
|---|---|---|
| [`task_pivot_memo.md`](task_pivot_memo.md) | Merge the task-pair pivot rationale (`{next_poi, next_region}` → `{next_category, next_region}`) into `docs/NORTH_STAR.md` Validation status section, OR into a `articles/[BRACIS]_*/src/sections/` paragraph. | The pivot itself is reflected in live docs (NORTH_STAR uses the new task pair); only the *historical why* is documented standalone here. Merging into NORTH_STAR is a polish item that needs careful wording for paper-facing reviewers. Deferred to next pass. |
| [`paper_canon_reevaluation.md`](paper_canon_reevaluation.md) | Re-running `RESULTS_TABLE.md §0.1` n=20 multi-seed numbers (all 5 states) under whichever selector + arch wins from the next two studies. | Pay the publication revision cost once — after `mtl-protocol-fix` and `mtl_architecture_revisit.md` both land. Sequenced AFTER `mtl_architecture_revisit.md`. |
| [`substrate_adaptive_mtl_balancing.md`](substrate_adaptive_mtl_balancing.md) | Leak-free re-evaluation of NashMTL revival, GradNorm, PCGrad, FAMO, Aligned-MTL, per-task LR decay, gradient-masking under per-fold log_T. | Scope depends on what `mtl-protocol-fix` reveals — if F1 closes the gap, this is polish; if F1 leaves >2 pp residual, this becomes load-bearing. |
| [`mtl_architecture_revisit.md`](mtl_architecture_revisit.md) | Rigorous, faithfully-implemented MMoE / CGC / DSelect-K / cross-stitch / hybrid (cross-stitch + cross-attn, MMoE + cross-attn) with per-task evaluation; supersedes the legacy 1-fold × 10-epoch fusion-study ablation. | Next-tier successor to `mtl-protocol-fix`. Includes the deferred [`paper_canon_reevaluation.md`](paper_canon_reevaluation.md). |
| [`head_window_batch_audit.md`](head_window_batch_audit.md) | Per-task head re-design under leak-free protocol; sequence-window + causal-mask correctness audit; batch class-balance (weighted-CE vs class-balanced sampler vs focal-loss) at FL. | Diagnostic / variance-source audits; co-schedule with `mtl_architecture_revisit.md` because the architecture × head × batch axes interact. |
| [`reg_head_architecture_sweep.md`](reg_head_architecture_sweep.md) | Focused reg-head sweep (`next_stan_flow` vs `next_getnext` vs `next_lstm` vs `next_transformer_pf` vs `next_stan_baseline` vs `next_gru`) under per-fold log_T. | Narrower variant of [`head_window_batch_audit.md`](head_window_batch_audit.md) §A; rolls into that broader audit if launched. |

## Sibling folders

- [`../studies/`](../studies/) — *active* research tracks (canonical_improvement, merge_design, hgi_category_injection)
- [`../archive/`](../archive/) — *closed* studies and snapshots (fusion-study, pre-promotion plans, paper-closure, etc.)
- [`../findings/`](../findings/) — paper-supporting per-experiment findings (read-only F-trail)
