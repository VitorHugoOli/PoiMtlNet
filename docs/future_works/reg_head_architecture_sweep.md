# Future Work — Reg-head architecture sweep (§4.3)

**Date drafted:** 2026-05-20
**Updated:** 2026-05-28 — partial status:
- The **`log_T as supervisory signal (KD)`** sub-track from the table below was **PROMOTED** by `mtl-protocol-fix` Phase 3 Rank 1 ([`phase3_rank1_findings.md`](../results/mtl_protocol_fix/phase3_rank1_findings.md)): all 9 cells Wilcoxon-strict at p=0.0312, +2.40 / +5.06 / +2.32 pp on disjoint reg at AL/AZ/FL @ W=0.2. **Multi-seed n=20 promotion** moved to [`docs/studies/substrate-protocol-cleanup/`](../studies/substrate-protocol-cleanup/) Tier A as a paper-grade follow-up at small states.
- The **other 5-6 head sweep candidates** (next_lstm, next_transformer_pf, next_getnext, next_stan_baseline, next_gru-as-reg) are absorbed into [`docs/studies/mtl_improvement/`](../studies/mtl_improvement/) T7 (head re-ablation under the new backbone). This memo is the narrower precursor; once T7 lands it can be archived.

**Source:** [`docs/studies/canonical_improvement/`](../studies/canonical_improvement/) post-closure §4 alternatives (item §4.3)
**Sequencing:** deferred; this is a narrower variant of [`head_window_batch_audit.md`](head_window_batch_audit.md) §A (head re-design). If the broader head audit is launched, this rolls into it and the file can be archived.

## What's deferred

A focused, paper-grade reg-head architecture sweep under per-fold log_T (leak-free post-2026-05-15). The current reg head (`next_stan_flow` / `next_getnext_hard`) was chosen empirically in the P1 ablation BEFORE the leak fix. It has never been compared head-to-head with alternatives under the current leak-free protocol.

Candidates:

| Head | Source | Why try |
|---|---|---|
| `next_stan_flow` (current) | `src/models/heads/next.py` | Baseline |
| `next_getnext` (no hard-neg) | Same registry | Tests whether hard-neg sampling is load-bearing on reg |
| `next_lstm` | Same registry | Sequence baseline without attention |
| `next_transformer_pf` | Same registry | Faithful transformer with positional encoding |
| `next_stan_baseline` | Same registry | STAN without the `α · log_T[last_region_idx]` flow term |
| `next_gru` | Same registry | Currently the cat head; what if it's also the reg best? |
| **`next_*` × log_T as feature instead of as anchor** | New | Inject per-fold log_T as a learned-weight feature, not as a fixed `α` anchor |

## Why deferred

1. The current §0.1 reg numbers are reported under `next_stan_flow` + `α · log_T` anchor; if F1 selector fix already closes the MTL-vs-STL reg gap, the head choice is downstream polish.
2. ~5 GPU-h sweep is small but produces a 1-axis table — better delivered as part of the broader [`head_window_batch_audit.md`](head_window_batch_audit.md) so the cat-head ablation rides on the same compute.

## Acceptance criterion

When picked up:

1. 5-7 candidate reg heads run under per-fold log_T at FL × 3 seeds × 5 folds in MTL B9 + per-task disjoint best.
2. AL/AZ cross-state validation on the top-2 candidates from FL.
3. Three-frontier evaluation (MTL @ best joint, MTL @ best disjoint, STL ceiling).
4. Statistical promotion: paired Wilcoxon p ≤ 0.05 on reg at FL n=15+, no Δ_cat regression > 0.5 pp.

## Cost (estimated)

- FL × 5-7 heads × 3 seeds × 5 folds: ~10-15 GPU-h.
- AL/AZ top-2 × 5 seeds: ~6-8 GPU-h.
- **Total: ~15-25 GPU-h** (~3-4 days calendar).

## Live docs the work would touch

- `docs/results/P1/` — new head-sweep JSONs
- `docs/results/RESULTS_TABLE.md §0.4` — recipe-selection table may need a head-choice column
- `docs/NORTH_STAR.md` — possible recipe update
- `docs/CLAIMS_AND_HYPOTHESES.md` CH06 — "Champion MTL architecture" claim — possible refresh

## Pointers

- Head registry: `src/models/heads/next.py` + `src/models/next/`
- Existing P1 ablation script: `scripts/p1_region_head_ablation.py` (the pre-fix T1 JSONs are pessimistic; reread with care)
- log_T loader: `scripts/compute_region_transition.py`
- This memo's predecessor scope: [`head_window_batch_audit.md`](head_window_batch_audit.md) §A (broader audit including cat heads + window + batch balance)
