# Phase P7 — Dual-stream input & cross-attention (Options A + C)

**Gates:** P6 complete with `CH02 = confirmed` or `partial`. If P6 refutes CH02, **skip P7** — the paper pivots.

**Purpose:** Test whether explicit region-embedding input (Option A) and/or bidirectional cross-attention between streams (Option C) improves the headline 2-task MTL. Decision-gated: C only runs if A shows measurable lift.

## Context

From `CRITICAL_REVIEW.md §3` and the probe results:

- Alabama: check-in emb carries ~3× majority region signal (7.9% vs 2.3%).
- Florida: linear probe barely beats majority (23.5% vs 22.5%) — but massive class imbalance makes the LR probe unreliable. Transformer sequence model will likely do better.

**Hypothesis:** explicit region embeddings as a parallel input stream give the MTL model richer region-aware representation without requiring it to recover region info from check-in emb. Especially helpful on Florida-scale (thousands of regions).

Full architectural details in `OPTION_C_SPEC.md`.

## Structure

### P7a — Option A (dual-stream concat): *run first*

**Architecture change:** None (still `MTLnet` + champion from P5/P6). Input change only.

**Input:**
- Legacy: `x [B, 9, 64]` (check-in embeddings only).
- P7a: `x_dual [B, 9, 128]` (concatenate check-in + region embedding per timestep).

**Implementation:**
- New `src/data/inputs/next_region_dual.py` builds a parquet with doubled emb width.
- For each of the 9 timesteps in the window, look up `poi → region_emb` and concat to the check-in emb.
- Encoder first-layer Linear picks up the new width automatically.

**Experiments:**
- FL + AL, 5f × 50ep, same champion backbone + heads + optimiser from P5/P6.
- Compare: vanilla vs dual-stream on both `val_f1_next_category` and `val_acc1_next_region`.

**Decision gate → P7b:**
- If Option A improves `val_acc1_next_region` by ≥ 2 p.p. on FL: proceed to P7b.
- If improvement < 2 p.p.: **skip P7b**. Option A becomes a paper ablation row ("we also tried dual-stream input; no significant lift"), and Option C is documented as future work.

### P7b — Option C (cross-attention): *run only if P7a succeeds*

**Architecture change:** New `MTLnetCrossAttn` model class. See `OPTION_C_SPEC.md §2` for the block diagram.

Key design points (from the spec):
- 2 heads preserved (`next_category` on `h_c`, `next_region` on `h_r`).
- Loss: same `NashMTL` over two CEs.
- K = 2 cross-attention layers by default; K = 1 and K = 3 as small ablation sweep.
- Separate class — legacy `MTLnet` untouched, regression floors pinned.

**Experiments:**
- FL + AL, 5f × 50ep.
- K ∈ {1, 2, 3}.
- Compare to (a) vanilla P3 baseline and (b) P7a result.

**Risks (from OPTION_C_SPEC §7):**
- ~1.3× parameter budget; AL (12k sequences) may overfit. Monitor val/train gap.
- Cross-attention may collapse to trivial patterns on small data. Plan warm-start from P5/P6 champion weights if unstable.

## Claims touched

Add to `CLAIMS_AND_HYPOTHESES.md` after this phase doc lands:

- **CH12** (existing in review) — Region embeddings as input improve next-region Acc@1 over check-in-only input. *Tested by P7a.*
- **CH13** (new) — Bidirectional cross-attention between check-in and region streams improves joint_acc1 over dual-stream concat at equal parameter budget. *Tested by P7b.*
- **CH20** (new, paper-critical) — Any region-input gain is state-dependent: larger on Florida (4703 regions) than Alabama (1109 regions). *Requires both states.*

CH20 is the empirically interesting claim — it tells readers when our design choice matters and when it doesn't. Paper-valuable even if CH12/CH13 are partial.

## Budget

P7a: ~2h implementation + 4 runs × 22 min = ~3h total (AL + FL).
P7b: ~1 day implementation + 8 runs (K × state × vanilla-comparison) × 22 min = ~6h training.

**Total P7:** ~1.5 days if both A and B run; ~half-day if A alone.

## Outputs

- `docs/studies/check2hgi/results/P7/` — 4 runs for P7a + up to 8 for P7b.
- `results/P7/SUMMARY.md` — comparison table: vanilla / Option A / Option C (if run) per state.
- `results/P7/fig_region_input_gain.png` — bar chart of Δ Acc@1 vs vanilla per state.

## Paper table (headline for this section)

| Input setup | Architecture | AL `val_acc1_region` | FL `val_acc1_region` | AL `val_f1_category` | FL `val_f1_category` |
|---|---|---|---|---|---|
| Check-in only (vanilla) | MTLnet | — | — | — | — |
| Option A: dual-stream concat | MTLnet (wider input) | — | — | — | — |
| **Option C: cross-attention (K=2)** | MTLnetCrossAttn | — | — | — | — |

Pareto-optimal configuration bolded. Bootstrap 95% CI per cell.
