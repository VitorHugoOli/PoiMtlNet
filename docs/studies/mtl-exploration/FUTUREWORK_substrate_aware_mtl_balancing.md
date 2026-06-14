# Future Work — Substrate-aware MTL balancing for the canonical Check2HGI MTL recipe

> ✅ **PARTIALLY RESOLVED 2026-06-03 — the selector half of this memo is DONE.** The "structurally
> broken production selector" described below was **fixed and made the code default**: the joint
> checkpoint selector is now `joint_geom_simple = sqrt(cat_macroF1 · reg_Acc@10)`
> (`--checkpoint-selector geom_simple`; v11 repro = `joint_f1_mean`). See `docs/CONCERNS.md §C21`,
> `docs/results/CANONICAL_VERSIONS.md §selector`. The `mtl_cv.py:679/:710` line refs below are
> HISTORICAL (code has moved). The **still-open** part is the *loss-balancing* study, now
> consolidated with literature ranking in
> [`../../future_works/joint_selection_and_loss_combination.md`](../../future_works/joint_selection_and_loss_combination.md) Part 2.

**Status: ~~URGENT~~ (selector RESOLVED 2026-06-03; loss-balancing still open) — the production B9 joint selector is structurally broken on the canonical shipping substrate itself.** Not vague future work — matched-protocol analysis (2026-05-19) demonstrated that the canonical Check2HGI shipping recipe loses ~10.7 pp of reg-top10 capacity to the production selector at FL, with no substrate change involved. Every Check2HGI MTL number in the published canon (`RESULTS_TABLE.md §0`) is reported at this destabilised joint-best epoch. The next study under `mtl-exploration` should pick this up as its primary track.

**Date:** 2026-05-19 (created); rewritten same day with matched-protocol findings after the initial substrate-focused interpretation was falsified.
**Source study:** [`docs/studies/archive/canonical_improvement/`](../archive/canonical_improvement/) — Tier 6 / T6.4 closure log entry (2026-05-19 CORRECTION).
**Trigger:** T6.4 substrate hypothesis was falsified at matched protocol (Δ_reg = +0.08-0.17 pp over shipping, within σ). The dual-selector analysis intended to characterise T6.4's substrate effect instead exposed a load-bearing selector bug in the **shipping recipe**.

---

## Problem statement

The production B9 joint selector (`joint_score = 0.5 * (cat_macro_f1 + reg_macro_f1)` at `src/training/runners/mtl_cv.py:679`) is **structurally broken on the canonical Check2HGI MTL recipe itself**. Matched-protocol analysis (FL ep=50 single-seed=42 n=5, no substrate changes) shows the canonical Check2HGI substrate reaches reg top10 ≈ 76 % at the reg-best epoch (~4) but the production selector picks ep ~29 where reg has destabilised to ~65 % (σ ≈ 9 across folds). **The production selector throws away ~10.7 pp of reg-top10 capacity that the canonical substrate already produces, with no substrate change involved.**

### Evidence — canonical shipping FL ep=50 single-seed=42 n=5 (NO substrate changes)

| Selector | cat F1 | reg top10 | selected ep |
|---|---:|---:|---:|
| Per-task disjoint best | 70.49 ± 0.86 | **76.12 ± 0.33** | cat 35, reg 4 |
| `joint_geom_simple` = `sqrt(cat_f1 * reg_top10_indist)` | 67.93 ± 1.74 | 72.38 ± 2.20 | 14.0 ± 8.5 |
| `joint_canonical_b9` (production) | 69.99 ± 1.13 | **65.38 ± 9.10** ← σ huge | 29.2 ± 10.8 |

Cross-check: shipping FL §0.1 multi-seed n=20 reports reg top10 = 63.27 ± 0.10. The matched-protocol single-seed `joint_canonical_b9` value (65.38 ± 9.10) is consistent with §0.1 within single-seed variance. **§0.1 reports joint-best, not reg-best.** The "official" paper reg number under-represents the substrate's reg capacity by ~10 pp.

### Why the production selector fails

`reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise and stays ~16-18 % across the entire ep=1-50 trajectory. It is blind to `reg_top10_acc_indist`'s peak-and-collapse pattern (top10 peaks at ep ~4 around 76 %, then destabilises through ep 15-50 to ~65 %). The mean-of-F1s formula at `mtl_cv.py:679` is **scale-incoherent** when one head has 7 well-supported classes (cat_macro_f1 ≈ 0.70) and the other has 4 700 sparse classes (reg_macro_f1 ≈ 0.17): cat's F1 dominates the sum, and the selector tracks cat's plateau-to-late-epoch dynamics while ignoring reg_top10's much-earlier peak.

### Concrete trajectory (canonical shipping FL fold 1)

| ep | cat_f1 | reg_top10 | reg_macro_f1 | joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1) |
|---|---:|---:|---:|---:|
| 4 | ~62 | **~76 ← reg peak** | ~0.17 | ~0.40 |
| 7 | ~66 | ~73 | ~0.17 | ~0.42 |
| 14 | ~68 | ~70 | ~0.17 | ~0.42 (joint_geom_simple peak) |
| 25 | ~70 | ~65 | ~0.17 | ~0.43 |
| 29 | ~70 | **~65 ← joint_canonical_b9 picks here** | ~0.17 | ~0.43 |
| 50 | ~70 | ~55 | ~0.17 | ~0.43 |

The selector's chosen "best" epoch is ~25-50 because `cat_f1` is monotonically rising and `reg_macro_f1` is flat (it cannot see reg_top10's collapse). By the time the selector locks in, reg-top10 has lost ~11 pp from its peak.

### T6.4 substrate variants — FALSIFIED at matched protocol

The earlier canonical_improvement Tier-6 / T6.4 attempt (this memo's original 2026-05-19 framing) claimed +11-13 pp reg lift from T6.4 substrate variants. **That was a cross-selector comparison artefact**: T6.4 reg-best epoch was compared against shipping §0.1 joint-best multi-seed numbers — different selectors, not different substrates. At matched protocol, T6.4 variants add Δ_reg = +0.08-0.17 pp over shipping at per-task disjoint best — well within fold σ. **No measurable substrate improvement.** T6.4 closed as falsified; the InfoNCE-and-two-pass code paths in `Check2HGIModule.py` are opt-in default-off infrastructure available to future studies, but the variants alone are §Discussion-only.

The load-bearing finding is the selector bug in shipping itself — described above.

## Future-work proposals (for the mtl-exploration track)

These are out of scope for the canonical_improvement track but are exactly the kind of investigation the mtl-exploration track should pick up.

### F1. Substrate-aware checkpoint selection

The joint_score should weight reg dynamics in a way that catches top10 trajectory. Options:

- **Replace reg macro-F1 with reg top10_acc_indist in joint_score.** Cleanest fix. One-line change in `src/training/runners/mtl_cv.py:679`. Risk: cat_f1 is on a [0, 1] macro-F1 scale; reg top10 is on a [0, 1] accuracy scale. Same range — combining is natural. The reg_f1 used today is on a [0, 0.2] effective range due to 4703 sparse classes, so it under-weights reg in joint.
- **Scale-coherent geometric joint** (already partially implemented as `joint_geom_lift` at line 710): the geometric mean of per-head lifts over majority baseline. Documented as "the checkpoint monitor for the check2HGI track" in the comment, but not actually used as the primary selector in the canonical B9 recipe. Audit whether wiring this in changes the selected epoch.
- **Per-task best-epoch checkpoints with shared-backbone joint-best**: track per-task best states independently; ship the model with cat head from cat-best, reg head from reg-best, shared backbone from joint-best (or weighted blend). Architectural refactor, larger lift.

### F2. Substrate-adaptive MTL loss balancing

NashMTL or GradNorm with adaptive task weights would let the reg head update less aggressively as cat plateaus, preventing the reg destabilisation observed at ep 20+. Both losses already live in `src/losses/` (NashMTL was the pre-B9 canonical) — the question is whether they restore reg stability under improved substrates.

- **NashMTL revival**: it was demoted from canonical because of cvxpy solver instability on the small-state recipe. Could test it on **FL only** (large state, well-conditioned) where the solver issue rarely fires. If NashMTL stabilises reg through ep 50 while keeping cat lifts, it becomes the substrate-aware MTL recipe for improved encoders.
- **Per-task LR scheduling**: drop reg_lr aggressively after ep ~10 once reg peaks, while keeping cat_lr cosine to ep 50. Lightweight — just a per-task LR schedule override flag.
- **Gradient masking**: detect reg-head plateau via val loss and freeze the shared backbone's reg-side gradients afterward. Heavier — needs gradient-flow instrumentation.

### F3. Substrate-protocol coupling experiment

The cleanest scientific experiment: take **canonical Check2HGI (shipping)** and **T6.4 substrate (e.g. infonce τ=0.5)**, run BOTH under:

- **Canonical B9 (ep=50, joint_best on macro F1)** — current
- **Short-train (ep=15, joint_best on macro F1)** — protocol-fit-to-substrate
- **NashMTL or geometric-lift (ep=50)** — substrate-aware MTL balancing

Goal: untangle "substrate effect" from "protocol effect". If the canonical shipping benefits from ep=15 or NashMTL, the protocol fix is universal. If only the T6.4 substrate benefits, the protocol fix is substrate-specific (and we ship them together).

## What canonical_improvement did (and what it can no longer claim)

**Original attempt (rejected):** run the two T6.4 FL variants at `--epochs 15` to terminate before reg collapse. Advisor consult #1 (2026-05-19) attacked this as val-leak (the cap was chosen post-hoc from val trajectories). **Decision: abandoned the ep=15 cap.**

**Current frame (locked 2026-05-19):** train full ep=50 horizon, then report **three selectors** in parallel:

1. **Per-task disjoint best** — substrate-capacity diagnostic (cat from cat-best epoch, reg from reg-best epoch).
2. **`joint_geom_simple = sqrt(cat_f1 * reg_top10_indist)`** — principled single-checkpoint selector. Simpler than the `joint_geom_lift` already coded at `mtl_cv.py:710` (no majority-baseline normalisation) but preserves the head-collapse penalty.
3. **`joint_canonical_b9 = 0.5*(cat_f1 + reg_macro_f1)`** — current production selector, kept for reference / to show the protocol-specific gap.

Tool: `scripts/canonical_improvement/analyze_t64_selectors.py` (zero retraining; reads per-fold val CSVs).

### Concrete dual-selector numbers — matched-protocol (single-seed=42, n=5 folds, ep=50, FL)

All three arms at the same protocol. Source: `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}`.

| Selector | Arm | cat F1 | reg top10 | selected ep |
|---|---|---:|---:|---|
| **Per-task disjoint** | **shipping (no substrate change)** | 70.49 ± 0.86 | **76.12 ± 0.33** | cat 35, reg 4 |
| Per-task disjoint | T6.4 two_pass | 70.55 ± 0.85 | 76.20 ± 0.27 | cat 38, reg 5 |
| Per-task disjoint | T6.4 infonce τ=0.5 | 70.49 ± 0.95 | 76.29 ± 0.29 | cat 37, reg 4 |
| **joint_geom_simple** | **shipping** | 67.93 ± 1.74 | 72.38 ± 2.20 | 14 ± 8.5 |
| joint_geom_simple | T6.4 two_pass | 67.33 ± 2.06 | 73.33 ± 2.28 | 12 ± 9.5 |
| joint_geom_simple | T6.4 infonce τ=0.5 | 67.12 ± 2.45 | 73.48 ± 2.48 | 12 ± 9.6 |
| **joint_canonical_b9 (production)** | **shipping** | 69.99 ± 1.13 | **65.38 ± 9.10** | 29 ± 11 |
| joint_canonical_b9 | T6.4 two_pass | 70.13 ± 1.06 | 61.19 ± 11.86 | 30 ± 11 |
| joint_canonical_b9 | T6.4 infonce τ=0.5 | 70.28 ± 0.82 | 56.78 ± 11.79 | 32 ± 11 |

Cross-check: shipping FL §0.1 multi-seed n=20 reports cat 68.56 ± 0.79, reg top10 63.27 ± 0.10. The matched-protocol single-seed `joint_canonical_b9` values are consistent with §0.1 within single-seed variance.

**The protocol-axis gap is the load-bearing finding.** On the canonical shipping substrate alone (no substrate change), per-task disjoint best reaches reg top10 = 76.12 while joint_canonical_b9 reaches 65.38 — a ~10.7 pp capacity gap that exists in the published canon and is invisible to anyone reading §0.1.

**Substrate axis is null.** T6.4 vs shipping at per-task disjoint: Δ_cat = +0.00-0.06 pp, Δ_reg = +0.08-0.17 pp — well within fold σ (~0.3). T6.4 substrate adds no measurable capacity above canonical+v3c+T3.2.

### What canonical_improvement Tier 6 claims (locked 2026-05-19)

- **CH23-A — T6.4 substrate hypothesis FALSIFIED at matched protocol.** Δ_reg = +0.08-0.17 pp over shipping at per-task disjoint best (within σ at n=5). No T6.4 shipping promotion under any selector. The T6.4 code paths land as opt-in default-off infrastructure.
- **CH23-B — `joint_canonical_b9` selector throws away ~10.7 pp reg-top10 capacity from shipping itself.** Allowed paper claim: "Under per-task disjoint best (or a substrate-aware `joint_geom_simple` selector), the canonical Check2HGI substrate reaches reg top10 ≈ 76 % at FL — ~+13 pp above the §0.1 joint-best number. The production B9 joint selector (`0.5 * (cat_macro_f1 + reg_macro_f1)`) is structurally broken on this MTL setup because `reg_macro_f1` over ~4 700 sparse FL regions is dominated by rare-class noise."

### What canonical_improvement CANNOT claim

- ❌ "T6.4 ships a substrate improvement." (FALSIFIED.)
- ❌ "T6.4 enables cat +2 / reg +12 lift." (The cross-selector lift exists in shipping itself, not in T6.4 specifically.)
- ❌ Any reg-axis comparison drawn from §0.1 numbers as a statement about the substrate's reg capacity. §0.1 reports joint-best where reg has destabilised; substrate's reg-best reaches ~76 % at FL.

This memo and the mtl-exploration follow-on study are responsible for the F1/F2/F3 work that would deliver a single deployable model with both heads at substrate-capacity — for **shipping itself**, not just for substrate variants.

## Cross-references

- **Matched-protocol source-of-truth**: `docs/results/canonical_improvement/T6_4_dual_selector_final.{json,md}` (3 arms × 3 selectors at FL ep=50 single-seed=42 n=5).
- **Per-arm run dirs (per-epoch val CSVs for any future selector experiment, no retraining needed):**
  - shipping: `docs/results/canonical_improvement/shipping_florida_mtl_ep50_seed42/mtlnet_lr1.0e-04_bs2048_ep50_20260519_042801_960302/`
  - T6.4 two_pass: `docs/results/canonical_improvement/T6_4_two_pass/florida_mtl/mtlnet_lr1.0e-04_bs2048_ep50_20260519_022648_899018/`
  - T6.4 infonce τ=0.5: `docs/results/canonical_improvement/T6_4_infonce_tau0_5/florida_mtl/mtlnet_lr1.0e-04_bs2048_ep50_20260519_024447_907925/`
- **Trajectory CSVs**: `metrics/fold{1..5}_next_{category,region}_val.csv` inside each run dir.
- **Joint-score implementation** (the bug site): `src/training/runners/mtl_cv.py:679` — `joint_score = 0.5 * (f1_val_task_b + f1_val_task_a)`.
- **Geometric-lift alternative** (already coded but unused as the primary selector): `src/training/runners/mtl_cv.py:710` — `joint_geom_lift = sqrt(task_b_lift * task_a_lift)`.
- **Dual-selector analysis tool**: `scripts/canonical_improvement/analyze_t64_selectors.py` — zero retraining; any new selector can be re-evaluated against existing per-fold val CSVs.
- **Closure-of-Tier-6 log entry** (the CORRECTION supersedes the earlier same-day entries): `docs/studies/archive/canonical_improvement/log.md` (2026-05-19 entry titled "CORRECTION (supersedes the earlier 2026-05-19 entries above): T6.4 FALSIFIED at matched protocol; shipping selector is the actual bug").
- **Project-wide doc cross-references** (all updated 2026-05-19 with the matched-protocol framing):
  - `docs/CONCERNS.md` C21 — open concern (selector-bug-on-shipping).
  - `docs/CLAIMS_AND_HYPOTHESES.md` CH23-A (T6.4 falsified), CH23-B (selector bug as paper §Discussion-only).
  - `docs/AGENT_CONTEXT.md` — mandatory blocker callout for future agents.
  - `docs/NORTH_STAR.md` — selector-limitation flag above the B9 champion config.
  - `docs/CHANGELOG.md` — 2026-05-19 timeline entry.
  - `docs/studies/archive/canonical_improvement/INDEX.html` — T6.4 Results section (FALSIFIED verdict).
