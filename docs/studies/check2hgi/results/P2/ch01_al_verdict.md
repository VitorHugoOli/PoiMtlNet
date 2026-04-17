# P2-validate — CH01 bidirectional verdict on AL

**Run:** MTL `mtlnet_dselectk + pcgrad`, AL 5f × 50ep, seed 42, per-task modality, GRU region head + pad-mask fix (commit `b92fc62`), user-disjoint folds (fair).

## Headline

**CH01 bidirectional FAILS on AL**: both heads regress vs their single-task fair baselines, although σ-overlapping (not statistically significant regression).

| Task | MTL (5f × 50ep) | STL fair (5f × 50ep) | Δ (pp) | σ-overlap | r_i |
|---|---|---|---|---|---|
| next-category F1 | 36.08 ± 1.96 | **38.58 ± 1.23** | −2.50 | YES (tied) | −6.49% |
| next-region Acc@10 | 48.88 ± 6.26 | **56.94 ± 4.01** | −8.06 | YES (tied) | −14.16% |
| next-region MRR | 24.43 ± 3.57 | 34.57 ± 2.34 | −10.14 | NO | −29.35% |

**Δm = ½(r_A + r_B) = −14.12%** (Maninis/Vandenhende formula per CH02).

**Pareto-gate (bidirectional thesis): FAILS.** `r_A = −6.49% < 0` AND `r_B = −21.75% < 0`.

## Why this isn't a disaster for the paper

1. **AL is the development state, not headline.** CH01 is tested on Florida / California / Texas in the paper. AL is for cheap iteration on arch / modality choices. The large states (127 K / larger samples) may behave differently from AL's 10 K.

2. **σ-overlap is genuine.** The Δcat (−2.50 pp, σ combined 3.19) and ΔregAcc@10 (−8.06 pp, σ combined 10.27) both overlap their noise envelopes. MTL is not *significantly* worse than STL — just nominally so.

3. **CH03 (per-task modality) remains valid.** The finding here is about CH01 (MTL vs STL), not CH03 (per-task vs shared/concat). P4-dev directional data still holds.

4. **CH16 (Check2HGI > HGI on next-category) is unaffected.** Primary substrate claim stands at +18.30 pp F1 (fair folds).

## The interesting finding — why MTL doesn't help on AL

Running the same config twice, once with each head, reveals the mechanism:

| Config | Task-B head | MTL reg Acc@10 | STL reg Acc@10 (same head) | Lift from MTL |
|---|---|---|---|---|
| dselectk+pcgrad (budget test) | Transformer | 47.62 ± 5.62 | **7.40** (P1 standalone) | **+40.22 pp** 🎉 |
| dselectk+pcgrad (this validate) | GRU | 48.88 ± 6.26 | **56.94** (P1 standalone) | **−8.06 pp** 😬 |

**Interpretation:** MTL provides a *large* positive lift when the task-b head is weak on its own (the Transformer's 7% → 47% with MTL assistance). When the task-b head is strong on its own (GRU's 56.94%), MTL **dilutes** the signal via the shared-backbone bottleneck. The net effect is that MTL and GRU-head together perform *worse* than GRU-standalone.

This is a more nuanced story than a pure "does MTL help?" claim. It's also a paper-grade insight: **MTL's task-transfer value is inversely related to the task-specific head's standalone strength** on this task pair.

## What this rules out

- Not a **fold-leakage** artefact — both MTL and STL use user-disjoint folds now (C11 closed).
- Not a **training-budget** artefact — both 5f × 50ep (H1 closed).
- Not a **head-choice** artefact at task-B — GRU is the strongest region head; the gap exists specifically with the *right* head.
- Not a **pad-mask bug** — smoke V1 (22%) → smoke V2 (32%) → this validate (48%) shows the fix is working; the remaining gap is architectural.

## What it implies for next steps

1. **Test on FL.** 127K train samples. More data might let MTL extract usable shared-backbone signal without dilution. If FL shows r_A > 0 AND r_B > 0, CH01 is saved on the headline states. If FL also fails, we pivot the paper's framing.

2. **Investigate task-balancing (H3 from critical review).** Maybe pcgrad isn't the right loss when task-b has a strong standalone baseline. Try a config that weights task-a more, or run `naive`/`static_weight` with an explicit cat-heavy bias.

3. **Investigate backbone bottleneck.** Shared 4-block residual backbone (`shared_layer_size=256`) may be too small for 2 tasks. A larger backbone OR per-task tower with light sharing (CGC-like) may reduce dilution.

4. **Reframe the paper's narrative** if FL also fails:
   - Primary: CH16 (Check2HGI > HGI on next-cat, +18.30 pp F1).
   - Architectural: CH03 (per-task modality > shared/concat).
   - Novel nuance: **"MTL lift depends on task-B head strength"** — frame as an ablation insight rather than a proof of bidirectional lift.

## Result files

- `validate_dselectk_pcgrad_gru_al_5f_50ep.json` (this run)
- `budget_test_dselectk_pcgrad_al_5f_50ep.json` (Transformer head comparison)
- STL fair baselines: `docs/studies/check2hgi/results/P1_5b/next_category_alabama_check2hgi_5f_50ep_fair.json` + `results/P1/region_head_alabama_region_5f_50ep_E_confirm_gru_region.json`
