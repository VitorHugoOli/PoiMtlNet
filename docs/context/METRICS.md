# Evaluation Metrics & Significance Testing

Reference for the metrics + statistical-testing protocol used across the check2hgi study and the BRACIS paper. New agents should read this before claiming "X beats Y" in any analysis.

## Metrics by task

### Category task (`next_category`)

7-class classification (Community, Entertainment, Food, Nightlife, Outdoors, Shopping, Travel).

| Metric | Definition | Use |
|---|---|---|
| **Macro-F1** | Unweighted mean of per-class F1 | **Primary paper metric** for cat. Insensitive to class imbalance. |
| Accuracy | (TP + TN) / total | Secondary (sanity check). Inflated by majority-class baseline; do NOT cite as headline. |
| Per-class precision/recall/F1 | Standard | For class-imbalance audits + supplementary tables. |

### Region task (`next_region`)

K-class classification where K is state-dependent (1,109 regions at AL → 8,501 at CA — see `DATASETS.md`).

| Metric | Definition | Use |
|---|---|---|
| **Acc@10** | Fraction of test rows where the true region is in the top-10 predictions | **Primary paper metric** for reg. Headline 5-state architectural-Δ table uses this. |
| Acc@5 | Top-5 variant | Secondary (FL Markov-saturation analysis). |
| **MRR** | Mean reciprocal rank of the true region | Primary metric for joint Δm scoreboard. Captures "MTL produces better-ranked predictions even where raw top-10 is worse" (the FL Pareto-positive cell). |

**`top10_acc_indist` vs `top10_acc`**: in-distribution variant excludes test rows whose region was unseen in training. Used for MTL B9 (matched protocol with STL ceiling). See `src/training/runners/mtl_eval.py`.

### Joint Δm scoreboard

`Δm = (cat F1 lift) + (reg metric lift)`, normalized. Two variants reported:

- **PRIMARY**: `Δm-MRR = cat F1 % change + reg MRR % change`
- **SECONDARY**: `Δm-Acc@10 = cat F1 % change + reg Acc@10 % change`

The MRR-vs-Acc@10 split at FL is paper-grade significant in both directions (Δm-MRR positive, Δm-Acc@10 negative) — see `results/RESULTS_TABLE.md §0.2`.

## Significance testing

### Paired Wilcoxon signed-rank

The standard test for paired comparisons (e.g., MTL vs STL on the same fold-and-seed pairs). Used throughout the paper.

**Sample sizes and what they mean:**

| n_pairs | Floor p-value (two-sided) | Status |
|---:|---:|---|
| 5 (single-seed × 5 folds) | **0.0625** | Below paper-grade. n=5 ceiling: best achievable is **p = 0.0312** (one-sided) when all 5 pairs share the same sign. Use for screening, NOT for headline claims. |
| 20 (4 seeds × 5 folds) | < 1e-04 | **Paper-grade**. The headline architectural-Δ table (RESULTS_TABLE §0.1) uses n=20 for all 5 states. |
| 25 (5 seeds × 5 folds) | < 1e-07 | Used for FL multi-seed Δm-MRR (paper-grade Pareto-positive cell). |

**Why the n=5 ceiling matters:** at n=5 with all 5 pairs sign-consistent, the strongest claim you can make is `p = 0.0312`. Any "MTL beats STL at p=0.0312, n=5" is **at the ceiling**, not a strong significance — it just means all 5 folds had the same sign. Multi-seed extensions (n=20) are required for paper-headline claims.

**Reference**: `scripts/analysis/*.py` (e.g., `arch_delta_wilcoxon.py`, `gap_fill_wilcoxon.py`, `f50_delta_m_leakfree.py`) implement the paired tests; outputs land in `results/paired_tests/*.json`.

### TOST (Two One-Sided T-tests)

Used for "statistical tie" claims (e.g., CH15 reframing: HGI ties Check2HGI on reg under matched-head STL `next_stan_flow`). Equivalence margin chosen per-comparison (typically 2 pp on Acc@10).

## F51 canonical extraction

The agreed-upon protocol for extracting per-fold metrics from training logs (settled in F51, 2026-04 multi-seed pass):

> **Per-fold metric = max value over `epoch ≥ 5`** (skips the unstable warmup epochs while not anchoring to a single ep).

Implemented in `scripts/finalize_phase3.py` and the analysis pipeline. The "for ep ≥ 5" qualifier appears throughout `RESULTS_TABLE.md`.

Pre-F51 protocols (best-val-F1 epoch selection, fixed-epoch extraction) are leak-confounded or noise-inflated and have been retired.

## Practical guidance

- **Headline claims need n ≥ 20** (multi-seed). The 4-seed × 5-fold pooling is the project standard.
- **n=5 results are screening**, not paper-grade. Acceptable for "is this direction worth pursuing?" but never for "MTL beats STL".
- **Paired tests over independent t-tests**: the pairing structure (same fold-and-seed across MTL and STL) gives much higher power than treating them as two independent samples.
- **Sign-consistency is informative even at small n**: "5/5 fold-pairs positive" carries qualitative weight even at the ceiling p=0.0312.
- **TOST for ties**: never claim "no difference" from a high p-value alone. Use TOST with an explicit equivalence margin.

## Where the numbers are reported

- **Canonical paper table**: `docs/results/RESULTS_TABLE.md §0` (v11, 2026-05-02)
- **Wilcoxon JSONs**: `docs/results/paired_tests/*.json`
- **F-trail per-experiment metric reads**: `docs/findings/*.md`
- **Paper LaTeX**: `articles/[BRACIS]_Beyond_Cross_Task/src/sections/results.tex`

## See also

- `TASKS.md` — task definitions (what's being predicted)
- `DATA_SPLITS.md` — how the fold protocol generates the pairs Wilcoxon operates on
- `MTL_OPTIMIZERS.md` — the optimization side (how MTL combines per-task losses)
- `../findings/F50_DELTA_M_FINDINGS_LEAKFREE.md` — the load-bearing example of multi-seed paired Wilcoxon at n=25
