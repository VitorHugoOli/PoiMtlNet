# analysis_protocol — the statistical record behind the reported numbers

This folder is the analysis record for *"Predicting the Next Category and Region of a Visit:
A Check-in-Level Multi-Task Study on Mobility Data"* (MobiWac 2026). It is what the paper's
Section V-C means by "the plan and this departure are released with the code".

Added 2026-07-25. Before that date this branch shipped the test scripts but not the plan, so the
pre-registration and its deviation log were not readable by a reviewer. That gap is what this folder closes.

## What is here

| File | What it is |
|---|---|
| `STATISTICAL_PROTOCOL.md` | **The pre-registration.** Committed 2026-06-21, before the first result cell (2026-06-22). Section 1 registers the two comparison families; Section 2 the superiority test (paired Wilcoxon on per-fold differences, n=20); Section 3.2 the equivalence margin (delta_reg = 2 pp); Section 5.2 the multiplicity families; Section 8 the deviation-log rule. |
| `DEVIATION_LOG.md` | **The deviation log the protocol's Section 8 mandates.** Entries D-1 to D-4 under the 2026-07-25 heading are the ones that bear on the paper. |
| `EXECUTED_ANALYSIS.md` | The executed analysis at the reported footing (per-seed means, n=4), with its own deviation log and its honest-gaps section. |
| `m2_prereg_output.txt` | Output of the registered test at its registered footing (per-fold, n=20), with a 24/24 artifact-to-board reproduction gate. |
| `JOINT_BEST_RESULTS.md` | The reported epoch-selection convention (one saved model per fold) and its parity gates against the alternative. |
| `JOINT_BEST_SCORING.md` | Both epoch-selection conventions defined precisely, and how not to mix them. |
| `CEILINGS_N20_FINAL.md` | The dedicated single-task ceilings the joint model is compared against, and how each was tuned. |
| `istanbul_cat_ceiling_perfold/` | The four per-fold arrays for Istanbul's dedicated category ceiling. Recovered 2026-07-25; these are what made the registered test runnable at all six datasets. |

## What the plan registered, and what departed from it

Read this before reading any p-value in the paper.

**Registered.** Superiority for next-category, non-inferiority for next-region, assigned **per task**
(not per dataset), with the two-point margin pinned in advance. The superiority test is a paired
Wilcoxon signed-rank test on the per-fold differences, n=20 = 4 seeds x 5 folds. Holm-Bonferroni is
applied within the six-dataset next-category set; the region equivalence cells are explicitly not
pooled into it.

**Two departures, both in `DEVIATION_LOG.md`.**

1. The reported tests pair **per-seed means (n=4)**, not per-fold values. At the time the per-fold
   arrays for the joint model were not in the committed tree.
2. Superiority is reported with a **paired t**, because at four pairs the exact one-sided Wilcoxon p
   cannot fall below 1/2^4 = 0.0625 regardless of effect size. The registered test is reported
   alongside it.

**One claim outside the plan.** The plan registered next-region as non-inferiority only. The four
next-region improvements (Istanbul, Florida, Texas, California) are therefore **secondary results**,
and the paper says so. They are reported in their own Holm family rather than uncorrected.

**The registered test agrees.** Since 2026-07-25 the registered per-fold n=20 Wilcoxon runs at all six
datasets. Every next-category cell rejects (Holm-adjusted p = 5.72e-06, m=6), each at the exact n=20
floor of 9.54e-07 with 20 of 20 folds favoring the joint model; the four next-region cells reject in
their own family (Holm-adjusted 3.81e-06, m=4). No verdict and no reported interval changed.

## Reproducing

The scripts read per-fold score files produced by the training runs of Section 5 of the top-level
`README.md`. Those files are not shipped (see "What is NOT included" there); the recipes regenerate them.

```bash
# the registered test at its registered footing: per-fold, n=20, Holm within the six-dataset
# next-category family (+ the four next-region cells as their own family)
python scripts/closing_data/m2_prereg_perfold.py

# the reported footing: per-seed means, n=4, paired t with the Wilcoxon alongside
python scripts/closing_data/m1_stats_n20.py

# equivalence: TOST non-inferiority at the two-point margin
python scripts/closing_data/region_match_tost.py
python analysis/tost_region.py

# the epoch-selection convention used for every reported joint result
python scripts/closing_data/score_joint_best.py <rundir> --seed <seed> --tag <tag>
```

`m2_prereg_perfold.py` aborts if any recomputed aggregate stops matching the reported cell, so a
silent drift between the artifacts and the paper's table is not possible.

## Known inconsistencies in the shipped scripts (2026-07-25)

Two docstrings and one results section describe next-region superiority as pre-registered. They are
wrong: no such family appears in `STATISTICAL_PROTOCOL.md`. The protocol is authoritative; the
docstrings are pending a correction pass.

- `scripts/closing_data/superiority_wilcoxon.py`, module docstring
- `scripts/closing_data/m1_stats_n20.py`, the label printed for the region cells
- `analysis_protocol/EXECUTED_ANALYSIS.md`, Section 1b
