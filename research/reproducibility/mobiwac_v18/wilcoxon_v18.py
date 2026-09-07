"""Paired one-sided Wilcoxon signed-rank on the 20 matched fold differences, computed on
the v18 / joint-best (served-checkpoint) arrays that Chapter 5 actually reports.

WHY THIS SCRIPT EXISTS
----------------------
The analysis plan pre-registered on 2026-06-21
(docs/reproducibility/mobiwac_v17/STATISTICAL_PROTOCOL.md §2) names a paired one-sided
Wilcoxon signed-rank on the matched per-fold deltas, n = 20 = 4 seeds x 5 folds, as the
PRIMARY superiority test for next category. It was executed once, on 2026-07-25, by
research/reproducibility/mobiwac_v17/m2_prereg_perfold.py -- against the v17 ladder.

On 2026-08-11 the ladder was recomputed under the served-checkpoint (joint-best)
convention. Category dropped from six wins to one; region from four to two. The sentence
that REPORTED the Wilcoxon was deleted from 5_mobiwac/06_results.tex in the same commit
wave (ce5a7006), but the sentence that PROMISES it lives in 05_setup.tex and was never
touched. The registered test was therefore never re-run at the footing the delivered
numbers use.

Three places in the delivered text assert that the two tests agree:
  src/chapters/2_fundamentals.tex:1777  "is reported alongside it and reaches the same decisions"
  GLOSSARY.md:112                        "is reported alongside it and agrees"
  wrapup/ESTUDOS_DEFESA.md:385           "os dois concordam"
None of those had been computed for this ladder. This script computes them.

WHAT CARRIES WHAT STANDING
--------------------------
- CATEGORY, all six datasets: superiority, joint > dedicated. PRE-REGISTERED (protocol
  §2 for the test, §5.2 for the Holm family "within the cat-superiority set (6 states)").
  This is the test the text promises.

- REGION: the protocol pins next region to NON-INFERIORITY only (§1 "reg -> TOST (§3)"),
  and no region-superiority family appears in it. The two region superiority claims
  (TX, CA) are post-hoc and carry their own Holm family m = 4, per deviation D-4
  (2026-07-25). The Wilcoxon is reported here for the region axis as a sensitivity
  reading, NOT as a registered test. The registered region cells are the TOST
  non-inferiority ones at delta_reg = 2 pp -- protocol §3.2, not this script.

- The delivered p-values in wrapup/evidence/ladder_recompute.json are a one-sided paired
  t on the FOUR per-seed means (df = 3), Holm-corrected. That is a different footing from
  this one (n = 4 seed means vs n = 20 folds). Both are reported side by side below so
  the "reaches the same decisions" claim can be evaluated rather than assumed.

FOOTING
-------
Paired per (seed, fold) on the same partition: for a given seed both arms read the same
StratifiedGroupKFold split, so fold k of seed s is matched across arms. Pairing across
SEEDS is not valid (each seed draws its own partition,
science/fold_partition_and_seeds.md) -- but the Wilcoxon here operates on the pooled set
of 20 matched differences, which is the footing the protocol registered and the footing
the promised sentence names.

Known limitation, stated because the protocol's own D-1 states it: the five folds inside
one seed share ~75-80% of their training data, so the 20 differences are not 20
independent replicates. The registered test uses this footing anyway; the seed-level t
(n = 4) is the more conservative reading and is the one the chapter reports.

INPUTS (both committed)
-----------------------
  joint:      docs/results/closing_data/v18/joint_best_perfold.json
              cells['<state>_s<seed>_joint'].folds['1'..'5'].joint_best
              cat = cat_f1 (fraction), reg = top10_full (fraction)
  dedicated:  docs/studies/closing_data/v18/data/v18_results.json
              per_run[] matched on (state, seed) -> stl_cat_folds, stl_reg_folds (percent)

The script refuses to report any test until it has reproduced the delivered per-cell
means from these arrays. A parse that returns the wrong mean is a broken instrument, and
in the output it is indistinguishable from a result (GUARDRAILS §4b V13).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[3]
JOINT = REPO / "docs/results/closing_data/v18/joint_best_perfold.json"
DED = REPO / "docs/studies/closing_data/v18/data/v18_results.json"

SEEDS = [0, 1, 7, 100]
STATES = ["istanbul", "alabama", "arizona", "florida", "texas", "california"]

# Delivered cell means, recomputed from the same arrays (mean of 5 folds inside a seed,
# then mean of the 4 seeds). Used as the instrument check, not as an input.
EXPECTED = {
    "istanbul": (35.4228, 75.0804),
    "alabama": (30.5868, 69.2430),
    "arizona": (34.5702, 59.0440),
    "florida": (37.5484, 76.5376),
    "texas": (36.1941, 66.1503),
    "california": (35.6287, 64.5362),
}
TOL = 0.005


def load():
    joint = json.loads(JOINT.read_text())["cells"]
    ded_runs = json.loads(DED.read_text())["per_run"]
    ded = {(r["state"], r["seed"]): r for r in ded_runs}
    return joint, ded


def series(joint, ded, state):
    """Return (cat_pairs, reg_pairs) as lists of (joint, dedicated) in percent,
    ordered seed-major then fold, 20 entries each."""
    cat, reg = [], []
    for seed in SEEDS:
        cell = joint[f"{state}_s{seed}_joint"]["folds"]
        run = ded[(state, seed)]
        for k in range(1, 6):
            jb = cell[str(k)]["joint_best"]
            cat.append((jb["cat_f1"] * 100.0, run["stl_cat_folds"][k - 1]))
            reg.append((jb["top10_full"] * 100.0, run["stl_reg_folds"][k - 1]))
    return cat, reg


def seed_means(pairs):
    """Mean of each arm within each seed -> 4 pairs."""
    out = []
    for i in range(0, 20, 5):
        blk = pairs[i : i + 5]
        out.append((sum(a for a, _ in blk) / 5.0, sum(b for _, b in blk) / 5.0))
    return out


def holm(pvals):
    """Holm-Bonferroni step-down. Returns adjusted p in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def check_instrument(joint, ded):
    print("INSTRUMENT CHECK -- reproduce the delivered cell means before testing")
    ok = True
    for st in STATES:
        cat, reg = series(joint, ded, st)
        cm = sum(a for a, _ in seed_means(cat)) / 4.0
        rm = sum(a for a, _ in seed_means(reg)) / 4.0
        ec, er = EXPECTED[st]
        good = abs(cm - ec) < TOL and abs(rm - er) < TOL
        ok &= good
        print(f"  {st:<11} cat {cm:8.4f} (esperado {ec:8.4f})   "
              f"reg {rm:8.4f} (esperado {er:8.4f})   {'ok' if good else 'FALHA'}")
    if not ok:
        sys.exit("instrument check failed -- not reporting any test")
    print()


def run_axis(joint, ded, axis, idx, family_note):
    print(f"{'=' * 78}\n{axis}\n{family_note}\n{'=' * 78}")
    rows, praw = [], []
    for st in STATES:
        pairs = series(joint, ded, st)[idx]
        d20 = [a - b for a, b in pairs]
        pos = sum(1 for x in d20 if x > 0)
        # one-sided: joint greater than dedicated
        w = stats.wilcoxon(d20, alternative="greater", method="exact")
        sm = seed_means(pairs)
        d4 = [a - b for a, b in sm]
        t = stats.ttest_rel([a for a, _ in sm], [b for _, b in sm], alternative="greater")
        rows.append((st, sum(d20) / 20.0, pos, w.pvalue, sum(d4) / 4.0, t.pvalue))
        praw.append(w.pvalue)
    adj = holm(praw)
    print(f"  {'dataset':<12}{'delta20':>9}{'folds+':>8}{'W exato p':>12}"
          f"{'Holm':>10}   |{'delta4':>9}{'t p':>11}")
    for (st, d20m, pos, pw, d4m, pt), pa in zip(rows, adj):
        print(f"  {st:<12}{d20m:>9.4f}{pos:>5}/20{pw:>12.2e}{pa:>10.4f}   |"
              f"{d4m:>9.4f}{pt:>11.2e}")
    print()
    return rows, adj


def main():
    joint, ded = load()
    check_instrument(joint, ded)
    run_axis(joint, ded, "NEXT CATEGORY -- superiority (joint > dedicated), macro-F1",
             0, "PRE-REGISTERED test, protocol §2. Holm within the six-state family (m=6).")
    run_axis(joint, ded, "NEXT REGION -- superiority (joint > dedicated), Acc@10",
             1, "NOT pre-registered as superiority. Protocol pins region to TOST "
                "non-inferiority.\nReported here as a sensitivity reading; the two "
                "delivered region claims carry\ntheir own post-hoc Holm family (m=4), "
                "deviation D-4.")
    print("delta20 = mean of the 20 matched fold differences; delta4 = mean of the 4 "
          "per-seed differences.\n't p' is the one-sided paired t on the four per-seed "
          "means (df=3) -- the footing the\nchapter reports. Holm here is applied to the "
          "Wilcoxon p only, within its own family.")


if __name__ == "__main__":
    main()
