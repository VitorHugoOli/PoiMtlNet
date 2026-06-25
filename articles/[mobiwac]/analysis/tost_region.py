#!/usr/bin/env python3
"""
TOST equivalence analysis for the small-state next-region "matches".

Question (PAPER_PLAN.md §5.3 / §6.2, GLOSSARY honesty rule): the small-state
next-region cells are MATCHES (AL Delta_reg -0.18, AZ -0.06, Istanbul -0.52).
To call them "statistically non-inferior within a two-point margin" we run a
paired TOST (two one-sided t-tests) at delta = 2 points of Acc@10, and we report
the achieved power to reject a true 2-point gap given the observed fold variance.

Paired design: the single joint model (MTL champion-G) vs the dedicated STL
region ceiling, both at seed 0 over 5 user-disjoint folds, on the SAME folds and
the SAME overlapping-window (dk_ovl) representation, so the per-fold differences
are well-defined paired observations.

Metric: next-region Acc@10 (top-10 accuracy), in points (x100).
  - MTL champion per-fold = reg_full = top10_acc_indist * (1 - ood_frac) at the
    in-distribution-best epoch (the board's deployable "full" reg score).
  - STL region ceiling per-fold = top10_acc at the top10-best epoch (the board's
    dedicated single-task reg ceiling).
Both are the board's matched-score convention; the fold means reproduce the
RESULTS_BOARD Delta_reg exactly (AL -0.18, AZ -0.06, Istanbul -0.52).

Sources (committed JSONs):
  MTL champion-G (reg_per_fold):
    AL : docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json
    AZ : docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json
    IST: docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json
  STL region ceiling (heads.next_stan_flow.per_fold[*].top10_acc):
    AL : docs/results/P1/region_head_alabama_region_5f_50ep_al_dkovl_leakfree_prior.json
    AZ : docs/results/P1/region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json
    IST: docs/results/P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json

Run:
  /Users/vitor/Desktop/mestrado/ingred/.venv/bin/python \
      "articles/[mobiwac]/analysis/tost_region.py"
Writes the markdown summary next to this script (tost_region.md).
"""
from __future__ import annotations

import json
import math
import os
from datetime import date

import numpy as np
from scipy import stats

REPO = "/Users/vitor/Desktop/mestrado/ingred"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "tost_region.md")

# Non-inferiority margin (points of Acc@10). Justified a-priori in PAPER_PLAN
# §5.3: a deployment-grounded "two points of Acc@10 is negligible over 1k-8.5k
# region classes" margin, fixed before looking at whether the claim survives.
DELTA = 2.0
ALPHA = 0.05  # one-sided alpha for each of the two TOST tests

# Per-state source files. mtl_field = key in the MTL JSON holding the per-fold
# reg list; stl path goes through heads.next_stan_flow.per_fold[*].top10_acc.
STATES = {
    "Alabama (1,109 regions)": dict(
        short="AL",
        n_regions=1109,
        mtl=os.path.join(REPO, "docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json"),
        mtl_field="reg_per_fold",
        stl=os.path.join(REPO, "docs/results/P1/region_head_alabama_region_5f_50ep_al_dkovl_leakfree_prior.json"),
    ),
    "Arizona (1,547 regions)": dict(
        short="AZ",
        n_regions=1547,
        mtl=os.path.join(REPO, "docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json"),
        mtl_field="reg_per_fold",
        stl=os.path.join(REPO, "docs/results/P1/region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json"),
    ),
    "Istanbul (520 mahalle)": dict(
        short="IST",
        n_regions=520,
        mtl=os.path.join(REPO, "docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json"),
        mtl_field="reg_per_fold",
        stl=os.path.join(REPO, "docs/results/P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json"),
    ),
}


def load_mtl_per_fold(path: str, field: str) -> np.ndarray:
    """MTL reg per-fold, already in points (x100)."""
    with open(path) as fh:
        d = json.load(fh)
    return np.asarray(d[field], dtype=float)


def load_stl_per_fold(path: str) -> np.ndarray:
    """STL region ceiling per-fold top10_acc, scaled to points (x100)."""
    with open(path) as fh:
        d = json.load(fh)
    heads = d["heads"]
    # the dedicated reg head is next_stan_flow (the only head in these files)
    if "next_stan_flow" in heads:
        head = heads["next_stan_flow"]
    else:
        head = next(iter(heads.values()))
    return np.asarray([row["top10_acc"] * 100.0 for row in head["per_fold"]], dtype=float)


def paired_tost(mtl: np.ndarray, stl: np.ndarray, delta: float, alpha: float):
    """
    Paired TOST for non-inferiority of MTL relative to STL, margin = delta points.

    diff_i = mtl_i - stl_i (positive favors the joint model).
    Non-inferiority is established if the (1 - 2*alpha) CI for mean(diff) lies
    entirely above -delta. Equivalence (both bounds) is also reported.

    Lower one-sided test  H0L: mean(diff) <= -delta   vs  H1L: mean(diff) > -delta
    Upper one-sided test  H0U: mean(diff) >= +delta   vs  H1U: mean(diff) < +delta
    TOST p-value = max(p_lower, p_upper). Non-inferiority p-value = p_lower.
    """
    diff = mtl - stl
    n = diff.size
    dof = n - 1
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1))
    se_d = sd_d / math.sqrt(n)

    # one-sided t statistics
    t_lower = (mean_d - (-delta)) / se_d      # for H1L: diff > -delta
    p_lower = stats.t.sf(t_lower, dof)         # P(T >= t_lower) under H0L boundary
    t_upper = (mean_d - (delta)) / se_d        # for H1U: diff < +delta
    p_upper = stats.t.cdf(t_upper, dof)        # P(T <= t_upper) under H0U boundary
    p_tost = max(p_lower, p_upper)

    # (1 - 2*alpha) two-sided CI == the TOST CI
    tcrit = stats.t.ppf(1.0 - alpha, dof)
    ci_lo = mean_d - tcrit * se_d
    ci_hi = mean_d + tcrit * se_d

    non_inferior = p_lower < alpha          # CI lower bound > -delta
    equivalent = p_tost < alpha             # CI inside (-delta, +delta)

    return dict(
        diff=diff, n=n, dof=dof, mean_d=mean_d, sd_d=sd_d, se_d=se_d,
        t_lower=t_lower, p_lower=p_lower, t_upper=t_upper, p_upper=p_upper,
        p_tost=p_tost, ci_lo=ci_lo, ci_hi=ci_hi,
        non_inferior=non_inferior, equivalent=equivalent,
        ci_level=1.0 - 2.0 * alpha,
    )


def achieved_power_noninferiority(sd_d: float, n: int, delta: float, alpha: float,
                                  true_diff: float = 0.0) -> float:
    """
    Achieved power of the one-sided non-inferiority test (the lower TOST arm) to
    reject a true gap of size -delta, evaluated at the assumed true mean
    difference true_diff (default 0, i.e. truly no difference), given the
    observed paired-difference SD and n. Noncentral-t formulation.

    The test rejects H0L (gap >= delta) when t_lower > t_{1-alpha, dof}.
    Under the alternative with mean = true_diff, t_lower ~ noncentral t with
    ncp = (true_diff + delta) / (sd/sqrt(n)).
    """
    dof = n - 1
    se = sd_d / math.sqrt(n)
    ncp = (true_diff - (-delta)) / se
    tcrit = stats.t.ppf(1.0 - alpha, dof)
    # power = P(noncentral_t(dof, ncp) > tcrit)
    return float(stats.nct.sf(tcrit, dof, ncp))


def power_curve_min_detectable(sd_d: float, n: int, alpha: float, target_power: float = 0.80):
    """Smallest margin delta* that the test could reject with target power at
    true_diff=0 (informational: how wide a margin we are powered for)."""
    dof = n - 1
    se = sd_d / math.sqrt(n)
    tcrit = stats.t.ppf(1.0 - alpha, dof)
    # solve for delta s.t. nct.sf(tcrit, dof, delta/se) = target_power
    lo, hi = 0.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = stats.nct.sf(tcrit, dof, mid / se)
        if p < target_power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    results = {}
    for name, cfg in STATES.items():
        mtl = load_mtl_per_fold(cfg["mtl"], cfg["mtl_field"])
        stl = load_stl_per_fold(cfg["stl"])
        assert mtl.size == stl.size == 5, f"{name}: expected 5 folds, got {mtl.size}/{stl.size}"
        r = paired_tost(mtl, stl, DELTA, ALPHA)
        r["power_at_true0"] = achieved_power_noninferiority(r["sd_d"], r["n"], DELTA, ALPHA, true_diff=0.0)
        # achieved power to reject a TRUE 2-point gap: evaluate the test's ability
        # to reject H0(gap>=delta) when the truth sits AT the margin boundary is 0
        # by construction; the paper-relevant power is "power to detect equivalence
        # if the truth is no gap", reported above. We also report the (more
        # conservative) power if the true difference equals the observed mean_d.
        r["power_at_obs"] = achieved_power_noninferiority(r["sd_d"], r["n"], DELTA, ALPHA, true_diff=r["mean_d"])
        r["min_detectable_margin_80"] = power_curve_min_detectable(r["sd_d"], r["n"], ALPHA, 0.80)
        r["mtl"], r["stl"] = mtl, stl
        r["n_regions"] = cfg["n_regions"]
        r["short"] = cfg["short"]
        results[name] = r

    write_markdown(results)
    # console echo
    for name, r in results.items():
        verdict = "NON-INFERIOR" if r["non_inferior"] else "NOT non-inferior"
        print(f"{name}: mean diff {r['mean_d']:+.3f} pp, "
              f"{int(r['ci_level']*100)}% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}], "
              f"p_NI={r['p_lower']:.4f} -> {verdict}; "
              f"power(true gap=0)={r['power_at_true0']:.2f}")


def fmt_list(a):
    return ", ".join(f"{x:.2f}" for x in a)


def write_markdown(results):
    today = date.today().isoformat()
    lines = []
    L = lines.append
    L("# Region non-inferiority (TOST) for the small-state matches")
    L("")
    L(f"> Generated by `analysis/tost_region.py` on {today}. Paired TOST equivalence "
      f"test of the single joint model against the dedicated single-task region "
      f"ceiling on next-region Acc@10, with a non-inferiority margin of "
      f"delta = {DELTA:.0f} points. n = 5 (seed 0, five user-disjoint folds). "
      f"This file is the source for the §5.3 protocol line and the §6.2 / §7 "
      f"small-state region prose.")
    L("")
    L("## The test")
    L("")
    L("For each small-region-count dataset we form the per-fold difference "
      "`diff_i = MTL_i - STL_i` in points of Acc@10, where MTL is the single joint "
      "model (champion) and STL is the dedicated single-task region model, both at "
      "seed 0 on the same five folds and the same overlapping-window representation. "
      "We run two one-sided paired t-tests (TOST) at a margin of "
      f"delta = {DELTA:.0f} points: the joint model is declared "
      "**statistically non-inferior within the two-point margin** if the lower one-sided "
      f"test rejects a true gap of {DELTA:.0f} points (equivalently, if the lower bound "
      "of the 90% confidence interval for the mean difference lies above "
      f"-{DELTA:.0f}). We report the mean paired difference, its 90% CI, the "
      "non-inferiority p-value, and the achieved power to reject a true two-point "
      "gap at the observed fold variance.")
    L("")
    L("## Verdict per dataset")
    L("")
    L("| Dataset | regions | STL reg | MTL reg | mean diff (pp) | 90% CI (pp) | NI p-value | non-inferior (delta=2)? | power vs true 2-pt gap |")
    L("|---|---:|---:|---:|---:|:---:|---:|:---:|---:|")
    for name, r in results.items():
        ci = f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]"
        ni = "**yes**" if r["non_inferior"] else "no"
        L(f"| {name} | {r['n_regions']:,} | {r['stl'].mean():.2f} | {r['mtl'].mean():.2f} | "
          f"{r['mean_d']:+.2f} | {ci} | {r['p_lower']:.4f} | {ni} | {r['power_at_true0']:.2f} |")
    L("")
    L("_STL reg = dedicated single-task region ceiling (Acc@10 fold mean); "
      "MTL reg = single joint model (Acc@10 fold mean); mean diff = MTL - STL. "
      "The 90% CI is the TOST interval (1 - 2*alpha, alpha = 0.05). Power vs true "
      "2-point gap = achieved power of the non-inferiority test to reject a true "
      "gap of two points when the truth is no gap, at the observed paired-difference "
      "SD and n = 5._")
    L("")
    L("## Per-fold detail (Acc@10, points)")
    L("")
    for name, r in results.items():
        L(f"**{name}.** "
          f"MTL folds: [{fmt_list(r['mtl'])}]; "
          f"STL folds: [{fmt_list(r['stl'])}]; "
          f"paired diffs: [{fmt_list(r['diff'])}] "
          f"(mean {r['mean_d']:+.3f}, SD {r['sd_d']:.3f}, SE {r['se_d']:.3f}). "
          f"NI t = {r['t_lower']:.3f} on {r['dof']} df, p = {r['p_lower']:.4f}. "
          f"TOST (two-sided equivalence) p = {r['p_tost']:.4f} "
          f"({'equivalent' if r['equivalent'] else 'not two-sided equivalent'}). "
          f"Margin the test is 80%-powered for at this variance: "
          f"~{r['min_detectable_margin_80']:.2f} pp.")
        L("")
    L("## Power statement (for §5.3 / §6.2)")
    L("")
    any_underpowered = any(r["power_at_true0"] < 0.80 for r in results.values())
    pw = {r["short"]: r["power_at_true0"] for r in results.values()}
    mdm = {r["short"]: r["min_detectable_margin_80"] for r in results.values()}
    L(f"At the observed per-fold variance and n = 5, the non-inferiority test has "
      f"achieved power "
      + ", ".join(f"{s} {pw[s]:.2f}" for s in pw)
      + f" to reject a true two-point gap when the joint model is in fact no worse "
      f"than the dedicated model. ")
    if any_underpowered:
        L("")
        L("**Caveat (honest):** with only five folds at a single seed, the test is "
          "underpowered at some datasets (power below 0.80), so a non-inferiority "
          "result is suggestive rather than conclusive; the test is 80%-powered only "
          "for wider margins ("
          + ", ".join(f"{s} ~{mdm[s]:.1f} pp" for s in mdm)
          + "). The planned {1,7,100} top-up to n = 20 will sharpen this. We report "
          "the n = 5 result as provisional and label it as such in the paper.")
    else:
        L("")
        L("The test is adequately powered (>= 0.80) at every small-region-count "
          "dataset, so the non-inferiority results are not an artifact of low power.")
        L("")
        L("**Why the power is high at only n = 5 (read this honestly).** The single "
          "joint model and the dedicated model are evaluated on the *same* folds and "
          "the *same* representation, so their per-fold Acc@10 scores move together "
          "almost fold-for-fold. The paired difference therefore has a very small SD "
          "("
          + ", ".join(f"{r['short']} {r['sd_d']:.2f}" for r in results.values())
          + " pp), which is far inside the two-point margin; that low difference-"
          "variance, not a large sample, is what drives the power. This is a property "
          "of the paired design, and it is also why the test is 80%-powered for "
          "margins well under one point ("
          + ", ".join(f"{r['short']} ~{r['min_detectable_margin_80']:.2f} pp" for r in results.values())
          + "). The remaining limitation is the single seed: n = 5 bounds the "
          "*seed-to-seed* variation we have not yet sampled, so we still label the "
          "result provisional (n = 5) and the planned {1,7,100} top-up to n = 20 "
          "remains the confirmation.")
    L("")
    L("## Suggested prose (drop-in, plain words)")
    L("")
    L("> On next-region the joint model stays within a two-point margin of the "
      "dedicated model at the small region counts. A paired two one-sided test "
      "(TOST) at a two-point Acc@10 margin finds the joint model statistically "
      "non-inferior to the dedicated region model at "
      + describe_ni(results)
      + ". The margin was fixed in advance on deployment grounds (two points of "
      "Acc@10 over one thousand to eight thousand regions is negligible), not chosen "
      "to fit the result.")
    L(">")
    if any_underpowered:
        L("> These small-state results rest on five folds at a single seed, so the "
          "non-inferiority test is underpowered against a true two-point gap; we "
          "report them as provisional (n = 5) and a multi-seed run is planned.")
    else:
        L("> Because the two models share folds and representation, their per-fold "
          "scores are tightly paired, so even with five folds the test rejects a "
          "true two-point gap with high power; we still report these as provisional "
          "(n = 5, single seed) pending a multi-seed run.")
    L("")
    L("## Provenance")
    L("")
    L("MTL champion per-fold (`reg_per_fold`, already in points): "
      "AL `docs/results/closing_data/h100/alabama_s0_mtl_fp32_matched_score.json`, "
      "AZ `docs/results/closing_data/h100/arizona_s0_mtl_fp32_matched_score.json`, "
      "Istanbul `docs/results/second_dataset/istanbul/istanbul_stride1_s0_mtl_fp32_matched_score.json`. "
      "STL region ceiling per-fold (`heads.next_stan_flow.per_fold[*].top10_acc`, x100): "
      "AL `docs/results/P1/region_head_alabama_region_5f_50ep_al_dkovl_leakfree_prior.json`, "
      "AZ `docs/results/P1/region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json`, "
      "Istanbul `docs/results/P1/region_head_istanbul_region_5f_50ep_istanbul_stride1_stl_reg_s0.json`. "
      "Fold means reproduce the RESULTS_BOARD Delta_reg (AL -0.18, AZ -0.06, Istanbul -0.52).")
    L("")

    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def describe_ni(results):
    parts = []
    for name, r in results.items():
        tag = name.split(" (")[0]
        if r["non_inferior"]:
            parts.append(f"{tag} (mean {r['mean_d']:+.2f} pp, NI p = {r['p_lower']:.3f})")
    if not parts:
        return "no small-region-count dataset (the test did not reach non-inferiority at n = 5)"
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


if __name__ == "__main__":
    main()
