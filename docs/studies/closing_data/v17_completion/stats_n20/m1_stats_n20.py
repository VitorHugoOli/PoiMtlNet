#!/usr/bin/env python3
"""M1-PARTIAL — v17 pre-registered stats at the 4 fully-n=20 datasets (AL, AZ, FL, Istanbul).

Pre-registration: docs/studies/closing_data/STATISTICAL_PROTOCOL.md (§2 paired Wilcoxon cat
superiority; §3 TOST reg non-inferiority, delta_reg = 2 pp; §4 pairing discipline; §5 Holm).
Conventions mirror scripts/closing_data/superiority_wilcoxon.py + region_match_tost.py.

Reads ONLY committed artifacts (run from repo root: `.venv/bin/python
docs/studies/closing_data/v17_completion/stats_n20/m1_stats_n20.py`). It does NOT read any
gitignored rundir — where the committed tree lacks the MTL-side per-seed/per-fold values
(AL/AZ/FL v17 MTL: A40-only, see RESULTS.md LIMITS) the cell is reported as PENDING with a
descriptive delta only; nothing is approximated or fabricated.

Pairing levels actually used (per RESULTS.md):
  - Istanbul: SEED-LEVEL paired, n=4 (MTL side committed only as per-seed 5-fold means).
    NOTE the n=4 exact one-sided Wilcoxon floor = 1/2^4 = 0.0625 > 0.05 — Wilcoxon cannot
    clear alpha at this footing regardless of effect size; the paired t (df=3) is reported
    alongside as the powered seed-level test (deviation logged per protocol §8).
  - AL/AZ/FL: PENDING (no committed MTL-side values at any granularity).
"""
import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parents[5]
DELTA_REG = 2.0   # pp, pre-registered per-axis margin (STATISTICAL_PROTOCOL §3.2)
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Committed sources
# ---------------------------------------------------------------------------
SWEEP = REPO / "docs/studies/closing_data/v17_completion/cat_ceiling_sweep/sweep_results"
P1 = REPO / "docs/results/P1"
IST_RUNS = REPO / "docs/studies/closing_data/v17_completion/h3_istanbul/step3_runs"
SEEDS = [0, 1, 7, 100]

# STL cat ceiling arms (best-vs-best, CEILINGS_N20_FINAL.md; AZ = bs8192@0.005 per the
# 2026-07-08 correction — NOT bs2048@0.005):
CAT_CEIL_ARM = {"AL": ("alabama", "bs2048_lr0.005"),
                "AZ": ("arizona", "bs8192_lr0.005"),
                "FL": ("florida", "bs8192_lr0.005")}

# STL reg ceiling JSONs (per-fold, next_stan_flow prior-off, dk_ovl) — board s0 + topup {1,7,100}:
REG_CEIL_JSON = {
    "AL": {0: "region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_s0.json",
           1: "region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_topup_s1.json",
           7: "region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_topup_s7.json",
           100: "region_head_alabama_region_5f_50ep_alabama_ovl_stl_reg_topup_s100.json"},
    "AZ": {0: "region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_s0.json",
           1: "region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_topup_s1.json",
           7: "region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_topup_s7.json",
           100: "region_head_arizona_region_5f_50ep_arizona_ovl_stl_reg_topup_s100.json"},
    "FL": {0: "region_head_florida_region_5f_50ep_florida_ovl_stl_reg_s0.json",
           1: "region_head_florida_region_5f_50ep_florida_ovl_stl_reg_topup_s1.json",
           7: "region_head_florida_region_5f_50ep_florida_ovl_stl_reg_topup_s7.json",
           100: "region_head_florida_region_5f_50ep_florida_ovl_stl_reg_topup_s100.json"},
    "Istanbul": {s: f"region_head_istanbul_region_5f_50ep_istanbul_ovl_stl_reg_s{s}.json"
                 for s in SEEDS},
}

# v17 MTL committed aggregates (perhead_lr_n20.md — n=20 mean +/- cross-seed pstdev). The
# underlying per-seed/per-fold values are NOT committed (A40 gitignored rundirs) -> PENDING.
MTL_AGG = {"AL": (64.540, 69.801), "AZ": (65.835, 59.563), "FL": (79.848, 77.421)}


def cat_ceiling_perfold(key):
    """{seed: [5 fold macro-F1]} for the best-vs-best STL cat ceiling arm."""
    state, arm = CAT_CEIL_ARM[key]
    out = {}
    for s in SEEDS:
        d = json.load(open(SWEEP / f"{state}_{arm}_s{s}.json"))
        out[s] = d["cat_per_fold"]
    return out


def reg_ceiling_perfold(key):
    """{seed: [5 fold top10*100]} for the STL reg ceiling (next_stan_flow prior-off)."""
    out = {}
    for s, fn in REG_CEIL_JSON[key].items():
        d = json.load(open(P1 / fn))
        out[s] = [x["top10_acc"] * 100 for x in d["heads"]["next_stan_flow"]["per_fold"]]
    return out


def ist_perseed(prefix):
    """Istanbul step3 per-seed 5-fold means (mtl_cat / mtl_reg / cat_ceil / reg_ceil)."""
    return {s: float(open(IST_RUNS / f"{prefix}_s{s}.txt").read().strip()) for s in SEEDS}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def paired_superiority(mtl, ceil, n_label):
    """One-sided paired Wilcoxon (MTL > ceiling; exact) + paired t (df=n-1) on the same pairs."""
    d = [m - c for m, c in zip(mtl, ceil)]
    n = len(d)
    pos = sum(1 for x in d if x > 0)
    w = stats.wilcoxon(mtl, ceil, alternative="greater", zero_method="wilcox", mode="exact")
    t = stats.ttest_rel(mtl, ceil, alternative="greater")
    return dict(n=n, n_label=n_label, mean_d=st.mean(d), sd_d=(st.stdev(d) if n > 1 else 0.0),
                pos=f"{pos}/{n}", d=[round(x, 4) for x in d],
                p_wilcoxon=float(w.pvalue), p_t=float(t.pvalue))


def paired_tost(mtl, ceil, n_label, delta=DELTA_REG, alpha=ALPHA):
    """Paired TOST non-inferiority/equivalence vs +/-delta (t-based, as region_match_tost.py)."""
    d = [m - c for m, c in zip(mtl, ceil)]
    n = len(d)
    mean_d, sd_d = st.mean(d), st.stdev(d)
    se = sd_d / n ** 0.5
    df = n - 1
    p_lower = stats.t.sf((mean_d + delta) / se, df)   # H0a: mu <= -delta (non-inferiority side)
    p_upper = stats.t.cdf((mean_d - delta) / se, df)  # H0b: mu >= +delta
    p_tost = max(p_lower, p_upper)
    tcrit = stats.t.ppf(1 - alpha, df)
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)   # 90% = (1-2*alpha) CI
    return dict(n=n, n_label=n_label, mean_d=mean_d, sd_d=sd_d, d=[round(x, 4) for x in d],
                p_noninf=float(p_lower), p_tost=float(p_tost), ci90=ci,
                non_inferior=(ci[0] > -delta and ci[1] < delta))


def check(label, got, want, tol=0.006):
    ok = abs(got - want) <= tol
    print(f"  [{'OK' if ok else 'MISMATCH'}] {label}: recomputed {got:.4f} vs board {want}")
    if not ok:
        sys.exit(f"ABORT: committed artifacts do not reproduce the board value for {label}")


# ---------------------------------------------------------------------------
print("=" * 88)
print("M1-PARTIAL — v17 vs n=20 best-vs-best ceilings (AL/AZ/FL/Istanbul); CA/TX await A1")
print("=" * 88)

# --- 0 · reproduce the CEILINGS_N20_FINAL board numbers from the committed artifacts ------
print("\n--- 0 · artifact -> board reproduction gate ---")
cat_ceil = {k: cat_ceiling_perfold(k) for k in CAT_CEIL_ARM}
reg_ceil = {k: reg_ceiling_perfold(k) for k in REG_CEIL_JSON}
board_cat = {"AL": 56.82, "AZ": 56.43, "FL": 74.51}
board_reg = {"AL": 70.11, "AZ": 59.46, "FL": 76.70, "Istanbul": 75.16}
for k in ["AL", "AZ", "FL"]:
    allf = [v for s in SEEDS for v in cat_ceil[k][s]]
    check(f"{k} STL cat ceiling (n=20)", st.mean(allf), board_cat[k])
for k in ["AL", "AZ", "FL", "Istanbul"]:
    seed_means = [st.mean(reg_ceil[k][s]) for s in SEEDS]
    check(f"{k} STL reg ceiling (n=20)", st.mean(seed_means), board_reg[k])
ist = {p: ist_perseed(p) for p in ["mtl_cat", "mtl_reg", "cat_ceil", "reg_ceil"]}
check("Istanbul MTL cat (n=20)", st.mean(ist["mtl_cat"].values()), 63.33)
check("Istanbul MTL reg (n=20)", st.mean(ist["mtl_reg"].values()), 75.44)
check("Istanbul STL cat ceiling (n=20)", st.mean(ist["cat_ceil"].values()), 54.74)
# Istanbul reg-ceiling txt == P1 JSON fold-means (same runs, two committed encodings)
for s in SEEDS:
    assert abs(ist["reg_ceil"][s] - st.mean(reg_ceil["Istanbul"][s])) < 0.001, s

# --- 1 · Istanbul — the one dataset with a committed MTL side (per-seed, n=4) -------------
print("\n--- 1 · Istanbul (dk_ovl+v17, H3) — SEED-LEVEL paired, n=4 ---")
mtl_cat = [ist["mtl_cat"][s] for s in SEEDS]
stl_cat = [ist["cat_ceil"][s] for s in SEEDS]
mtl_reg = [ist["mtl_reg"][s] for s in SEEDS]
stl_reg = [st.mean(reg_ceil["Istanbul"][s]) for s in SEEDS]

sup = paired_superiority(mtl_cat, stl_cat, "n=4 seeds (each a 5-fold mean)")
print(f"  CAT superiority (MTL > ceiling): Δ={sup['mean_d']:+.3f} ± {sup['sd_d']:.3f} pp, "
      f"pairs+={sup['pos']}, per-seed Δ={sup['d']}")
print(f"    exact one-sided Wilcoxon p = {sup['p_wilcoxon']:.4f}  "
      f"(= the n=4 floor 1/16 — cannot clear 0.05 at this n; power-limited, not evidence-limited)")
print(f"    paired t (df=3), one-sided p = {sup['p_t']:.2e}   "
      f"(effect ≈ {sup['mean_d']/sup['sd_d']:.0f}× the cross-seed σ_d)")
ist_cat_p = sup["p_t"]

tost = paired_tost(mtl_reg, stl_reg, "n=4 seeds (each a 5-fold mean)")
print(f"  REG TOST non-inferiority (δ_reg = {DELTA_REG:.0f} pp): Δ={tost['mean_d']:+.3f} ± "
      f"{tost['sd_d']:.3f} pp, per-seed Δ={tost['d']}")
print(f"    TOST p = {tost['p_tost']:.2e}; 90% CI = ({tost['ci90'][0]:+.3f}, {tost['ci90'][1]:+.3f}) "
      f"pp -> {'NON-INFERIOR (CI within ±2)' if tost['non_inferior'] else 'fails'}")
sup_reg = paired_superiority(mtl_reg, stl_reg, "n=4")
print(f"    (supplementary, Δ>0 'beats': Wilcoxon p = {sup_reg['p_wilcoxon']:.4f} [n=4 floor]; "
      f"paired t p = {sup_reg['p_t']:.2e}; CI already entirely > 0)")

# --- 2 · AL/AZ/FL — MTL side not committed -> PENDING (descriptive Δ only) ----------------
print("\n--- 2 · AL / AZ / FL — pre-registered tests PENDING (v17 MTL per-seed/per-fold values")
print("        are A40-only, gitignored; see RESULTS.md LIMITS). Descriptive Δ from committed")
print("        aggregates (perhead_lr_n20.md MTL vs artifact-recomputed ceilings):")
for k in ["AL", "AZ", "FL"]:
    cat_c = st.mean([v for s in SEEDS for v in cat_ceil[k][s]])
    reg_c = st.mean([st.mean(reg_ceil[k][s]) for s in SEEDS])
    dcat = MTL_AGG[k][0] - cat_c
    dreg = MTL_AGG[k][1] - reg_c
    ceil_sd_cat = st.pstdev([st.mean(cat_ceil[k][s]) for s in SEEDS])
    ceil_sd_reg = st.pstdev([st.mean(reg_ceil[k][s]) for s in SEEDS])
    print(f"  {k}: Δcat = {dcat:+.3f} pp (ceiling cross-seed σ {ceil_sd_cat:.3f}); "
          f"Δreg = {dreg:+.3f} pp (ceiling cross-seed σ {ceil_sd_reg:.3f})  -> PENDING")

# --- 3 · family correction (protocol §5.2, scoped to the M1-partial 4-dataset family) -----
print("\n--- 3 · Holm across the 4-dataset cat-superiority family (M1-PARTIAL) ---")
print("  Family = {AL, AZ, FL, Istanbul} cat superiority (m=4). Only Istanbul has a runnable")
print("  test today -> full Holm CANNOT be computed; we report the conservative bound:")
print(f"    Istanbul Bonferroni-bounded p_adj <= m * p = 4 x {ist_cat_p:.2e} = {4*ist_cat_p:.2e}"
      f"  ({'<' if 4*ist_cat_p < ALPHA else '>='} {ALPHA})")
print("    (Holm adj p <= Bonferroni adj p, so Istanbul survives ANY completion of the family.)")
print("  Reg TOST cells are equivalence tests with their own δ_reg verdict — NOT pooled into")
print("  the cat Holm family (protocol §5.2).")
print("\n  ⚠ M1-PARTIAL: the paper's headline family is 6 datasets (protocol §5.2). The full")
print("    6-dataset Holm re-runs after A1 lands CA/TX n=20 + the AL/AZ/FL MTL per-seed/per-fold")
print("    artifacts are committed from the A40.")
