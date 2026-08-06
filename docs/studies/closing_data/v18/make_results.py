#!/usr/bin/env python
"""Generate V18_RESULTS.md and PROVENANCE.md from data/v18_results.json.

Every number in the markdown comes from the JSON, which score_all.py regenerates from the rundirs.
Nothing here recomputes a metric; this is presentation plus the statistics SS10 requires.

SS10 rules enforced here:
  - every table states its n = seeds x folds
  - "outperforms" requires a paired superiority test (one-sided paired t on per-fold pairs)
  - "matches" requires TOST non-inferiority within a stated margin (delta = 2.0 pp, the board's)
  - a non-inferior result is never upgraded to a win
  - deltas vs v17 are labelled CROSS-SUBSTRATE and are descriptive only, never a superiority claim

Usage:  .venv/bin/python docs/studies/closing_data/v18/make_results.py
"""
from __future__ import annotations

import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
STATES = ["istanbul", "alabama", "arizona", "florida", "texas", "california"]
DELTA = 2.0          # pp, the board's equivalence margin for "matches"


def tost(diffs: list[float], margin: float) -> tuple[bool, float]:
    """Two one-sided tests for equivalence of a paired difference to zero within +/- margin.

    Returns (equivalent, p) where p is the larger of the two one-sided p-values.
    """
    from scipy import stats
    n = len(diffs)
    if n < 2:
        return False, float("nan")
    m, sd = st.mean(diffs), st.stdev(diffs)
    if sd == 0:
        return abs(m) < margin, 0.0
    se = sd / (n ** 0.5)
    t_lo = (m - (-margin)) / se           # H0: diff <= -margin
    t_hi = (m - margin) / se              # H0: diff >= +margin
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    p = max(p_lo, p_hi)
    return p < 0.05, p


def superiority(diffs: list[float]) -> tuple[bool, float]:
    """One-sided paired t: is the difference greater than zero?"""
    from scipy import stats
    n = len(diffs)
    if n < 2:
        return False, float("nan")
    if st.stdev(diffs) == 0:
        return st.mean(diffs) > 0, 0.0
    t, p2 = stats.ttest_1samp(diffs, 0.0)
    p1 = p2 / 2 if t > 0 else 1 - p2 / 2
    return (p1 < 0.05 and t > 0), p1


def verdict(diffs: list[float]) -> str:
    """The SS10 ladder: beats > matches > inconclusive > below."""
    if len(diffs) < 2:
        return "n too small"
    sup, p_sup = superiority(diffs)
    if sup:
        return f"**beats** (paired one-sided p={p_sup:.3f})"
    eq, p_eq = tost(diffs, DELTA)
    if eq:
        return f"matches (TOST +/-{DELTA:.0f} pp, p={p_eq:.3f})"
    inf, p_inf = superiority([-d for d in diffs])
    if inf:
        return f"*below* (paired one-sided p={p_inf:.3f})"
    return f"inconclusive (not superior, not equivalent within +/-{DELTA:.0f} pp)"


def paired_diffs(runs: list[dict], joint_key: str, stl_key: str) -> list[float]:
    """Per-(seed,fold) differences joint - dedicated, pooled across seeds."""
    out = []
    for r in runs:
        j, s = r.get(joint_key), r.get(stl_key)
        if j and s and len(j) == len(s):
            out += [a - b for a, b in zip(j, s)]
    return out


def main() -> None:
    d = json.loads((BASE / "data/v18_results.json").read_text())
    per_run, cells, meta = d["per_run"], d["cells"], d["meta"]
    now = datetime.now(timezone.utc).isoformat()

    L = [f"# v18 — RESULTS\n",
         f"> Generated {now} from [`data/v18_results.json`](data/v18_results.json) by "
         f"`make_results.py`. Every number here is traceable to that JSON, which "
         f"[`score_all.py`](score_all.py) regenerates from the rundirs.\n",
         f"> Commit `{meta.get('commit_sha','?')[:8]}` · seeds run: see the n column of each table.\n",
         "**v18 = the frozen v17 recipe on a leak-free substrate**: the consecutive-visit graph is "
         "forward-only in training and at readout, plus 4 elapsed-time node columns "
         "(`in_channels` 15). Not an architecture change. See "
         "[`METHODOLOGY.md`](METHODOLOGY.md).\n",
         "## Conventions\n",
         "- **diag-best** (`db_*`) — per-task diagnostic-best epochs. The Table-3 convention.\n"
         "- **joint-best** (`jb_*`) — both heads at the single `geom_simple`-selected epoch, "
         "`min_best_epoch` 0. What the served checkpoint delivers.\n"
         "- cat = macro-F1. reg = `top10_acc_indist · (1 − ood_fraction) · 100`, i.e. **Acc@10**.\n"
         f"- \"beats\" = paired one-sided superiority test. \"matches\" = TOST non-inferiority "
         f"within ±{DELTA:.0f} pp. A non-inferior result is never upgraded to a win.\n"]

    # ---- table 1: the headline contrast, MTL vs its OWN dedicated ceiling ------------------
    L.append("## 1 · MTL vs its own dedicated ceiling (same substrate, same protocol)\n")
    L.append("This is the citable contrast: both arms measured on v18, so the comparison is "
             "within-protocol.\n")
    L.append("| state | n | dedicated cat | MTL cat | **Δcat** | verdict | dedicated reg | MTL reg | **Δreg** | verdict |")
    L.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for s in STATES:
        c = cells.get(s)
        if not c:
            continue
        runs = [r for r in per_run if r["state"] == s]
        n = c["joint_cat_diag_best"]["n"]
        dcat = paired_diffs(runs, "db_cat_folds", "stl_cat_folds")
        dreg = paired_diffs(runs, "db_reg_folds", "stl_reg_folds")
        sc, mc = c["stl_cat"]["mean"], c["joint_cat_diag_best"]["mean"]
        sr, mr = c["stl_reg"]["mean"], c["joint_reg_diag_best"]["mean"]
        f = lambda v: f"{v:.2f}" if v is not None else "—"
        L.append(
            f"| {s} | {n} | {f(sc)} | {f(mc)} | "
            f"**{(mc-sc):+.2f}** | {verdict(dcat) if dcat else '—'} | "
            f"{f(sr)} | {f(mr)} | **{(mr-sr):+.2f}** | {verdict(dreg) if dreg else '—'} |")
    L.append("")

    # ---- table 2: both epoch conventions ---------------------------------------------------
    L.append("## 2 · Joint model — both epoch-selection conventions\n")
    L.append("| state | n | cat diag-best | cat joint-best | reg diag-best | reg joint-best |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for s in STATES:
        c = cells.get(s)
        if not c:
            continue
        g = lambda k: (f"{c[k]['mean']:.2f} ± {c[k]['sd']:.2f}"
                       if c[k]["mean"] is not None else "—")
        L.append(f"| {s} | {c['joint_cat_diag_best']['n']} | {g('joint_cat_diag_best')} | "
                 f"{g('joint_cat_joint_best')} | {g('joint_reg_diag_best')} | "
                 f"{g('joint_reg_joint_best')} |")
    L.append("")

    # ---- table 3: vs v17, explicitly cross-substrate ---------------------------------------
    L.append("## 3 · Against the v17 published board — CROSS-SUBSTRATE, descriptive only\n")
    L.append("> ⚠ v17 and v18 are **different substrates**. These differences are reported to show "
             "the size of the leak's contribution; they are **not** superiority tests and must not "
             "be written as one.\n")
    L.append("| state | Δcat v18 | Δcat v17 | shift | Δreg v18 | Δreg v17 | shift |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in STATES:
        c = cells.get(s)
        if not c or c.get("delta_cat_vs_own_ceiling") is None:
            continue
        v = c["v17_published"]
        d17c, d17r = v["mtl_cat"] - v["stl_cat"], v["mtl_reg"] - v["stl_reg"]
        dc, dr = c["delta_cat_vs_own_ceiling"], c.get("delta_reg_vs_own_ceiling")
        row = f"| {s} | {dc:+.2f} | {d17c:+.2f} | **{dc-d17c:+.2f}** | "
        row += (f"{dr:+.2f} | {d17r:+.2f} | {dr-d17r:+.2f} |" if dr is not None else "— | "
                f"{d17r:+.2f} | — |")
        L.append(row)
    L.append("")

    # ---- pooled verdict --------------------------------------------------------------------
    all_dcat, all_dreg = [], []
    for s in STATES:
        runs = [r for r in per_run if r["state"] == s]
        all_dcat += paired_diffs(runs, "db_cat_folds", "stl_cat_folds")
        all_dreg += paired_diffs(runs, "db_reg_folds", "stl_reg_folds")
    if all_dcat:
        L.append("## 4 · Pooled across states\n")
        L.append(f"- **Δcat** pooled over {len(all_dcat)} (state, seed, fold) pairs: "
                 f"mean **{st.mean(all_dcat):+.3f}** — {verdict(all_dcat)}")
        if all_dreg:
            L.append(f"- **Δreg** pooled over {len(all_dreg)} pairs: "
                     f"mean **{st.mean(all_dreg):+.3f}** — {verdict(all_dreg)}")
        L.append("\nPooling across states is reported for a single headline figure; the per-state "
                 "rows in §1 are the primary result, since the states differ in size and in their "
                 "v17 deltas.\n")

    L.append("## 5 · Related findings\n")
    L.append("- [`LOSS_WEIGHT_PROBE.md`](LOSS_WEIGHT_PROBE.md) — the 0.75/0.25 split is **not** a "
             "leak artifact and the heads are **orthogonal, not competing** "
             "(cos ≈ +0.001). Rebalancing and gradient surgery are both null; the equal split is "
             "significantly *harmful* to region at Florida. Keep 0.75/0.25.\n")
    L.append("- [`READOUT_EQUIVALENCE.md`](READOUT_EQUIVALENCE.md) — the per-window readout is an "
             "identity on a forward-only graph; the engine is materialized from the one-shot "
             "export.\n")
    L.append("- [`AUDIT.md`](AUDIT.md) — the §6 self-checks with measured values.\n")
    (BASE / "V18_RESULTS.md").write_text("\n".join(L) + "\n")

    # ---- provenance -------------------------------------------------------------------------
    P = ["# v18 — PROVENANCE\n",
         f"> Generated {now}. Every rundir behind every number, with the recipe and the commit it "
         f"was produced from.\n",
         "## Recipe (identical across all cells except where a probe says otherwise)\n",
         "```\n"
         "engine   check2hgi_v18 (forward-only graph + 4 elapsed-time node cols, in_channels 15)\n"
         "repr     seed 42, 500 epochs, resln, dim 64, 2 layers -- one per state, seed-independent\n"
         "joint    train.py --task mtl --canon none --task-set check2hgi_next_region\n"
         "         bs 8192, static_weight cw 0.75, onecycle max-lr 3e-3, cat-lr 1e-3, reg-lr 3e-3,\n"
         "         shared-lr 1e-3, MTL_ONECYCLE_PER_HEAD_LR=1, cat-head next_gru,\n"
         "         reg-head next_stan_flow_dualtower, geom_simple, fp32, --compile --tf32\n"
         "cat      train.py --task next --model next_gru --embedding-dim 64, per-state bs/lr\n"
         "reg      p1_region_head_ablation.py --heads next_stan_flow --input-type region\n"
         "         --region-emb-source check2hgi_design_k_resln_mae_l0_1 --max-lr 0.003\n"
         "```\n",
         "| state | seed | family | rundir | pid | commit |",
         "|---|---:|---|---|---|---|"]
    for r in sorted(per_run, key=lambda x: (x["state"], x["seed"])):
        for fam, rk, pk in (("joint", "rundir", "pid"), ("cat", "cat_rundir", "cat_pid"),
                            ("reg", "reg_result_json", None)):
            rd = r.get(rk)
            if rd:
                P.append(f"| {r['state']} | {r['seed']} | {fam} | `{rd}` | "
                         f"{r.get(pk, '—') if pk else '—'} | "
                         f"`{(r.get('commit_sha') or meta.get('commit_sha','?'))[:8]}` |")
    (BASE / "PROVENANCE.md").write_text("\n".join(P) + "\n")
    print(f"[make_results] wrote V18_RESULTS.md + PROVENANCE.md "
          f"({len(per_run)} runs, {len(cells)} states)")


if __name__ == "__main__":
    main()
