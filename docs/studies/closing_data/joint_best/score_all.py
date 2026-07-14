#!/usr/bin/env python3
"""J1 joint-best scoring + aggregation for the v17/dk_ovl MobiWac board.

Runs scripts/closing_data/score_joint_best.py (SSOT selection logic) on every
v17 MTL rundir, then aggregates to Table-3 cells with three built-in audits:
  (A) scorer diag-best  ==  committed a40_matched_score.json  (per rundir)
  (B) scorer joint_epoch  ==  folds/foldN_info.json primary_checkpoint.epoch (per fold)
  (C) diag-best cell  ==  paper tbl3_results.tex rendered value

Reads rundirs from the MAIN checkout; runs the scorer from the WORKTREE (standardized
src/tracking/scoring.py). CPU-only, no retraining.
"""
import json
import subprocess
import statistics as st
from pathlib import Path

MAIN = Path("/home/vitor.oliveira/PoiMtlNet")                       # holds run data
WORKTREE = Path("/home/vitor.oliveira/PoiMtlNet/.claude/worktrees/joint-result")  # holds scorer
PY = "/home/vitor.oliveira/.venv/bin/python"
SCORER = WORKTREE / "scripts/closing_data/score_joint_best.py"

# (state, seed, PID)  -- authoritative mapping verified against summary.tsv + each
# rundir's a40_matched_score.json "seed" field.
MANIFEST = [
    ("alabama",   0,   983885), ("alabama",   1,   983884), ("alabama",   7,   988104), ("alabama",   100, 988155),
    ("arizona",   0,   992231), ("arizona",   1,   992283), ("arizona",   7,   998602), ("arizona",   100, 998757),
    ("florida",   0,  1058652), ("florida",   1,  1088683), ("florida",   7,  1097178), ("florida",   100, 1119752),
    ("california",0,  1363454),
    ("texas",     0,  1419069),
    ("istanbul",  0,  3852943), ("istanbul",  1,  3858409), ("istanbul",  7,  3863852), ("istanbul",  100, 3869098),
]

# Paper tbl3 rendered DIAG-BEST cells (cat, reg) -- the parity-(C) target.
PAPER_DIAG = {
    "istanbul":  (63.33, 75.44), "alabama": (64.54, 69.80), "arizona": (65.84, 59.56),
    "florida":   (79.85, 77.42), "texas":   (77.23, 67.07), "california": (77.04, 65.69),
}
# Dedicated ceilings (n=20 best-vs-best, CEILINGS_N20_FINAL.md): (cat, reg)
CEIL = {
    "istanbul": (54.74, 75.16), "alabama": (56.82, 70.11), "arizona": (56.43, 59.46),
    "florida":  (74.51, 76.70), "texas":   (69.79, 64.95), "california": (70.60, 63.49),
}
ORDER = ["istanbul", "alabama", "arizona", "florida", "texas", "california"]


def find_rundir(state, pid):
    hits = sorted((MAIN / "results/check2hgi_dk_ovl" / state).glob(f"mtlnet_*_{pid}"))
    if not hits:
        raise SystemExit(f"MISSING rundir: {state} pid={pid}")
    return hits[0]


def fold_ckpt_epochs(rundir):
    """primary_checkpoint.epoch per fold from folds/foldN_info.json (training's joint gate)."""
    out = {}
    for f in sorted((rundir / "folds").glob("fold*_info.json")):
        try:
            d = json.loads(f.read_text())
            fn = d.get("fold_number")
            ep = d.get("primary_checkpoint", {}).get("epoch")
            if fn is not None and ep is not None:
                out[int(fn)] = int(ep)
        except (json.JSONDecodeError, OSError):
            pass
    return out


def run_one(state, seed, pid):
    rundir = find_rundir(state, pid)
    tag = f"{state}_s{seed}"
    r = subprocess.run([PY, str(SCORER), str(rundir), "--seed", str(seed), "--tag", tag],
                       capture_output=True, text=True, cwd=str(WORKTREE))
    if r.returncode != 0:
        raise SystemExit(f"scorer FAILED for {tag}:\n{r.stderr}\n{r.stdout}")
    sc = json.loads((rundir / "joint_best_score.json").read_text())
    a40 = json.loads((rundir / "a40_matched_score.json").read_text())
    ckpt = fold_ckpt_epochs(rundir)

    # --- audit A: scorer diag-best == committed a40 diag-best
    audit_a = {
        "cat_scorer": sc["diag_best"]["cat_macro_f1_mean"], "cat_a40": a40["cat_macro_f1_mean"],
        "reg_scorer": sc["diag_best"]["reg_full_top10_mean"], "reg_a40": a40["reg_full_top10_mean"],
    }
    audit_a["ok"] = (abs(audit_a["cat_scorer"] - audit_a["cat_a40"]) < 0.01 and
                     abs(audit_a["reg_scorer"] - audit_a["reg_a40"]) < 0.01)
    # --- audit B: scorer joint_epoch (1-based CSV) == training primary_checkpoint.epoch (0-based loop) + 1.
    # The +1 is a confirmed indexing convention, not a mismatch (verified: same cat_f1/reg_indist/joint_score).
    jb_epochs = {row["fold"]: row["joint_best"]["epoch"] for row in sc["per_fold"] if row["joint_best"]}
    mism = {f: (jb_epochs.get(f), ckpt.get(f)) for f in sorted(set(jb_epochs) & set(ckpt))
            if jb_epochs[f] != ckpt[f] + 1}
    audit_b = {"scorer_epochs": jb_epochs, "ckpt_epochs": ckpt, "mismatches": mism, "ok": not mism}
    # --- collapse tell: reg diag best-epoch <= 5 (precision-collapse signature)
    collapse = [e for e in sc["diag_best"]["reg_best_epochs"] if e <= 5]

    return {
        "state": state, "seed": seed, "pid": pid, "rundir": str(rundir),
        "jb_cat": sc["joint_best"]["cat_macro_f1_mean"], "jb_reg": sc["joint_best"]["reg_full_top10_mean"],
        "jb_cat_folds": sc["joint_best"]["cat_per_fold"], "jb_reg_folds": sc["joint_best"]["reg_per_fold"],
        "jb_epochs": sc["joint_best"]["epochs"],
        "db_cat": sc["diag_best"]["cat_macro_f1_mean"], "db_reg": sc["diag_best"]["reg_full_top10_mean"],
        "delta_cat": sc["delta_joint_minus_diag"]["cat"], "delta_reg": sc["delta_joint_minus_diag"]["reg"],
        "audit_a": audit_a, "audit_b": audit_b, "reg_collapse_epochs": collapse,
        "warnings": sc.get("warnings", []),
    }


def agg(vals):
    m = st.mean(vals); s = st.pstdev(vals) if len(vals) > 1 else 0.0
    return round(m, 4), round(s, 4)


def main():
    per_run = [run_one(*m) for m in MANIFEST]
    by_state = {}
    for r in per_run:
        by_state.setdefault(r["state"], []).append(r)

    cells = {}
    for state, runs in by_state.items():
        runs.sort(key=lambda r: r["seed"])
        n_seeds = len(runs)
        if n_seeds > 1:  # n=20 : mean over per-seed fold-means, cross-seed pstdev
            jc, jcs = agg([r["jb_cat"] for r in runs]);  jr, jrs = agg([r["jb_reg"] for r in runs])
            dc, dcs = agg([r["db_cat"] for r in runs]);  dr, drs = agg([r["db_reg"] for r in runs])
            basis = f"n={n_seeds*5} ({n_seeds} seeds x5f); +/- = cross-seed sd of per-seed means"
        else:            # single seed : fold-mean already, fold pstdev
            r0 = runs[0]
            jc, jcs = agg(r0["jb_cat_folds"]); jr, jrs = agg(r0["jb_reg_folds"])
            dc, dcs = r0["db_cat"], None; dr, drs = r0["db_reg"], None
            basis = "seed-0 x5f (provisional); +/- = fold sd"
        cells[state] = {
            "seeds": [r["seed"] for r in runs], "basis": basis,
            "jb_cat": jc, "jb_cat_sd": jcs, "jb_reg": jr, "jb_reg_sd": jrs,
            "db_cat": dc, "db_cat_sd": dcs, "db_reg": dr, "db_reg_sd": drs,
            "paper_diag": PAPER_DIAG[state], "ceiling": CEIL[state],
            "jb_dcat_vs_ceil": round(jc - CEIL[state][0], 2), "jb_dreg_vs_ceil": round(jr - CEIL[state][1], 2),
            "db_dcat_vs_ceil": round(dc - CEIL[state][0], 2), "db_dreg_vs_ceil": round(dr - CEIL[state][1], 2),
            "gap_cat": round(jc - dc, 3), "gap_reg": round(jr - dr, 3),
        }

    out = {"per_run": per_run, "cells": cells, "order": ORDER}
    outp = WORKTREE / "docs/studies/closing_data/joint_best/data/j1_results.json"
    outp.write_text(json.dumps(out, indent=2))

    # ---------- report ----------
    print("=" * 100)
    print("AUDIT A (scorer diag-best == committed a40_matched_score.json) + AUDIT B (joint_epoch == training ckpt epoch + 1, indexing)")
    print("=" * 100)
    all_ok = True
    for r in per_run:
        a, b = r["audit_a"], r["audit_b"]
        flag = "OK " if (a["ok"] and b["ok"] and not r["reg_collapse_epochs"]) else "!! "
        if not (a["ok"] and b["ok"]):
            all_ok = False
        print(f"{flag}{r['state']:11}s{r['seed']:<4} A:cat {a['cat_scorer']:.3f}/{a['cat_a40']:.3f} "
              f"reg {a['reg_scorer']:.3f}/{a['reg_a40']:.3f} [{'ok' if a['ok'] else 'MISMATCH'}]"
              f"  B:{'ok' if b['ok'] else 'MISMATCH '+str(b['mismatches'])}"
              f"  reg_collapse={r['reg_collapse_epochs'] or '-'}")
    print(f"\nAll parity A+B OK: {all_ok}\n")

    print("=" * 100)
    print("CELLS  (JB = joint-best single checkpoint ; DB = diag-best two-epoch ; ceil = dedicated n=20)")
    print("=" * 100)
    hdr = f"{'state':11} {'basis':46} {'JBcat':>8} {'DBcat':>8} {'paperC':>7} | {'JBreg':>8} {'DBreg':>8} {'paperR':>7} | {'gapC':>6} {'gapR':>6}"
    print(hdr)
    for s in ORDER:
        c = cells[s]
        pc, pr = c["paper_diag"]
        cok = "ok" if abs(c["db_cat"] - pc) < 0.05 else "DIFF"
        rok = "ok" if abs(c["db_reg"] - pr) < 0.05 else "DIFF"
        print(f"{s:11} {c['basis']:46} {c['jb_cat']:8.3f} {c['db_cat']:8.3f} {pc:7.2f} | "
              f"{c['jb_reg']:8.3f} {c['db_reg']:8.3f} {pr:7.2f} | {c['gap_cat']:6.2f} {c['gap_reg']:6.2f}  [C:{cok} R:{rok}]")

    print("\n" + "=" * 100)
    print("DELTA vs DEDICATED CEILING  --  does the verdict change under joint-best?")
    print("=" * 100)
    print(f"{'state':11} {'DBdcat':>7} {'JBdcat':>7} | {'DBdreg':>7} {'JBdreg':>7}   verdict-shift?")
    for s in ORDER:
        c = cells[s]
        print(f"{s:11} {c['db_dcat_vs_ceil']:+7.2f} {c['jb_dcat_vs_ceil']:+7.2f} | "
              f"{c['db_dreg_vs_ceil']:+7.2f} {c['jb_dreg_vs_ceil']:+7.2f}")


if __name__ == "__main__":
    main()
