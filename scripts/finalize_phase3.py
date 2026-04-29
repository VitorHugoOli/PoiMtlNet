"""Finalize Phase 3 MTL CH18 closure on CA + TX.

Extracts per-fold metrics from MTL run dirs, writes per-fold JSONs to
`phase1_perfold/`, runs paired tests for cat F1 + reg Acc@10 + reg MRR,
prints a cross-state CH18 status board.

Usage:
    python3 scripts/finalize_phase3.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PERFOLD = REPO / "docs" / "studies" / "check2hgi" / "results" / "phase1_perfold"
PAIRED = REPO / "docs" / "studies" / "check2hgi" / "results" / "paired_tests"
PERFOLD.mkdir(parents=True, exist_ok=True)
PAIRED.mkdir(parents=True, exist_ok=True)


def _latest_run_dir(engine: str, state: str, prefix: str = "mtlnet") -> Path | None:
    base = RESULTS / engine / state
    if not base.exists():
        return None
    candidates = sorted(base.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def extract_mtl_perfold(engine: str, state: str) -> dict | None:
    """Extract MTL per-fold cat + reg metrics from latest mtlnet_* run dir."""
    rd = _latest_run_dir(engine, state)
    if not rd:
        print(f"  [extract_mtl] no run dir for {engine}/{state}")
        return None

    cat_out: dict = {}
    reg_out: dict = {}
    folds_dir = rd / "folds"
    for fold_idx in range(5):
        fp = folds_dir / f"fold{fold_idx + 1}_info.json"
        if not fp.exists():
            print(f"  [extract_mtl] missing {fp}")
            return None
        d = json.load(fp.open())
        be = d["diagnostic_best_epochs"]
        cat_m = be["next_category"]["metrics"]
        reg_m = be["next_region"]["metrics"]
        cat_out[f"fold_{fold_idx}"] = {
            "f1": cat_m["f1"],
            "accuracy": cat_m["accuracy"],
        }
        reg_out[f"fold_{fold_idx}"] = {
            "f1": reg_m.get("f1"),
            "acc1": reg_m.get("top1_acc"),
            "acc5": reg_m.get("top5_acc"),
            "acc10": reg_m.get("top10_acc_indist", reg_m.get("top10_acc")),
            "mrr": reg_m.get("mrr_indist", reg_m.get("mrr")),
        }

    upstate = state[:2].upper()  # CA, TX
    cat_path = PERFOLD / f"{upstate}_{engine}_mtl_cat.json"
    reg_path = PERFOLD / f"{upstate}_{engine}_mtl_reg.json"
    cat_path.write_text(json.dumps(cat_out, indent=2))
    reg_path.write_text(json.dumps(reg_out, indent=2))
    print(f"  [extract_mtl] {state}/{engine} → {cat_path.name}, {reg_path.name} (run_dir={rd.name})")
    return {"cat": cat_path, "reg": reg_path, "run_dir": rd}


def run_paired(c2_path: Path, hgi_path: Path, metric: str, task: str,
               state: str, tost: float | None = None) -> Path:
    cmd = [
        sys.executable, str(REPO / "scripts/analysis/substrate_paired_test.py"),
        "--check2hgi", str(c2_path),
        "--hgi", str(hgi_path),
        "--metric", metric,
        "--task", task,
        "--state", state,
        "--out-dir", str(PAIRED),
    ]
    if tost is not None:
        cmd += ["--tost-margin", str(tost)]
    print(f"\n  [paired] state={state} task={task} metric={metric}")
    subprocess.run(cmd, check=True)
    return PAIRED / f"{state}_{task}_{metric}.json"


def status_board() -> None:
    """Print cross-state CH18 status board (cat F1 + reg Acc@10)."""
    print("\n=== CH18 cross-state confirmation: cat F1 (MTL+C2HGI > MTL+HGI) ===")
    print(f"{'State':<5}  {'C2HGI':>14}  {'HGI':>14}  {'Δ':>7}  {'Wilcoxon p':>10}")
    for code, full in (("AL", "alabama"), ("AZ", "arizona"), ("FL", "florida"),
                       ("CA", "california"), ("TX", "texas")):
        c = PERFOLD / f"{code}_check2hgi_mtl_cat.json"
        h = PERFOLD / f"{code}_hgi_mtl_cat.json"
        if not c.exists() or not h.exists():
            print(f"{code:<5}  per-fold MTL cat JSONs missing — skipping")
            continue
        cf = np.array([json.load(c.open())[f"fold_{i}"]["f1"] for i in range(5)])
        hf = np.array([json.load(h.open())[f"fold_{i}"]["f1"] for i in range(5)])
        pt = PAIRED / f"{full}_mtl_cat_f1.json"
        p = "?"
        if pt.exists():
            p = json.load(pt.open())["superiority"]["wilcoxon_p_greater"]
            p = f"{p:.4f}"
        print(f"{code:<5}  {cf.mean()*100:>5.2f} ± {cf.std()*100:.2f}  {hf.mean()*100:>5.2f} ± {hf.std()*100:.2f}  {(cf-hf).mean()*100:+5.2f}    {p:>10}")

    print("\n=== CH18 cross-state confirmation: reg Acc@10 (MTL+C2HGI > MTL+HGI) ===")
    print(f"{'State':<5}  {'C2HGI':>14}  {'HGI':>14}  {'Δ':>7}  {'Wilcoxon p':>10}")
    for code, full in (("AL", "alabama"), ("AZ", "arizona"), ("FL", "florida"),
                       ("CA", "california"), ("TX", "texas")):
        c = PERFOLD / f"{code}_check2hgi_mtl_reg.json"
        h = PERFOLD / f"{code}_hgi_mtl_reg.json"
        if not c.exists() or not h.exists():
            print(f"{code:<5}  per-fold MTL reg JSONs missing — skipping")
            continue
        cf = np.array([json.load(c.open())[f"fold_{i}"]["acc10"] for i in range(5)])
        hf = np.array([json.load(h.open())[f"fold_{i}"]["acc10"] for i in range(5)])
        pt = PAIRED / f"{full}_mtl_reg_acc10.json"
        p = "?"
        if pt.exists():
            p = json.load(pt.open())["superiority"]["wilcoxon_p_greater"]
            p = f"{p:.4f}"
        print(f"{code:<5}  {cf.mean()*100:>5.2f} ± {cf.std()*100:.2f}  {hf.mean()*100:>5.2f} ± {hf.std()*100:.2f}  {(cf-hf).mean()*100:+5.2f}    {p:>10}")


def main():
    print("=== Phase 3 finalization ===\n")

    # 1. Extract per-fold for all 4 cells
    print("[1/3] Extracting per-fold metrics from MTL run dirs...")
    extracted: dict = {}
    for state in ("california", "texas"):
        for engine in ("check2hgi", "hgi"):
            r = extract_mtl_perfold(engine, state)
            if r:
                extracted[(state, engine)] = r
            else:
                print(f"  [warn] {state}/{engine} extraction failed — paired tests for this state will be skipped")

    # 2. Run paired tests
    print("\n[2/3] Running paired tests...")
    for state in ("california", "texas"):
        c2 = extracted.get((state, "check2hgi"))
        hgi = extracted.get((state, "hgi"))
        if not c2 or not hgi:
            print(f"  [skip] {state} — missing one or both engines")
            continue
        # Cat F1
        run_paired(c2["cat"], hgi["cat"], "f1", "mtl_cat", state)
        # Reg Acc@10 + MRR
        run_paired(c2["reg"], hgi["reg"], "acc10", "mtl_reg", state)
        run_paired(c2["reg"], hgi["reg"], "mrr",   "mtl_reg", state)

    # 3. Status board
    print("\n[3/3] Cross-state CH18 status board:")
    status_board()

    print("\n=== Done ===")
    print("Next: update PHASE3_TRACKER.md, append CH18 closure section to ")
    print("research/SUBSTRATE_COMPARISON_FINDINGS.md, then commit + push.")


if __name__ == "__main__":
    main()
