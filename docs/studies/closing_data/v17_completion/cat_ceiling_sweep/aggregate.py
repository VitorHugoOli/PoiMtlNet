#!/usr/bin/env python3
"""Aggregate the STL cat-ceiling sweep → LR-response table + B*/LR* selection.

Reads sweep_results/<state>_bs<BS>_lr<LR>_s<SEED>.json (each = one 5-fold run, key cat_macro_f1_mean).
Prints, per (batch, lr) arm: per-state means and the POOLED mean over the requested screen states.
Selection rule (advisor panel): global (batch,lr) = argmax of the pooled screen-state mean. Never per-seed/fold max.

Usage: aggregate.py [--screen-states alabama,arizona] [--all]
"""
import argparse, glob, json, os, re, statistics as st
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
COLL = os.path.join(BASE, "sweep_results")
PAT = re.compile(r"^(?P<st>[a-z]+)_bs(?P<bs>\d+)_lr(?P<lr>[0-9.]+)_s(?P<seed>\d+)\.json$")

ap = argparse.ArgumentParser()
ap.add_argument("--screen-states", default="alabama,arizona")
ap.add_argument("--all", action="store_true", help="show every state, not just screen states")
args = ap.parse_args()
screen = args.screen_states.split(",")

# arm -> state -> {seed: f1}
data = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(os.path.join(COLL, "*.json")):
    m = PAT.match(os.path.basename(f))
    if not m:
        continue
    d = json.load(open(f))
    arm = (int(m["bs"]), float(m["lr"]))
    data[arm][m["st"]][int(m["seed"])] = d["cat_macro_f1_mean"]

def statecell(seedmap):
    vs = list(seedmap.values())
    return (st.mean(vs), st.pstdev(vs) if len(vs) > 1 else 0.0, len(vs))

arms = sorted(data.keys(), key=lambda a: (a[0], a[1]))
all_states = sorted({s for arm in data for s in data[arm]})
show_states = all_states if args.all else screen

print(f"{'batch':>6} {'lr':>7} | " + " ".join(f"{s[:4]:>14}" for s in show_states) + f" | {'POOLED('+'+'.join(x[:2] for x in screen)+')':>16}")
print("-" * (16 + 15 * len(show_states) + 20))
best = None
for arm in arms:
    bs, lr = arm
    cells = []
    for s in show_states:
        if s in data[arm]:
            m, sd, n = statecell(data[arm][s])
            cells.append(f"{m:6.2f}±{sd:4.2f}/{n}")
        else:
            cells.append(f"{'--':>14}")
    # pooled mean over screen states = mean of per-run values across screen states (all seeds present)
    pooled_vals = []
    for s in screen:
        pooled_vals.extend(data[arm].get(s, {}).values())
    pooled = st.mean(pooled_vals) if pooled_vals else float("nan")
    npool = len(pooled_vals)
    flag = ""
    if pooled_vals and (best is None or pooled > best[1]):
        best = (arm, pooled, npool)
    print(f"{bs:>6} {lr:>7} | " + " ".join(f"{c:>14}" for c in cells) + f" | {pooled:8.3f} (n={npool})")

if best:
    (bbs, blr), bmean, bn = best
    print(f"\n>>> B*/LR* (argmax pooled {'+'.join(screen)} mean): batch={bbs} lr={blr}  pooled={bmean:.3f} (n={bn})")
    # bs2048 best for the fallback comparison
    b2 = [(a, st.mean([v for s in screen for v in data[a].get(s, {}).values()]))
          for a in arms if a[0] == 2048 and any(data[a].get(s) for s in screen)]
    b8 = [(a, st.mean([v for s in screen for v in data[a].get(s, {}).values()]))
          for a in arms if a[0] == 8192 and any(data[a].get(s) for s in screen)]
    if b2 and b8:
        best2 = max(b2, key=lambda x: x[1]); best8 = max(b8, key=lambda x: x[1])
        print(f"    bs2048 best: lr={best2[0][1]} pooled={best2[1]:.3f}   |   bs8192 best: lr={best8[0][1]} pooled={best8[1]:.3f}")
        print(f"    Δ(bs8192_best − bs2048_best) = {best8[1]-best2[1]:+.3f} pp  → "
              + ("adopt bs8192 (non-inferior)" if best8[1] >= best2[1] - 0.30 else "FALL BACK to bs2048"))
