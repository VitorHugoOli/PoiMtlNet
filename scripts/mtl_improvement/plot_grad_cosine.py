#!/usr/bin/env python3
"""Plot task-gradient cosine cos(∇L_cat, ∇L_reg) on the shared trunk over training,
for the champion-recipe (static_weight) runs at AL (small) and FL (large). The
finding: cosine ≈ 0 → the two tasks' gradients are orthogonal → there is no
gradient conflict for MTL balancers (PCGrad/CAGrad/Nash/GradNorm/...) to resolve,
which is why tuned static scalarization is on the Pareto front (T4 audit, 2026-06-08).
"""
import csv, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "Florida (large · ~4,700 regions)":
        ("results/check2hgi_design_k_resln_mae_l0_1/florida/"
         "mtlnet_lr1.0e-04_bs2048_ep50_20260608_043241_2845765", "#1f77b4"),
    "Alabama (small · ~1,100 regions)":
        ("results/check2hgi_design_k_resln_mae_l0_1/alabama/"
         "mtlnet_lr1.0e-04_bs2048_ep50_20260608_043059_2845268", "#d62728"),
}


def load(rd):
    """Return (epochs, mean, std) of grad_cosine_shared across folds."""
    per_ep = {}
    for f in sorted(glob.glob(rd + "/diagnostics/fold*_diagnostics.csv")):
        for r in csv.DictReader(open(f)):
            v = r.get("grad_cosine_shared")
            if v in (None, "", "nan"):
                continue
            try:
                ep = int(float(r["epoch"])); val = float(v)
            except (ValueError, KeyError):
                continue
            per_ep.setdefault(ep, []).append(val)
    eps = sorted(per_ep)
    mean = np.array([np.mean(per_ep[e]) for e in eps])
    std = np.array([np.std(per_ep[e]) for e in eps])
    allv = np.array([v for e in eps for v in per_ep[e]])
    return np.array(eps), mean, std, allv


fig, ax = plt.subplots(figsize=(9, 5.2))
overall = []
for label, (rd, color) in RUNS.items():
    eps, mean, std, allv = load(rd)
    overall.append((label, allv.mean()))
    ax.plot(eps, mean, color=color, lw=2.0, label=f"{label}   (mean = {allv.mean():+.3f})")
    ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.15)

ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.7)
ax.axhspan(-0.05, 0.05, color="grey", alpha=0.10)
ax.text(0.99, 0.05, "orthogonal band  (|cos| < 0.05)", transform=ax.get_yaxis_transform(),
        ha="right", va="bottom", fontsize=8, color="grey")

ax.set_xlabel("Training epoch", fontsize=11)
ax.set_ylabel(r"cosine$\,(\nabla \mathcal{L}_{\mathrm{cat}},\ \nabla \mathcal{L}_{\mathrm{reg}})$"
              "\non shared parameters", fontsize=11)
ax.set_title("Next-category vs. next-region gradients are orthogonal\n"
             "→ no task conflict for MTL gradient-balancers to resolve",
             fontsize=12.5, fontweight="bold")
ax.set_ylim(-0.35, 0.35)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
fig.text(0.5, -0.02,
         "Champion MTL recipe (static_weight 0.75/0.25), 5-fold mean ± 1 s.d.  ·  "
         "cos ≈ 0 at both scales  ·  PoiMTLnet / Check2HGI",
         ha="center", fontsize=8, color="#444")
fig.tight_layout()
out = "docs/studies/mtl_improvement/figs/grad_cosine_tasks.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print("wrote", out)
for label, m in overall:
    print(f"  {label}: mean cosine = {m:+.4f}")
