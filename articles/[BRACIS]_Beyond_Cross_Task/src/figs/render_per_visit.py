"""Render F1: per-visit context counterfactual at Alabama and Arizona.

Two-panel figure (AL | AZ), three bars per panel: HGI / POI-pooled Check2HGI /
canonical Check2HGI, under the matched-head GRU single-task ceiling. The
panel format preserves per-state shares (72% vs 64%) which are the mechanism
evidence — averaging across n=2 states would hide that.

Numbers (verified Round 6, 2026-05-04 against per-fold JSONs in
docs/studies/check2hgi/results/phase1_perfold/):
  AL: HGI 25.26, POI-pooled 29.57, canonical 40.76  → per-visit ~72%
  AZ: HGI 28.99, POI-pooled 34.09, canonical 43.17  → per-visit ~64%
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# (label, HGI, POI-pooled Check2HGI, canonical Check2HGI)
PANELS = [
    ("Alabama (AL)",  25.26, 29.57, 40.76),
    ("Arizona (AZ)",  28.99, 34.09, 43.17),
]

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

fig, axes = plt.subplots(
    1, 2, figsize=(8.4, 3.6), dpi=300, sharey=True,
    gridspec_kw={"wspace": 0.18},
)

bar_labels = ["HGI\n(POI-stable)", "POI-pooled\nCheck2HGI", "canonical\nCheck2HGI"]
colors = ["#bbbbbb", "#7aa6c2", "#1f4e79"]
y_max = 50

for ax, (state_label, hgi, pooled, canonical) in zip(axes, PANELS):
    values = [hgi, pooled, canonical]
    per_visit = canonical - pooled
    train_sig = pooled - hgi
    gap = canonical - hgi
    pv_pct = 100.0 * per_visit / gap
    ts_pct = 100.0 * train_sig / gap

    xs = [0, 1, 2]
    ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
    for x, v in zip(xs, values):
        ax.text(x, v + 0.7, f"{v:.2f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")

    # horizontal reference lines extending right from each bar top
    right_edge = 2.55
    for x, v in zip(xs, values):
        ax.plot([x + 0.275, right_edge], [v, v],
                color="gray", linestyle=":", lw=0.7)

    br_x = right_edge - 0.05
    # bracket: training-signal (HGI -> pooled)
    ax.annotate(
        "",
        xy=(br_x, pooled),
        xytext=(br_x, hgi),
        arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0),
    )
    ax.text(
        br_x + 0.08,
        (hgi + pooled) / 2,
        f"train-sig\n+{train_sig:.2f} pp ({ts_pct:.0f}%)",
        fontsize=8,
        va="center", ha="left",
        color="#444444",
    )

    # bracket: per-visit context (pooled -> canonical)
    ax.annotate(
        "",
        xy=(br_x, canonical),
        xytext=(br_x, pooled),
        arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.2),
    )
    ax.text(
        br_x + 0.08,
        (pooled + canonical) / 2,
        f"per-visit\n+{per_visit:.2f} pp ({pv_pct:.0f}%)",
        fontsize=8,
        va="center", ha="left",
        color="#1f4e79", fontweight="bold",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(0, y_max)
    ax.set_title(state_label, fontsize=10.5, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("next-category macro-F1 (%)", fontsize=10)

fig.suptitle(
    "Per-visit context decomposition under the matched-head GRU single-task ceiling",
    fontsize=10, y=1.02,
)

plt.tight_layout()
out = Path(__file__).parent / "per-visit.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
