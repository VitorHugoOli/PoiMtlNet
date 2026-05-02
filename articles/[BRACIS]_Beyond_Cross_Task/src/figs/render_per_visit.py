"""Render F1: per-visit context counterfactual at Alabama.

3 bars at HGI / POI-pooled Check2HGI / canonical Check2HGI under matched-head
next_gru STL ceiling, with horizontal reference lines and delta annotations.

Numbers from PAPER_DRAFT.md / mechanism.tex Section 6.1:
  HGI                       25.26
  POI-pooled Check2HGI      29.57
  canonical Check2HGI       40.76
  per-visit  = 40.76 - 29.57 = +11.19 pp ~72%
  train-sig  = 29.57 - 25.26 =  +4.31 pp ~28%
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HGI = 25.26
POOLED = 29.57
CANONICAL = 40.76
PER_VISIT = CANONICAL - POOLED
TRAIN_SIG = POOLED - HGI
GAP = CANONICAL - HGI
PER_VISIT_PCT = 100.0 * PER_VISIT / GAP
TRAIN_SIG_PCT = 100.0 * TRAIN_SIG / GAP

plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "DejaVu Sans"

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=300)

labels = ["HGI\n(POI-stable)", "POI-pooled\nCheck2HGI", "canonical\nCheck2HGI"]
values = [HGI, POOLED, CANONICAL]
colors = ["#bbbbbb", "#7aa6c2", "#1f4e79"]

xs = [0, 1, 2]
bars = ax.bar(xs, values, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
for x, v in zip(xs, values):
    ax.text(x, v + 0.7, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# horizontal reference lines extending right from each bar top
y_max = 50
right_edge = 2.55
for x, v in zip(xs, values):
    ax.plot([x + 0.275, right_edge], [v, v], color="gray", linestyle=":", lw=0.7)

# bracket: training-signal (HGI -> pooled), drawn at right_edge
br_x = right_edge - 0.05
ax.annotate(
    "",
    xy=(br_x, POOLED),
    xytext=(br_x, HGI),
    arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.0),
)
ax.text(
    br_x + 0.08,
    (HGI + POOLED) / 2,
    f"training-signal\n+{TRAIN_SIG:.2f} pp ({TRAIN_SIG_PCT:.0f}%)",
    fontsize=8.5,
    va="center",
    ha="left",
    color="#444444",
)

# bracket: per-visit context (pooled -> canonical)
ax.annotate(
    "",
    xy=(br_x, CANONICAL),
    xytext=(br_x, POOLED),
    arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.2),
)
ax.text(
    br_x + 0.08,
    (POOLED + CANONICAL) / 2,
    f"per-visit context\n+{PER_VISIT:.2f} pp ({PER_VISIT_PCT:.0f}%)",
    fontsize=8.5,
    va="center",
    ha="left",
    color="#1f4e79",
    fontweight="bold",
)

ax.set_ylabel("next-category macro-F1 (%)", fontsize=10)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.5, 4.0)
ax.set_ylim(0, y_max)
ax.set_title(
    "Per-visit context decomposition at Alabama (matched-head next_gru STL ceiling)",
    fontsize=10,
)
ax.grid(axis="y", linestyle=":", alpha=0.35)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out = Path(__file__).parent / "per-visit.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Wrote {out} ({out.stat().st_size} bytes)")
