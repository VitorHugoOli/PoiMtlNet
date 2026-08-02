#!/usr/bin/env python
"""
Figure 4 (MobiWac 2026): headline signed-deltas plot.

Per dataset, ordered by region count, plot:
  - the next-category delta (MTL - dedicated STL category ceiling), always positive;
  - the next-region   delta (MTL - dedicated STL region   ceiling), rising with
    region count across the five U.S. states and also positive at Istanbul.

A shaded +/- 2-point non-inferiority band and a zero line sit on the region axis
so the reader sees the AL/AZ deltas inside the band and the Istanbul/FL/TX/CA
deltas positive, while the category delta is up everywhere.

Numbers are the v17 JOINT-BEST deltas (2026-07-18: Joint(v17), one saved model
per fold, minus the n=20 best-vs-best dedicated ceilings; JOINT_BEST_RESULTS.md
+ CEILINGS_N20_FINAL.md). ALL SIX datasets are n=20 on both arms.

Run with the repo venv:
    /Users/vitor/Desktop/mestrado/ingred/.venv/bin/python fig4_deltas.py
Writes fig4_deltas.pdf next to this script.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Board numbers (Delta = MTL - dedicated STL ceiling, percentage points).
# Ordered by region count (ascending): Istanbul 520, AL 1109, AZ 1547,
# FL 4703, TX 6553, CA 8501.
# ---------------------------------------------------------------------------
STATES = [
    # label,      region count, category delta, region delta
    # v17 JOINT-BEST deltas (convention switched 2026-07-18, considerations_v3
    # #10): Joint(v17, one saved model per fold) minus the n=20 best-vs-best
    # dedicated ceilings (JOINT_BEST_RESULTS.md + CEILINGS_N20_FINAL.md).
    # ALL SIX datasets n=20 on both arms (CA/TX A1 landed 2026-07-11).
    ("Istanbul", 520, 8.58, 0.19),
    ("AL", 1109, 7.69, -0.41),
    ("AZ", 1547, 9.35, 0.00),
    ("FL", 4703, 5.33, 0.71),
    ("TX", 6553, 7.45, 2.11),
    ("CA", 8501, 6.45, 2.20),
]

labels = [s[0] for s in STATES]
region_counts = [s[1] for s in STATES]
cat_delta = [s[2] for s in STATES]
reg_delta = [s[3] for s in STATES]

# x tick labels carry the region count so the ordering axis is explicit.
xticklabels = [f"{lab}\n{cnt:,}" for lab, cnt in zip(labels, region_counts)]

NI_MARGIN = 2.0  # non-inferiority band, +/- 2 points of Acc@10

# ---------------------------------------------------------------------------
# Style: IEEE-friendly, serif, legible at ~3.3 in two-column width.
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,  # editable / embeddable text in the PDF
        "ps.fonttype": 42,
    }
)

CAT_COLOR = "#1f4e79"  # deep blue: category (always up)
REG_COLOR = "#c0392b"  # brick red: region (crosses the band)
BAND_COLOR = "#b8b8b8"
GRID_COLOR = "#dddddd"

x = list(range(len(STATES)))
bar_w = 0.40

fig, ax = plt.subplots(figsize=(5.2, 2.3))

# --- non-inferiority band and zero line (region reference frame) ----------
ax.axhspan(
    -NI_MARGIN,
    NI_MARGIN,
    color=BAND_COLOR,
    alpha=0.30,
    zorder=0,
    linewidth=0,
)
ax.axhline(0.0, color="#444444", linewidth=0.7, zorder=1)

# --- grouped bars ---------------------------------------------------------
xs_cat = [xi - bar_w / 2 for xi in x]
xs_reg = [xi + bar_w / 2 for xi in x]

bars_cat = ax.bar(
    xs_cat,
    cat_delta,
    width=bar_w,
    color=CAT_COLOR,
    edgecolor="white",
    linewidth=0.4,
    zorder=3,
    label="category",
)
bars_reg = ax.bar(
    xs_reg,
    reg_delta,
    width=bar_w,
    color=REG_COLOR,
    edgecolor="white",
    linewidth=0.4,
    zorder=3,
    label="region",
)

# --- value labels ---------------------------------------------------------
for xi, v in zip(xs_cat, cat_delta):
    ax.annotate(
        f"+{v:.1f}",
        (xi, v),
        textcoords="offset points",
        xytext=(0, 2.0),
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=CAT_COLOR,
    )
for xi, v in zip(xs_reg, reg_delta):
    sign = "+" if v >= 0 else "−"  # unicode minus, no em-dash policy unaffected
    va = "bottom" if v >= 0 else "top"
    off = 2.0 if v >= 0 else -2.0
    ax.annotate(
        f"{sign}{abs(v):.1f}",
        (xi, v),
        textcoords="offset points",
        xytext=(0, off),
        ha="center",
        va=va,
        fontsize=8.0,
        color=REG_COLOR,
    )

# NOTE (2026-07-18): all six datasets are n=20 on both arms (CA/TX A1 top-up
# landed 2026-07-11; single-seed disclosure retired from the paper).

# --- axes cosmetics -------------------------------------------------------
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_xlabel("dataset  (region count, low to high)", labelpad=2)
ax.set_ylabel("$\\Delta$ vs dedicated (pp)", labelpad=2)

ymax = max(cat_delta) + 2.2
ymin = min(min(reg_delta) - 1.6, -NI_MARGIN - 2.0)
ax.set_ylim(ymin, ymax)
ax.set_xlim(-0.6, len(STATES) - 0.4)

ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# The band is explained by the legend's "±2 pp band" entry alone; the earlier
# in-plot annotation overlapped the band edge and axis (removed 2026-07-10).

# --- legend ---------------------------------------------------------------
legend_handles = [
    Patch(facecolor=CAT_COLOR, edgecolor="white", label="category $\\Delta$"),
    Patch(facecolor=REG_COLOR, edgecolor="white", label="region $\\Delta$"),
    Patch(facecolor=BAND_COLOR, alpha=0.30, edgecolor="none", label="$\\pm2$ pp band"),
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.16),
    ncol=3,
    frameon=False,
    handlelength=1.2,
    handletextpad=0.5,
    columnspacing=1.1,
    borderaxespad=0.0,
)

fig.tight_layout(pad=0.4)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_deltas.pdf")
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
print(f"wrote {out_path}")
