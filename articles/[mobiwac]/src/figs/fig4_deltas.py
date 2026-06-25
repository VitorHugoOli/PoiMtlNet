#!/usr/bin/env python
"""
Figure 4 (MobiWac 2026): headline signed-deltas plot.

Per state, ordered by region count, plot:
  - the next-category delta (MTL - dedicated STL category ceiling), always positive;
  - the next-region   delta (MTL - dedicated STL region   ceiling), negative at the
    small region counts and positive at the large.

A shaded +/- 2-point non-inferiority band and a zero line sit on the region axis
so the reader sees the region delta crossing from within-the-band (small region
counts) to clearly positive (large region counts), while the category delta is up
everywhere.

Numbers are the authoritative board deltas (RESULTS_BOARD.md, Delta = MTL - STL
ceiling, in percentage points). n=5 (seed 0) provisional.

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
    ("Istanbul", 520, 6.69, -0.52),
    ("AL", 1109, 7.69, -0.18),
    ("AZ", 1547, 6.26, -0.06),
    ("FL", 4703, 4.68, 0.57),
    ("TX", 6553, 7.56, 2.06),
    ("CA", 8501, 7.07, 2.18),
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
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
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

fig, ax = plt.subplots(figsize=(3.3, 2.45))

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
        fontsize=6.0,
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
        fontsize=6.0,
        color=REG_COLOR,
    )

# NOTE: TX is now CLOSED at 5 folds (fp32 single-device, +2.06 region); the old
# "TX 2/5 folds" annotation was removed (stale, and it overlapped the CA bar).
# The whole board is n=5 (seed 0) provisional, stated once in the caption.

# --- axes cosmetics -------------------------------------------------------
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_xlabel("state  (region count, low to high)", labelpad=2)
ax.set_ylabel("delta vs dedicated ceiling (pp)", labelpad=2)

ymax = max(cat_delta) + 2.2
ymin = min(min(reg_delta) - 1.6, -NI_MARGIN - 2.0)
ax.set_ylim(ymin, ymax)
ax.set_xlim(-0.6, len(STATES) - 0.4)

ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

# annotate the band once, on the left, so its meaning is clear
ax.annotate(
    "non-inferiority band (±2 pp)",
    (-0.5, -NI_MARGIN),
    textcoords="offset points",
    xytext=(0, -8.5),
    ha="left",
    va="top",
    fontsize=5.6,
    color="#555555",
)

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
