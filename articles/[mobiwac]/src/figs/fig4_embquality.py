#!/usr/bin/env python
"""Figure 4 (MobiWac 2026): category embedding-quality panel.

Plots the two CATEGORY-separability metrics on which the check-in-level
Check2HGI representation clearly beats the place-level HGI embedding:

  * silhouette by next-category   (check2hgi ~0.48  vs  hgi ~0.00)
  * kNN-by-category accuracy      (check2hgi ~0.98  vs  hgi ~0.78)

Region geometry is deliberately NOT shown: on region both engines are at the
floor and HGI scores spuriously higher, which would mislead. The story here is
strictly that the check-in-level substrate separates by category; its
region-neutral behavior is carried in the caption, not in this panel.

Source: docs/studies/closing_data/PART1_QUALITY/summary.md (L0 geometry, next-cat;
5-fold CV, mean +/- SD over 6 states x 5 folds = 30 samples).

Run:
  /Users/vitor/Desktop/mestrado/ingred/.venv/bin/python fig4_embquality.py
Writes: fig4_embquality.pdf  (this directory)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Authoritative numbers (PART1_QUALITY/summary.md, L0 geometry -- next-cat).
# mean +/- SD over folds. CATEGORY metrics only; region intentionally omitted.
# ---------------------------------------------------------------------------
METRICS = ["Silhouette\n(by category)", "kNN accuracy\n(by category)"]

CHECK2HGI = {"mean": [0.4775, 0.9794], "sd": [0.0275, 0.0045]}
HGI = {"mean": [0.0000, 0.7780], "sd": [0.0042, 0.0292]}

# Colors: a saturated accent for the contributed substrate, a muted gray for
# the place-embedding baseline so the contrast reads at IEEE column size.
C_CHECK2HGI = "#1f4e79"  # deep blue  -- Check2HGI (ours)
C_HGI = "#b0b0b0"        # neutral gray -- HGI (place embedding)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,  # embed TrueType (no Type-3) for IEEE
            "ps.fonttype": 42,
        }
    )

    x = np.arange(len(METRICS))
    width = 0.36

    # ~3.3 in wide single-column IEEE figure.
    fig, ax = plt.subplots(figsize=(3.3, 2.05))

    b1 = ax.bar(
        x - width / 2,
        CHECK2HGI["mean"],
        width,
        yerr=CHECK2HGI["sd"],
        capsize=2.5,
        color=C_CHECK2HGI,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
        label="Check2HGI (check-in level)",
    )
    b2 = ax.bar(
        x + width / 2,
        HGI["mean"],
        width,
        yerr=HGI["sd"],
        capsize=2.5,
        color=C_HGI,
        edgecolor="black",
        linewidth=0.5,
        error_kw={"elinewidth": 0.7, "capthick": 0.7},
        label="HGI (place embedding)",
    )

    # Numeric labels above each bar.
    def annotate(bars, means, sds):
        for rect, m, s in zip(bars, means, sds):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                m + s + 0.025,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    annotate(b1, CHECK2HGI["mean"], CHECK2HGI["sd"])
    annotate(b2, HGI["mean"], HGI["sd"])

    ax.set_ylim(0, 1.12)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Category separability")
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.tick_params(axis="x", length=0)
    ax.axhline(0, color="black", linewidth=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        fontsize=6.8,
        handlelength=1.2,
        columnspacing=1.0,
        handletextpad=0.5,
    )

    fig.tight_layout(pad=0.4)
    out = Path(__file__).resolve().parent / "fig4_embquality.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
