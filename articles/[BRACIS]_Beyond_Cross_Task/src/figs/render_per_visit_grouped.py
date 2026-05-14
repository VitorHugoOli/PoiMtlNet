"""Render the per-visit context counterfactual as a compact grouped-bar plot.

Sibling to ``render_per_visit_5state.py``. Lays out all states on a single
shared axis as 3 bars per state group (HGI / POI-pooled Check2HGI / canonical
Check2HGI) with the per-visit share percentage annotated above each canonical
bar. Drops the per-visit / training-signal brackets that the 5-panel version
carries — the §6.1 prose now lists the per-state pp breakdown inline, and
keeping the figure compact saves ~30-40 % of figure-area at the same height.

Usage
-----
    python3 articles/[BRACIS]_Beyond_Cross_Task/src/figs/render_per_visit_grouped.py \\
        [--perfold-dir docs/studies/check2hgi/results/phase1_perfold] \\
        [--out articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png] \\
        [--states AL AZ FL CA TX] \\
        [--strict]

Auto-skips states with missing cells (warns and continues). Pass --strict to
error out instead. Output goes to per-visit.png by default so a swap is
mechanism.tex-side: just point ``\\includegraphics`` at this file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STATE_LABEL = {
    "AL": "Alabama (AL)",
    "AZ": "Arizona (AZ)",
    "FL": "Florida (FL)",
    "CA": "California (CA)",
    "TX": "Texas (TX)",
}

ENGINES = ("check2hgi", "check2hgi_pooled", "hgi")


def load_perfold(perfold_dir: Path, state_prefix: str, engine: str) -> list[float]:
    """Return per-fold macro-F1 (in percent) for (state, engine), or [].

    Picks the most recently modified file matching the pattern; index-agnostic
    on fold_* keys (handles both 0-indexed and 1-indexed JSONs).
    """
    pattern = f"{state_prefix}_{engine}_cat_gru_5f50ep*.json"
    candidates = list(perfold_dir.glob(pattern))
    if not candidates:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    with candidates[0].open() as fh:
        payload = json.load(fh)
    fold_keys = sorted(
        (k for k in payload if k.startswith("fold_")),
        key=lambda k: int(k.split("_", 1)[1]),
    )
    return [payload[k]["f1"] * 100 for k in fold_keys]


def state_means(perfold_dir: Path, state_prefix: str) -> dict[str, float] | None:
    out: dict[str, float] = {}
    for engine in ENGINES:
        folds = load_perfold(perfold_dir, state_prefix, engine)
        if len(folds) != 5:
            return None
        out[engine] = sum(folds) / len(folds)
    return out


def collect_states(perfold_dir, requested, strict):
    out = []
    for prefix in requested:
        means = state_means(perfold_dir, prefix)
        if means is None:
            if strict:
                raise FileNotFoundError(
                    f"Missing one or more cells for {prefix} under {perfold_dir}"
                )
            print(f"[warn] skipping {prefix}: not all three cells present.")
            continue
        out.append((prefix, means))
    return out


def render_grouped(states, out_path: Path) -> None:
    n = len(states)
    if n == 0:
        raise RuntimeError("No states with complete cells; nothing to render.")

    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Wider than tall; extra height for delta brackets + below-axis legend.
    fig, ax = plt.subplots(figsize=(8.0, 4.2), dpi=300)

    bar_width = 0.26
    group_spacing = 1.45  # > 3 * bar_width; extra space for right-side bracket
    group_centers = np.arange(n) * group_spacing
    bar_offsets = [-bar_width, 0.0, +bar_width]

    # Order: HGI, pooled, canonical (left → right within group).
    engines_order = ("hgi", "check2hgi_pooled", "check2hgi")
    bar_labels = ["HGI", "POI-pooled Check2HGI", "canonical Check2HGI"]
    colors = ["#bbbbbb", "#7aa6c2", "#1f4e79"]

    canonical_max = max(s[1]["check2hgi"] for s in states)
    # Tight headroom: just enough for the in-plot legend band above bars.
    y_max = 5 * ((int(canonical_max) // 5) + 4)

    # ── bars + per-bar value labels ──────────────────────────────────────────
    for j, engine in enumerate(engines_order):
        vals = [s[1][engine] for s in states]
        positions = group_centers + bar_offsets[j]
        ax.bar(positions, vals, width=bar_width,
               color=colors[j], edgecolor="black", linewidth=0.55,
               label=bar_labels[j])
        for x, v in zip(positions, vals):
            ax.text(x, v + y_max * 0.008, f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.0, fontweight="bold", color="#222222")

    # ── delta brackets on the right side of each group ──────────────────────
    # Three horizontal dotted guide lines at hgi/pooled/canonical heights,
    # plus a vertical bracket showing per-visit and training-signal gaps.
    bracket_x_off = bar_width * 1.55  # how far right of canonical bar
    tick_len = bar_width * 0.18
    for i, (_, means) in enumerate(states):
        canonical = means["check2hgi"]
        hgi = means["hgi"]
        pooled = means["check2hgi_pooled"]
        per_visit_gap = canonical - pooled
        train_gap = pooled - hgi
        total_gap = canonical - hgi
        per_visit_pct = 100.0 * per_visit_gap / total_gap if total_gap > 0 else 0.0

        gx = group_centers[i] + bar_offsets[2]  # canonical bar centre
        x_left = gx - bar_width / 2
        x_right = gx + bracket_x_off

        # dotted guide lines from canonical bar's right edge to bracket
        for y, c in ((hgi, "#888888"), (pooled, "#5a8aa8"), (canonical, "#1f4e79")):
            ax.plot([x_left, x_right], [y, y],
                    linestyle=":", linewidth=0.7, color=c, zorder=2)

        # vertical bracket spine
        ax.plot([x_right, x_right], [hgi, canonical],
                linestyle="-", linewidth=0.9, color="#333333", zorder=3)
        # ticks at hgi / pooled / canonical
        for y in (hgi, pooled, canonical):
            ax.plot([x_right - tick_len, x_right + tick_len], [y, y],
                    linestyle="-", linewidth=0.9, color="#333333", zorder=3)

        # delta labels to the right of the bracket
        x_lbl = x_right + tick_len + 0.02
        ax.text(x_lbl, (pooled + canonical) / 2,
                f"+{per_visit_gap:.1f}\n({per_visit_pct:.0f}%)",
                ha="left", va="center",
                fontsize=7.5, fontweight="bold", color="#1f4e79",
                linespacing=0.95)
        ax.text(x_lbl, (hgi + pooled) / 2,
                f"+{train_gap:.1f}",
                ha="left", va="center",
                fontsize=7.0, color="#5a8aa8")

    # ── axes cosmetics ───────────────────────────────────────────────────────
    ax.set_xticks(group_centers)
    ax.set_xticklabels([STATE_LABEL[p] for p, _ in states],
                       fontsize=10, fontweight="bold")
    ax.set_ylabel("next-category macro-F1 (%)", fontsize=10)
    ax.set_ylim(0, y_max)
    # extra right margin so the rightmost bracket + label fit
    ax.set_xlim(group_centers[0] - 0.55, group_centers[-1] + 0.85)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend at the top inside the plot frame; extra y-headroom keeps it
    # clear of the canonical bar value labels and delta brackets.
    ax.legend(loc="upper center", fontsize=9, frameon=True,
              ncol=3, bbox_to_anchor=(0.5, 0.99),
              handlelength=1.4, columnspacing=2.0,
              edgecolor="#cccccc", facecolor="white", framealpha=0.95)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes; "
          f"{n} states, grouped layout)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    repo = here.parents[3]
    p.add_argument(
        "--perfold-dir", type=Path,
        default=repo / "docs" / "studies" / "check2hgi" / "results" / "phase1_perfold",
        help="Directory with the per-fold JSONs.",
    )
    p.add_argument(
        "--out", type=Path, default=here / "per-visit.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--states", nargs="+",
        default=["AL", "AZ", "FL", "CA", "TX"],
        help="State prefixes in render order.",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Error out if any requested state is missing data.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    states = collect_states(args.perfold_dir, args.states, args.strict)
    if not states:
        raise SystemExit(
            f"No states with all three cells found under {args.perfold_dir}."
        )
    render_grouped(states, args.out)
