"""Render the per-visit context counterfactual at all five states.

Drop-in replacement for the AL-only `render_per_visit.py`. Consumes the
phase1_perfold JSONs produced by `scripts/probe/extract_pervisit_perfold.py`
for FL/CA/TX and the existing AL+AZ counterfactual JSONs.

The figure is a five-panel mini-bar-plot, one panel per state, three bars per
panel (HGI / POI-pooled Check2HGI / canonical Check2HGI), with per-visit /
training-signal brackets and percent annotations. Designed for camera-ready
when FL/CA/TX runs land; falls back to AL+AZ-only if FL/CA/TX JSONs are
missing.

Usage
-----
    python3 articles/[BRACIS]_Beyond_Cross_Task/src/figs/render_per_visit_5state.py \\
        [--perfold-dir docs/studies/check2hgi/results/phase1_perfold] \\
        [--out articles/[BRACIS]_Beyond_Cross_Task/src/figs/per-visit.png] \\
        [--shared-y]    # share y-axis across panels (cleaner if all states have similar magnitude)
        [--two-state]   # force AL+AZ only (skip FL/CA/TX)

By default the script auto-falls-back to AL+AZ if any of FL/CA/TX is missing
either of the three cells. To explicitly render the camera-ready 5-panel
version, pass --strict to error out instead of falling back.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Two-letter prefix used in phase1_perfold filenames.
STATE_PREFIX = {
    "AL": "Alabama (AL)",
    "AZ": "Arizona (AZ)",
    "FL": "Florida (FL)",
    "CA": "California (CA)",
    "TX": "Texas (TX)",
}

ENGINES = ("check2hgi", "check2hgi_pooled", "hgi")


def load_perfold(perfold_dir: Path, state_prefix: str, engine: str) -> list[float]:
    """Return per-fold macro-F1 (as percentages) for (state, engine), or [].

    Matches both timestamped (e.g. AZ_*_20260503.json) and untagged files
    (e.g. AL_check2hgi_cat_gru_5f50ep.json). When multiple matches exist,
    pick the most recently modified.
    """
    pattern = f"{state_prefix}_{engine}_cat_gru_5f50ep*.json"
    candidates = list(perfold_dir.glob(pattern))
    if not candidates:
        return []
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    with candidates[0].open() as fh:
        payload = json.load(fh)
    # Be index-agnostic: AL JSONs are 0-indexed (fold_0..fold_4) while AZ
    # 2026-05-03 JSONs are 1-indexed (fold_1..fold_5). Iterate over whatever
    # fold_* keys exist.
    fold_keys = sorted(
        (k for k in payload if k.startswith("fold_")),
        key=lambda k: int(k.split("_", 1)[1]),
    )
    return [payload[k]["f1"] * 100 for k in fold_keys]


def state_means(perfold_dir: Path, state_prefix: str) -> dict[str, float] | None:
    """Return {engine: mean_macro_f1_pct} for one state, or None if any cell missing."""
    out: dict[str, float] = {}
    for engine in ENGINES:
        folds = load_perfold(perfold_dir, state_prefix, engine)
        if len(folds) != 5:
            return None
        out[engine] = sum(folds) / len(folds)
    return out


def collect_states(perfold_dir: Path, requested: Iterable[str],
                   strict: bool) -> list[tuple[str, dict[str, float]]]:
    """Return [(state_prefix, means), ...] in the requested order.

    Skips states that don't have all three cells; if strict and any is
    missing, raise.
    """
    out: list[tuple[str, dict[str, float]]] = []
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


def render(states: list[tuple[str, dict[str, float]]], out_path: Path,
           shared_y: bool) -> None:
    n = len(states)
    if n == 0:
        raise RuntimeError("no states with complete cells; nothing to render.")

    # Tighter aspect for 5-panel; relax for 2-panel.
    if n == 1:
        figsize = (4.4, 3.6)
    elif n == 2:
        figsize = (8.5, 3.4)
    elif n <= 5:
        figsize = (3.0 * n, 3.4)
    else:
        figsize = (15.0, 3.4)

    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, axes = plt.subplots(
        1, n, figsize=figsize, dpi=300,
        sharey=shared_y,
        gridspec_kw={"wspace": 0.32 if shared_y else 0.40},
    )
    if n == 1:
        axes = [axes]

    bar_labels = ["HGI", "pooled\nCheck2HGI", "canonical\nCheck2HGI"]
    colors = ["#bbbbbb", "#7aa6c2", "#1f4e79"]

    # global y_max for non-shared panels: rounded up to nearest 5 above the max.
    global_max = max(s[1]["check2hgi"] for s in states)
    y_max = 5 * ((int(global_max) // 5) + 2)  # always leaves headroom for brackets

    for ax_idx, (prefix, means) in enumerate(states):
        ax = axes[ax_idx]
        hgi = means["hgi"]
        pooled = means["check2hgi_pooled"]
        canonical = means["check2hgi"]

        gap = canonical - hgi
        per_visit = canonical - pooled
        train_sig = pooled - hgi
        per_visit_pct = 100.0 * per_visit / gap if gap > 0 else 0.0
        train_sig_pct = 100.0 * train_sig / gap if gap > 0 else 0.0

        xs = [0, 1, 2]
        ax.bar(xs, [hgi, pooled, canonical], color=colors,
               edgecolor="black", linewidth=0.6, width=0.55)
        for x, v in zip(xs, [hgi, pooled, canonical]):
            ax.text(x, v + (y_max * 0.012), f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # horizontal reference lines extending right from each bar top
        right_edge = 2.55
        for x, v in zip(xs, [hgi, pooled, canonical]):
            ax.plot([x + 0.275, right_edge], [v, v],
                    color="gray", linestyle=":", lw=0.6)

        # bracket: training-signal (HGI -> pooled)
        br_x = right_edge - 0.05
        ax.annotate("", xy=(br_x, pooled), xytext=(br_x, hgi),
                    arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.9))
        ax.text(br_x + 0.10, (hgi + pooled) / 2,
                f"+{train_sig:.1f}\n({train_sig_pct:.0f}%)",
                fontsize=7.5, va="center", ha="left", color="#444444")

        # bracket: per-visit context (pooled -> canonical)
        ax.annotate("", xy=(br_x, canonical), xytext=(br_x, pooled),
                    arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.1))
        ax.text(br_x + 0.10, (pooled + canonical) / 2,
                f"+{per_visit:.1f}\n({per_visit_pct:.0f}%)",
                fontsize=7.5, va="center", ha="left",
                color="#1f4e79", fontweight="bold")

        ax.set_xticks(xs)
        ax.set_xticklabels(bar_labels, fontsize=7.5)
        ax.set_xlim(-0.5, 4.0)
        ax.set_ylim(0, y_max)
        ax.set_title(STATE_PREFIX.get(prefix, prefix), fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0 or not shared_y:
            ax.set_ylabel("macro-F1 (%)", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes; "
          f"{n} panel{'s' if n != 1 else ''})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).parent
    repo = here.parents[3]
    p.add_argument(
        "--perfold-dir", type=Path,
        default=repo / "docs" / "studies" / "check2hgi" / "results" / "phase1_perfold",
        help="Directory holding the simplified per-fold JSONs.",
    )
    p.add_argument(
        "--out", type=Path, default=here / "per-visit.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--states", nargs="+",
        default=["AL", "AZ", "FL", "CA", "TX"],
        help="State prefixes to render in the requested order.",
    )
    p.add_argument(
        "--two-state", action="store_true",
        help="Force AL+AZ panel (skip any FL/CA/TX entries).",
    )
    p.add_argument(
        "--shared-y", action="store_true",
        help="Share y-axis across panels (cleaner when state magnitudes are similar).",
    )
    p.add_argument(
        "--strict", action="store_true",
        help="Error out if any requested state is missing data; otherwise auto-skip.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    requested = ["AL", "AZ"] if args.two_state else args.states
    states = collect_states(args.perfold_dir, requested, args.strict)
    if not states:
        raise SystemExit(
            f"No states with all three cells found under {args.perfold_dir}. "
            "Did the FL/CA/TX runs complete? Did you run "
            "scripts/probe/extract_pervisit_perfold.py?"
        )
    render(states, args.out, shared_y=args.shared_y)
