"""Trainable-parameter counts for the v18 arms, recomputed from the code that ran them.

WHY THIS SCRIPT EXISTS
----------------------
The parameter audit of 2026-07-23 produced the six counts printed in Appendix G of the
supplementary volume and the two figures printed on p.73 of the main volume. Its own output
file says it used the "ceiling config"
(docs/results/closing_data/capacity_matched_stl_cat/alabama_h672/param_audit_pilot.txt):

    STL cat next_gru hidden=256 (ceiling config): 644359

It did not. It built the head with the MODULE default depth, not the config's. Every printed
count on the dedicated-category axis was therefore low by a factor of about 2.2, and the arm
labelled "capacity-matched, 100.2%" in fact carried 229.5% of the joint model's budget.

wrapup/NEW_VERSION.md §10.6 caught this in prose on 2026-08-20 and was right, but it left no
script and no artifact behind, so the correction could not be checked -- which is the same
failure the audit had. This script is that artifact. It recomputes every disputed number from
the code that ran the cells, needs no GPU, loads no data, and trains nothing.

WHAT DECIDES THE CATEGORY AXIS
------------------------------
NextHeadGRU is a GRU plus a LayerNorm+Linear classifier, so its count is a closed form and the
six disputed numbers are the same architecture at two depths and nothing else:

    hidden=256  layers=2 ->    644,359      layers=4 ->  1,433,863
    hidden=672  layers=2 ->  4,207,399      layers=4 ->  9,634,471
    hidden=752  layers=2 ->  5,249,719      layers=4 -> 12,044,791

Which depth the cells ran is not a matter of opinion:

  1. The cells ran `--task next --model next_gru`, width via `--model-param hidden_dim=<w>`,
     "nothing else changed" -- docs/results/closing_data/capacity_matched_stl_cat/README.md:17.
  2. scripts/train.py:339 maps task "next" -> ExperimentConfig.default_next.
  3. src/configs/experiment.py:530 -- default_next pins "num_layers": 4.
  4. scripts/train.py:1454-1460 -- without --replace-model-params, --model-param UPDATES the
     config's model_params, so num_layers=4 survives the width override.

Four layers. The published numbers are the two-layer build.

WHAT DECIDES THE REGION AXIS
----------------------------
The region arm is built by scripts/p1_region_head_ablation.py with `--heads next_stan_flow
--override-hparams freeze_alpha=True alpha_init=0.0` and no d_model, so d_model is the head's
default 128 (docs/studies/closing_data/v18/run_wave.sh, cell_reg). Its count scales with the
region vocabulary, which is why the joint model's total moves across datasets at all.

THE CROSS-CHECK THAT VALIDATES BOTH SIDES AT ONCE
-------------------------------------------------
The delivered joint logs record the optimizer's own partition, e.g.

    [('cat', 8e-05, 1731079), ('reg', 0.00012, 1835982), ('shared', 4e-05, 1584128)]

`cat` and `shared` are constant across every delivered dataset; only `reg` moves. Subtracting
the dedicated region head computed here from the joint's region partition gives the SAME
constant 465,297 at all five datasets that have a delivered log. A single arithmetic slip on
either side would break that identity, so it validates the log-derived joint counts and the
head instantiated here simultaneously.

Run:  PYTHONPATH=src python research/reproducibility/mobiwac_v18/param_counts.py
"""

from __future__ import annotations

import sys

# Region vocabulary per dataset, from the delivered Table 8
# (articles/dissertacao/src/tables/mobiwac/datasets.tex, Regions column).
REGIONS = {
    "istanbul": 520,
    "alabama": 1109,
    "arizona": 1547,
    "florida": 4703,
    "texas": 6553,
    "california": 8501,
}

# Joint-model partitions read from the delivered execution logs,
# docs/results/closing_data/v18_2/modal_runs/<state>_s7_lane_*/logs/<state>_s7_joint.out.
# Alabama has no delivered log in this checkout; its total comes from the 2026-07-23 audit,
# whose JOINT figures are sound (California's audit total equals its log total exactly).
JOINT_LOG = {
    "istanbul": (1731079, 806433, 1584128),
    "arizona": (1731079, 938916, 1584128),
    "texas": (1731079, 1584690, 1584128),
    "california": (1731079, 1835982, 1584128),
}
JOINT_TOTAL_AUDITED = {"alabama": 4197621}

EMBED_DIM = 64  # --embedding-dim 64, pinned by both drivers


def gru_head_params(embed: int, hidden: int, layers: int, classes: int = 7) -> int:
    """Closed form for NextHeadGRU: nn.GRU + LayerNorm + Linear.

    Per GRU layer: W_ih (3h x in), W_hh (3h x h), b_ih (3h), b_hh (3h).
    Classifier: LayerNorm(h) = 2h, Linear(h, C) = h*C + C. Dropout has no parameters.
    """
    n = 0
    for layer in range(layers):
        in_dim = embed if layer == 0 else hidden
        n += 3 * hidden * in_dim + 3 * hidden * hidden + 6 * hidden
    return n + 2 * hidden + hidden * classes + classes


def region_head_params(num_regions: int) -> int:
    """Instantiate the region head exactly as cell_reg() does and count trainable params."""
    import importlib

    import torch

    mod = importlib.import_module("models.next.next_stan_flow.head")
    cls = next(
        v
        for v in vars(mod).values()
        if isinstance(v, type)
        and issubclass(v, torch.nn.Module)
        and v.__module__ == mod.__name__
    )
    head = cls(
        embed_dim=EMBED_DIM,
        num_classes=num_regions,
        d_model=128,          # head default; cell_reg passes no d_model
        freeze_alpha=True,    # --override-hparams freeze_alpha=True
        alpha_init=0.0,       # --override-hparams alpha_init=0.0
    )
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def main() -> int:
    ok = True

    print("CATEGORY AXIS -- NextHeadGRU, closed form, embed_dim=64")
    print(f"  {'hidden':>7}{'layers=2':>14}{'layers=4':>14}   printed until 2026-09-02")
    for hidden, printed in ((256, 644359), (672, 4207399), (752, 5249719)):
        two = gru_head_params(EMBED_DIM, hidden, 2)
        four = gru_head_params(EMBED_DIM, hidden, 4)
        good = two == printed
        ok &= good
        print(f"  {hidden:>7}{two:>14,}{four:>14,}   {printed:,} {'(= layers=2)' if good else 'MISMATCH'}")
    print("  The cells ran layers=4 (see the docstring's four-step chain), so the")
    print("  right-hand column is what they carried and the printed column was the audit's error.\n")

    dedicated_cat = gru_head_params(EMBED_DIM, 256, 4)

    print("REGION AXIS -- NextHeadStanFlow instantiated as cell_reg() builds it")
    print(f"  {'dataset':<12}{'regions':>9}{'dedicated':>12}{'joint reg':>12}{'difference':>12}")
    diffs = set()
    ded_reg = {}
    for state, regions in REGIONS.items():
        ded = region_head_params(regions)
        ded_reg[state] = ded
        if state in JOINT_LOG:
            joint_reg = JOINT_LOG[state][1]
        elif state in JOINT_TOTAL_AUDITED:
            cat_c, _, sh_c = JOINT_LOG["california"]
            joint_reg = JOINT_TOTAL_AUDITED[state] - cat_c - sh_c
        else:
            print(f"  {state:<12}{regions:>9,}{ded:>12,}{'--':>12}{'--':>12}")
            continue
        diff = joint_reg - ded
        diffs.add(diff)
        print(f"  {state:<12}{regions:>9,}{ded:>12,}{joint_reg:>12,}{diff:>12,}")

    if len(diffs) == 1:
        print(f"  Identity holds: the joint's region pathway is the dedicated head + {diffs.pop():,}")
        print("  at every dataset. Both sides are validated by the same check.\n")
    else:
        ok = False
        print(f"  BROKEN: the difference is not constant ({sorted(diffs)}).\n")

    print("TWO DEDICATED MODELS COMBINED, against the joint model")
    print(f"  {'dataset':<12}{'joint':>12}{'cat':>12}{'reg':>12}{'sum':>12}{'ratio':>8}")
    for state, joint_total in (
        ("alabama", JOINT_TOTAL_AUDITED["alabama"]),
        ("california", sum(JOINT_LOG["california"])),
    ):
        s = dedicated_cat + ded_reg[state]
        print(
            f"  {state:<12}{joint_total:>12,}{dedicated_cat:>12,}"
            f"{ded_reg[state]:>12,}{s:>12,}{joint_total / s:>7.2f}x"
        )
    print("  The main volume printed 1.1 million and 2.0 million for these sums. Those close")
    print("  against the withdrawn 644,359; the dedicated category model alone exceeds them.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
