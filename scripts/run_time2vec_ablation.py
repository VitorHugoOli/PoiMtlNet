"""Three-variant ablation: Time2Vec paper-faithfulness study on Alabama.

Compares the current Time2Vec embedding pipeline to two paper-aligned variants
recommended by plans/time2vec_paper_analysis.md:

    baseline_current : legacy absolute-time contrastive sampling (current default)
    rand_init        : frozen random-init, no training (Rec 2 — measures how
                       much downstream F1 is inductive bias vs. training)
    feat_space       : wrap-aware contrastive sampling in (hour/24, dow/7)
                       space (Rec 1 — resolves D6 Issue A)

For each variant: build the embedding parquet, swap it in, regenerate the
next-task input, train MTLnet next-task (--epochs/--folds configurable), and
archive the results. Finally, parse per-fold macro F1 and write a comparison
table to plans/time2vec_ablation_results.md.

Usage:
    python scripts/run_time2vec_ablation.py --epochs 25 --folds 2
    python scripts/run_time2vec_ablation.py --skip_embed  # reuse .variants/
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research"))

from configs.globals import DEVICE
from configs.model import InputsConfig
from configs.paths import EmbeddingEngine, IoPaths
from data.inputs.builders import generate_next_input_from_checkins
from embeddings.time2vec.time2vec import create_embedding

STATE = "alabama"
EMB_PATH = IoPaths.get_embedd(STATE, EmbeddingEngine.TIME2VEC)
VARIANT_DIR = PROJECT_ROOT / "output" / "time2vec" / STATE / ".variants"
RESULTS_BASE = PROJECT_ROOT / "results" / "time2vec"
INPUT_DIR = PROJECT_ROOT / "output" / "time2vec" / STATE / "input"
TEMP_DIR = PROJECT_ROOT / "output" / "time2vec" / STATE / "temp"
FOLDS_DIR = PROJECT_ROOT / "output" / "time2vec" / STATE / "folds"

def _base_t2v_config(embed_epochs: int) -> Namespace:
    """Default Namespace matching pipelines/embedding/time2vec.pipe.py."""
    return Namespace(
        dim=InputsConfig.EMBEDDING_DIM,
        out_features=64,
        activation="sin",
        lr=1e-3,
        epoch=embed_epochs,
        batch_size=2048,
        r_pos_hours=1.0,
        r_neg_hours=24.0,
        max_pairs=2_000_000,
        k_neg_per_i=5,
        max_pos_per_i=20,
        seed=42,
        tau=0.3,
        device=DEVICE,
        compile=False,  # torch.compile has a warm-up cost; skip for ablation
        sampling_mode="absolute_time",
        r_pos_feat=0.03,
        r_neg_feat=0.30,
        no_train=False,
    )


# Variant name → dict of overrides applied to the base config
VARIANTS: dict[str, dict] = {
    "baseline_current": {},
    "rand_init": {"no_train": True},
    "feat_space": {"sampling_mode": "feat_space"},
}


def build_variant(variant: str, overrides: dict, embed_epochs: int) -> None:
    """Train a Time2Vec variant and archive its embedding parquet.

    Calls create_embedding(...) directly (same as pipelines/embedding/time2vec.pipe.py)
    then copies the resulting embeddings.parquet into .variants/.
    """
    print(f"\n  → building variant: {variant}")
    cfg = _base_t2v_config(embed_epochs)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    create_embedding(state=STATE, args=cfg)

    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    dst = VARIANT_DIR / f"embeddings_{variant}.parquet"
    shutil.copy(EMB_PATH, dst)
    print(f"  → archived embedding to {dst.name}")


def swap_in(variant: str) -> None:
    src = VARIANT_DIR / f"embeddings_{variant}.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing variant: {src}")
    print(f"  → swapping in {src.name}")
    shutil.copy(src, EMB_PATH)


def regenerate_inputs() -> None:
    """Wipe stale next-task caches and regenerate them from the swapped-in embedding."""
    print("  → regenerating next-task input")
    for d in (FOLDS_DIR, INPUT_DIR, TEMP_DIR):
        if d.exists():
            shutil.rmtree(d)
    generate_next_input_from_checkins(STATE, EmbeddingEngine.TIME2VEC)


def train_mtlnet(epochs: int, folds: int) -> None:
    print(f"  → training MTLnet next-task (epochs={epochs}, folds={folds})")
    cmd = [
        sys.executable, "scripts/train.py",
        "--task", "next",
        "--state", STATE,
        "--engine", "time2vec",
        "--epochs", str(epochs),
        "--folds", str(folds),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def archive_results(variant: str) -> None:
    src = RESULTS_BASE / STATE
    dst = RESULTS_BASE / f"{STATE}_{variant}"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  → archived results to {dst}")


def parse_f1(variant: str) -> tuple[float, float, list[float]]:
    """Extract per-fold macro F1 from the latest next_* run under the variant dir."""
    base = RESULTS_BASE / f"{STATE}_{variant}"
    run_dirs = sorted(base.glob("next_*"))
    if not run_dirs:
        raise FileNotFoundError(f"No next_* run dir under {base}")
    latest = run_dirs[-1]
    report_files = sorted((latest / "folds").glob("fold*_next_report.json"))
    if not report_files:
        raise FileNotFoundError(f"No fold*_next_report.json under {latest}/folds")
    f1s: list[float] = []
    for f in report_files:
        with f.open() as fh:
            data = json.load(fh)
        f1s.append(float(data["macro avg"]["f1-score"]))
    mean = statistics.mean(f1s)
    std = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
    return mean, std, f1s


def write_results_md(rows: list[dict], epochs: int, folds: int) -> Path:
    out = PROJECT_ROOT / "plans" / "time2vec_ablation_results.md"
    baseline = next((r for r in rows if r["variant"] == "baseline_current"), None)
    base_f1 = baseline["f1_mean"] if baseline else None

    lines = [
        "# Time2Vec paper-faithfulness ablation — results",
        "",
        f"Generated by `scripts/run_time2vec_ablation.py` on Alabama next-task, "
        f"{folds} folds × {epochs} epochs. F1 values are macro avg in percent.",
        "",
        "See `plans/time2vec_paper_analysis.md` for the full deviation analysis.",
        "",
        "## Downstream MTLnet next-task Macro F1",
        "",
        "| Variant | F1 mean | F1 std | Per-fold | Δ vs baseline |",
        "|---|---:|---:|---|---:|",
    ]
    for r in rows:
        delta = "—"
        if base_f1 is not None and r["variant"] != "baseline_current":
            delta = f"{(r['f1_mean'] - base_f1) * 100:+.2f}pp"
        per_fold = ", ".join(f"{v * 100:.2f}" for v in r["f1s"])
        lines.append(
            f"| `{r['variant']}` | {r['f1_mean'] * 100:.2f} | "
            f"±{r['f1_std'] * 100:.2f} | {per_fold} | {delta} |"
        )

    lines += [
        "",
        "## Interpretation guide",
        "",
        "- If `rand_init` ≈ `baseline_current` (within ~1pp): the legacy "
        "contrastive training is doing little work — embeddings are driven by "
        "the `sin(αh+βd+b)` inductive bias, not the optimizer. Fix or drop "
        "contrastive training (strong evidence for Rec 3, end-to-end joint).",
        "- If `feat_space` > `baseline_current`: Recommendation 1 wins; promote "
        "feat-space sampling to default in a follow-up.",
        "- If `feat_space` < `baseline_current`: feat-space sampling removes D6 "
        "Issue A but doesn't unlock downstream gains at this scale; consider "
        "Rec 3 next.",
    ]
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=25,
                        help="MTLnet next-task epochs per fold")
    parser.add_argument("--folds", type=int, default=2,
                        help="MTLnet CV folds to run")
    parser.add_argument("--embed_epochs", type=int, default=100,
                        help="Time2Vec training epochs (ignored for --no_train variants)")
    parser.add_argument("--skip_embed", action="store_true",
                        help="Skip variant building; assume .variants/ is already populated")
    args = parser.parse_args()

    if not args.skip_embed:
        print(f"\n{'='*72}\n  Building {len(VARIANTS)} variants (embed_epochs={args.embed_epochs})\n{'='*72}")
        for variant, overrides in VARIANTS.items():
            build_variant(variant, overrides, embed_epochs=args.embed_epochs)
    else:
        missing = [v for v in VARIANTS if not (VARIANT_DIR / f"embeddings_{v}.parquet").exists()]
        if missing:
            raise SystemExit(f"--skip_embed but missing variants: {missing}")
        print("--skip_embed: reusing existing .variants/")

    results: list[dict] = []
    for variant in VARIANTS:
        print(f"\n{'='*72}\n  Variant: {variant}\n{'='*72}")
        swap_in(variant)
        regenerate_inputs()
        train_mtlnet(epochs=args.epochs, folds=args.folds)
        archive_results(variant)
        mean, std, f1s = parse_f1(variant)
        results.append({"variant": variant, "f1_mean": mean, "f1_std": std, "f1s": f1s})
        print(f"  → {variant}: F1 = {mean * 100:.2f} ± {std * 100:.2f}")

    out = write_results_md(results, epochs=args.epochs, folds=args.folds)
    print(f"\n{'='*72}\n  Wrote {out}\n{'='*72}")
    for r in results:
        print(f"  {r['variant']:20s}  F1 = {r['f1_mean'] * 100:.2f} ± {r['f1_std'] * 100:.2f}")


if __name__ == "__main__":
    main()
