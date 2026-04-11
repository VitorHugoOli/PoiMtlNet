"""CLI entrypoint for checkpoint evaluation (Phase 6).

Usage:
    python scripts/evaluate.py --checkpoint results/dgi/florida/model/best.pt \\
        --state florida --engine dgi

Loads a saved checkpoint and evaluates it on the validation set.
All imports use final Phase 5 canonical paths.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.experiment import ExperimentConfig
from configs.paths import EmbeddingEngine, IoPaths
from data.folds import FoldCreator, TaskType
from training.evaluate import collect_predictions, build_report

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_VALID_ENGINES = [e.value for e in EmbeddingEngine]


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the validation fold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to .pt checkpoint file (state dict).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json saved alongside results (if not provided, "
             "defaults to <results_dir>/config.json).",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Dataset state (used to locate data when --config is absent).",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=_VALID_ENGINES,
        help="Embedding engine (used to locate data when --config is absent).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mtl",
        choices=("mtl", "category", "next"),
        help="Which task head to evaluate.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Which fold index to evaluate (0-based).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)

    # Load config
    config_path = args.config
    if config_path is None:
        # Look for config.json next to checkpoint or in parent results dir
        for candidate in (
            checkpoint_path.parent / "config.json",
            checkpoint_path.parent.parent / "config.json",
        ):
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path is not None:
        config = ExperimentConfig.load(config_path)
        logger.info("Loaded config from %s", config_path)
    else:
        if args.state is None or args.engine is None:
            logger.error("Provide --config or both --state and --engine.")
            sys.exit(1)
        config = ExperimentConfig.default_mtl(
            name=f"eval_{args.state}_{args.engine}",
            state=args.state,
            embedding_engine=args.engine,
        )

    engine = EmbeddingEngine(config.embedding_engine)
    task = args.task

    # Create fold data
    task_type_map = {"mtl": TaskType.MTL, "category": TaskType.CATEGORY, "next": TaskType.NEXT}
    creator = FoldCreator(
        task_type=task_type_map[task],
        n_splits=config.k_folds,
        batch_size=config.batch_size,
        seed=config.seed,
        use_weighted_sampling=False,
    )
    fold_results = creator.create_folds(config.state, engine)

    if args.fold >= len(fold_results):
        logger.error("--fold %d out of range (max: %d)", args.fold, len(fold_results) - 1)
        sys.exit(1)

    fold = fold_results[args.fold]

    # Load checkpoint and model
    from configs.globals import DEVICE
    from models.registry import create_model

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model = create_model(config.model_name, **config.model_params).to(DEVICE)
    model.load_state_dict(state_dict)
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Evaluate
    if task == "mtl":
        # MTLnet.forward expects (category_input, next_input) where the
        # category input is 2D (B, feature_size) and the next input is 3D
        # (B, seq_length, feature_size). The two heads are fully independent
        # inside forward() — encoders, FiLM modulation, shared layers, and
        # the heads themselves are computed on each task tensor separately
        # (see src/models/mtlnet.py:173-214) — so passing a zero-tensor
        # dummy on the unused side is numerically safe and only affects the
        # output of the dummy side, which we discard.
        #
        # We sample one real batch from each loader to capture the exact
        # tensor shapes the model was trained against, then evaluate each
        # head over its own validation loader with shape-correct dummies on
        # the other side.
        cat_sample = next(iter(fold.category.val.dataloader))[0]
        next_sample = next(iter(fold.next.val.dataloader))[0]
        feature_size = cat_sample.shape[-1]
        seq_length = next_sample.shape[1]

        # collect_predictions() iterates the loader, unpacks (X_batch, y_batch),
        # transfers X_batch to device, then calls forward_fn(model, X_batch).
        # So `batch` here is ALREADY the input tensor — not a (X, y) tuple.
        # Indexing batch[0] would slice into the first sample, which is the
        # exact bug the original simplified-forward path had.
        def _mtl_cat_forward(model, cat_x):
            next_dummy = torch.zeros(
                cat_x.size(0), seq_length, feature_size,
                dtype=cat_x.dtype, device=DEVICE,
            )
            out_cat, _ = model((cat_x, next_dummy))
            return out_cat

        def _mtl_next_forward(model, next_x):
            cat_dummy = torch.zeros(
                next_x.size(0), feature_size,
                dtype=next_x.dtype, device=DEVICE,
            )
            _, out_next = model((cat_dummy, next_x))
            return out_next

        cat_preds, cat_targets = collect_predictions(
            model, fold.category.val.dataloader, DEVICE,
            forward_fn=_mtl_cat_forward,
        )
        next_preds, next_targets = collect_predictions(
            model, fold.next.val.dataloader, DEVICE,
            forward_fn=_mtl_next_forward,
        )
        cat_report = build_report(cat_preds, cat_targets)
        next_report = build_report(next_preds, next_targets)
        logger.info(
            "Evaluation report (fold %d, task=mtl, head=category):\n%s",
            args.fold, cat_report,
        )
        logger.info(
            "Evaluation report (fold %d, task=mtl, head=next):\n%s",
            args.fold, next_report,
        )
        return

    if task == "category":
        loader = fold.category.val.dataloader
        preds, targets = collect_predictions(model, loader, DEVICE)
    else:
        loader = fold.next.val.dataloader
        preds, targets = collect_predictions(model, loader, DEVICE)

    report = build_report(preds, targets)
    logger.info("Evaluation report (fold %d, task=%s):\n%s", args.fold, task, report)


if __name__ == "__main__":
    main()
