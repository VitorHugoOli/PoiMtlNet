"""CLI entrypoint for MTLnet training (Phase 6).

Usage:
    python scripts/train.py --state florida --engine dgi --epochs 1 --folds 1
    python scripts/train.py --state florida --engine dgi --task category
    python scripts/train.py --config experiments/configs/mtl_hgi_florida.py

All imports use final Phase 5 canonical paths.
Notes:
    --folds N: run only the first N folds.  The split structure uses
    max(2, N) splits (StratifiedKFold requires >= 2), but execution stops
    after N folds.  Use this to run a quick smoke test without full CV.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# Ensure src/ is on sys.path when invoked directly.
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.experiment import ExperimentConfig
from configs.globals import CATEGORIES_MAP
from configs.paths import EmbeddingEngine, IoPaths
from data.folds import FoldCreator, TaskType
from training.callbacks import ModelCheckpoint
from utils.seed import seed_everything
from tracking import DatasetHistory, MLHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_VALID_TASKS = ("mtl", "category", "next")
_VALID_ENGINES = [e.value for e in EmbeddingEngine]


def _default_checkpoint_callbacks(
    results_path: Path,
    monitor: str,
    task: str,
) -> list:
    """Build the default callback list for a runner.

    Saves best-so-far model weights (by ``monitor``) to a per-invocation
    timestamped subdir under ``<results_path>/checkpoints/``. The seconds
    precision in the directory name guarantees concurrent runs do not
    overwrite each other's checkpoints. Required so ``scripts/evaluate.py``
    has something to load after training.
    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(results_path) / "checkpoints" / f"{task}_{ts}"
    return [
        ModelCheckpoint(
            save_dir=ckpt_dir,
            monitor=monitor,
            mode="max",
            save_best_only=True,
        )
    ]


# ---------------------------------------------------------------------------
# Runner dispatch
# ---------------------------------------------------------------------------

def _run_mtl(config: ExperimentConfig, results_path: Path, fold_results: dict) -> dict:
    from training.runners.mtl_cv import train_with_cross_validation

    history = MLHistory(
        model_name="MTLNet",
        tasks={"next", "category"},
        num_folds=len(fold_results),
        datasets={
            DatasetHistory(
                raw_data=IoPaths.get_next(config.state, EmbeddingEngine(config.embedding_engine)),
                folds_signature=None,
                description="Next-POI prediction input",
            ),
            DatasetHistory(
                raw_data=IoPaths.get_category(
                    config.state, EmbeddingEngine(config.embedding_engine)
                ),
                folds_signature=None,
                description="Category prediction input",
            ),
        },
        label_map=CATEGORIES_MAP,
        save_path=results_path,
        verbose=True,
    )

    with history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            config=config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(
                results_path, monitor="val_f1_category", task="mtl",
            ),
        )

    return results


def _run_category(config: ExperimentConfig, results_path: Path, fold_results: dict) -> dict:
    from training.runners.category_cv import run_cv

    folds = [
        (fold_results[i].category.train.dataloader, fold_results[i].category.val.dataloader)
        for i in sorted(fold_results)
    ]

    history = MLHistory(
        model_name="Category",
        model_type="Single-Task",
        tasks="category",
        num_folds=len(folds),
        datasets={
            DatasetHistory(
                raw_data=str(
                    IoPaths.get_category(config.state, EmbeddingEngine(config.embedding_engine))
                ),
                folds_signature=None,
                description="Category prediction input",
            )
        },
        label_map=CATEGORIES_MAP,
        save_path=str(results_path),
        verbose=True,
        display_report=True,
    )

    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(
                results_path, monitor="val_f1", task="category",
            ),
        )

    history.display.end_training()
    return results


def _run_next(config: ExperimentConfig, results_path: Path, fold_results: dict) -> dict:
    from training.runners.next_cv import run_cv

    folds = [
        (fold_results[i].next.train.dataloader, fold_results[i].next.val.dataloader)
        for i in sorted(fold_results)
    ]

    history = MLHistory(
        model_name="Next",
        model_type="Single-Task",
        tasks="next",
        num_folds=len(folds),
        datasets={
            DatasetHistory(
                raw_data=str(
                    IoPaths.get_next(config.state, EmbeddingEngine(config.embedding_engine))
                ),
                folds_signature=None,
                description="Next-POI prediction input",
            )
        },
        label_map=CATEGORIES_MAP,
        save_path=str(results_path),
        verbose=True,
        display_report=True,
    )

    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(
                results_path, monitor="val_f1", task="next",
            ),
        )

    history.display.end_training()
    return results


_RUNNERS = {
    "mtl": _run_mtl,
    "category": _run_category,
    "next": _run_next,
}

_TASK_TYPES = {
    "mtl": TaskType.MTL,
    "category": TaskType.CATEGORY,
    "next": TaskType.NEXT,
}

_DEFAULT_FACTORIES = {
    "mtl": ExperimentConfig.default_mtl,
    "category": ExperimentConfig.default_category,
    "next": ExperimentConfig.default_next,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MTLnet model via cross-validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Dataset state (e.g. florida, alabama). Required unless --config is provided.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=_VALID_ENGINES,
        help="Embedding engine. Required unless --config is provided.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mtl",
        choices=_VALID_TASKS,
        help="Task type to train (mtl / category / next).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Overrides config value.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=None,
        help=(
            "Number of CV folds to RUN (not the split count). "
            "Use 1 for a quick smoke test. "
            "The split structure uses max(2, N) splits."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to an experiment config file that exports config() -> ExperimentConfig. "
            "CLI flags override values from the file."
        ),
    )
    return parser.parse_args(argv)


def _load_config_from_file(path: str) -> ExperimentConfig:
    """Load ExperimentConfig from a Python file that exports config()."""
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_exp_cfg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "config"):
        raise AttributeError(f"Config file {path!r} must export a config() function.")
    return mod.config()


def main(argv=None) -> None:
    args = _parse_args(argv)

    # Build config
    if args.config is not None:
        config = _load_config_from_file(args.config)
    else:
        if args.state is None or args.engine is None:
            print(
                "error: --state and --engine are required when --config is not provided.",
                file=sys.stderr,
            )
            sys.exit(1)
        factory = _DEFAULT_FACTORIES[args.task]
        config = factory(
            name=f"{args.task}_{args.state}_{args.engine}",
            state=args.state,
            embedding_engine=args.engine,
        )

    # Apply CLI overrides
    import dataclasses

    if args.state is not None:
        config = dataclasses.replace(config, state=args.state)
    if args.engine is not None:
        config = dataclasses.replace(config, embedding_engine=args.engine)
    if args.epochs is not None:
        config = dataclasses.replace(config, epochs=args.epochs)
    if args.config is not None and args.task != config.task_type:
        config = dataclasses.replace(config, task_type=args.task)

    # --folds: limits execution, doesn't change split structure.
    # StratifiedKFold requires n_splits >= 2, so we use max(2, requested).
    max_folds = args.folds  # None means run all folds
    if max_folds is not None:
        n_splits = max(2, max_folds)
        config = dataclasses.replace(config, k_folds=n_splits)

    seed_everything(config.seed)

    engine = EmbeddingEngine(config.embedding_engine)
    task_key = config.task_type if config.task_type in _RUNNERS else "mtl"

    # Create folds
    creator = FoldCreator(
        task_type=_TASK_TYPES[task_key],
        n_splits=config.k_folds,
        batch_size=config.batch_size,
        seed=config.seed,
        use_weighted_sampling=False,
    )
    fold_results = creator.create_folds(config.state, engine)

    # Apply max_folds limit (run only first N folds).
    # config.k_folds stays as the split structure count (>= 2);
    # runners use len(fold_results) to determine actual execution count.
    if max_folds is not None and max_folds < len(fold_results):
        fold_results = dict(list(fold_results.items())[:max_folds])

    results_path = IoPaths.get_results_dir(config.state, engine)
    results_path.mkdir(parents=True, exist_ok=True)

    # Save config alongside results
    config.save(results_path / "config.json")

    logger.info("=" * 72)
    logger.info(
        "Training: state=%s  engine=%s  task=%s  epochs=%d  folds=%d",
        config.state,
        config.embedding_engine,
        task_key,
        config.epochs,
        len(fold_results),
    )
    logger.info("=" * 72)

    runner = _RUNNERS[task_key]
    runner(config, results_path, fold_results)

    logger.info("Done. Results written to: %s", results_path)


if __name__ == "__main__":
    main()
