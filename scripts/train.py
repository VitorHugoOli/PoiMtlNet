"""CLI entrypoint for MTLnet training (Phase 6).

Usage:
    python scripts/train.py --state florida --engine dgi --epochs 1 --folds 1
    python scripts/train.py --state florida --engine dgi --task category
    python scripts/train.py --state alabama --engine dgi --candidate cgc_equal
    python scripts/train.py --state alabama --engine dgi --mtl-loss static_weight --category-weight 0.25
    python scripts/train.py --config experiments/configs/mtl_hgi_florida.py

All imports use final Phase 5 canonical paths.
Notes:
    --folds N: run only the first N folds.  The split structure uses
    max(2, N) splits (StratifiedKFold requires >= 2), but execution stops
    after N folds.  Use this to run a quick smoke test without full CV.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Ensure repo root and src/ are on sys.path when invoked directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
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


def _make_run_dir(results_path: Path, task: str, config: ExperimentConfig) -> Path:
    """Create a per-invocation run directory and persist its config.

    Returns a unique subdirectory under ``<results_path>/checkpoints/`` named
    ``<task>_<timestamp>`` with the run's ``config.json`` already written
    inside it. The timestamp format is ``%Y%m%d_%H%M%S_<pid>`` so two runs
    of the same task started in the same second cannot collide
    (a sub-second resolution alone would still race; appending the PID
    is a cheap absolute guarantee).

    Saving the config here (before training starts) means the canonical
    per-run config record is preserved even if training crashes mid-fold.
    ``scripts/evaluate.py`` auto-discovers the file via ``parent /
    config.json`` (see ``scripts/evaluate.py`` config-discovery loop).
    """
    import os
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(results_path) / "checkpoints" / f"{task}_{ts}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.json")
    return run_dir


def _default_checkpoint_callbacks(run_dir: Path, monitor: str) -> list:
    """Build the default callback list for a runner.

    Wires a single ``ModelCheckpoint`` that watches ``monitor`` (a key in
    ``CallbackContext.metrics``, see ``src/training/callbacks.py``) and
    writes best-so-far weights into ``run_dir``. ``run_dir`` should come
    from :func:`_make_run_dir` so the checkpoints land alongside the
    run's ``config.json``.

    The monitor strings live in this file's call sites and must match
    the keys actually emitted by each runner:
        mtl_cv.py:241-247    → val_f1_next, val_f1_category, val_loss, ...
        category_cv.py / next_cv.py → val_f1, val_loss, val_acc, ...
    A typo here is silently dropped by ModelCheckpoint (`current is None
    → return`) and no checkpoint is ever saved.
    """
    return [
        ModelCheckpoint(
            save_dir=run_dir,
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

    run_dir = _make_run_dir(results_path, task="mtl", config=config)
    with history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            config=config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(run_dir, monitor="val_f1_category"),
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

    run_dir = _make_run_dir(results_path, task="category", config=config)
    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(run_dir, monitor="val_f1"),
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

    run_dir = _make_run_dir(results_path, task="next", config=config)
    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=_default_checkpoint_callbacks(run_dir, monitor="val_f1"),
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
        default=None,
        choices=_VALID_TASKS,
        help="Task type to train (mtl / category / next). Defaults to mtl.",
    )
    parser.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=None,
        help=(
            "Registered model name to use. For Phase 1/2 MTL experiments, "
            "examples are 'mtlnet', 'mtlnet_mmoe', 'mtlnet_cgc', and "
            "'mtlnet_dselectk'."
        ),
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help=(
            "Named MTL candidate from src/ablation/candidates.py, e.g. "
            "equal_weight, baseline_nash, cgc_equal, or cgc_famo. "
            "Explicit --model, --model-param, --mtl-loss, and "
            "--mtl-loss-param flags override candidate defaults."
        ),
    )
    parser.add_argument(
        "--model-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a model parameter. May be passed multiple times. "
            "Values are parsed as JSON when possible, so numbers and booleans "
            "keep their type."
        ),
    )
    parser.add_argument(
        "--mtl-loss",
        type=str,
        default=None,
        help=(
            "Registered MTL loss to use, e.g. nash_mtl, equal_weight, "
            "static_weight, uncertainty_weighting, random_weight, rlw, famo, "
            "fairgrad, bayesagg_mtl, go4align, excess_mtl, stch, or db_mtl."
        ),
    )
    parser.add_argument(
        "--mtl-loss-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override an MTL loss parameter. May be passed multiple times. "
            "Values are parsed as JSON when possible."
        ),
    )
    parser.add_argument(
        "--category-weight",
        type=float,
        default=None,
        help=(
            "Convenience parameter for --mtl-loss static_weight. "
            "Sets the category loss weight; next weight is 1 - category_weight."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps.",
    )
    parser.add_argument(
        "--next-target",
        type=str,
        choices=("next_category", "next_poi"),
        default=None,
        help="Target interface marker for next-task experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override for reproducibility checks.",
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
        "--embedding-dim",
        type=int,
        default=None,
        help=(
            "Embedding dimension. Overrides model_params feature_size/input_dim/embed_dim. "
            "Required for engines with non-default dimensions (e.g. fusion=128)."
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
    parser.add_argument(
        "--replace-model-params",
        action="store_true",
        default=False,
        help=(
            "When set, --model-param values REPLACE the default model_params "
            "instead of merging. Useful when switching to a model with a "
            "different constructor signature."
        ),
    )
    return parser.parse_args(argv)


def _coerce_cli_value(raw: str):
    """Parse CLI override values while keeping strings usable."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _parse_key_value_overrides(items: list[str], option_name: str) -> dict:
    overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{option_name} expects KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        if not key:
            raise ValueError(f"{option_name} requires a non-empty key")
        overrides[key] = _coerce_cli_value(raw_value)
    return overrides


def _apply_cli_overrides(
    config: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Apply CLI overrides without creating folds or touching the filesystem."""
    if args.state is not None:
        config = dataclasses.replace(config, state=args.state)
    if args.engine is not None:
        config = dataclasses.replace(config, embedding_engine=args.engine)
    if args.task is not None and args.task != config.task_type:
        config = dataclasses.replace(config, task_type=args.task)
    if args.epochs is not None:
        config = dataclasses.replace(config, epochs=args.epochs)
    if args.next_target is not None:
        config = dataclasses.replace(config, next_target=args.next_target)
    if args.seed is not None:
        config = dataclasses.replace(config, seed=args.seed)
    if args.gradient_accumulation_steps is not None:
        if args.gradient_accumulation_steps <= 0:
            raise ValueError("--gradient-accumulation-steps must be > 0")
        config = dataclasses.replace(
            config,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

    if args.candidate is not None:
        if config.task_type != "mtl":
            raise ValueError("--candidate can only be used with --task mtl")
        from ablation.candidates import get_candidate

        try:
            candidate = get_candidate(args.candidate)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        model_params = dict(config.model_params)
        model_params.update(candidate.model_params)
        config = dataclasses.replace(
            config,
            model_name=candidate.model_name,
            model_params=model_params,
            mtl_loss=candidate.mtl_loss,
            mtl_loss_params=dict(candidate.mtl_loss_params),
        )

    model_param_overrides = _parse_key_value_overrides(
        args.model_param, "--model-param"
    )
    if args.model_name is not None:
        config = dataclasses.replace(config, model_name=args.model_name)
    if model_param_overrides:
        if getattr(args, "replace_model_params", False):
            model_params = model_param_overrides
        else:
            model_params = dict(config.model_params)
            model_params.update(model_param_overrides)
        config = dataclasses.replace(config, model_params=model_params)

    loss_param_overrides = _parse_key_value_overrides(
        args.mtl_loss_param, "--mtl-loss-param"
    )
    if args.mtl_loss is not None:
        if config.task_type != "mtl":
            raise ValueError("--mtl-loss can only be used with --task mtl")
        config = dataclasses.replace(
            config,
            mtl_loss=args.mtl_loss,
            mtl_loss_params={},
        )
    if args.category_weight is not None:
        if config.mtl_loss != "static_weight":
            raise ValueError(
                "--category-weight requires --mtl-loss static_weight"
            )
        loss_param_overrides["category_weight"] = args.category_weight
    if loss_param_overrides:
        loss_params = dict(config.mtl_loss_params)
        loss_params.update(loss_param_overrides)
        config = dataclasses.replace(config, mtl_loss_params=loss_params)

    return config


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
        task = args.task or "mtl"
        factory = _DEFAULT_FACTORIES[task]
        config = factory(
            name=f"{task}_{args.state}_{args.engine}",
            state=args.state,
            embedding_engine=args.engine,
        )

    # Apply CLI overrides
    try:
        config = _apply_cli_overrides(config, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # --embedding-dim: override the dimension key in model_params.
    if args.embedding_dim is not None:
        _DIM_KEYS = {
            "mtl": "feature_size",
            "category": "input_dim",
            "next": "embed_dim",
        }
        dim_key = _DIM_KEYS.get(args.task, _DIM_KEYS.get(config.task_type, "feature_size"))
        updated_params = dict(config.model_params)
        updated_params[dim_key] = args.embedding_dim
        config = dataclasses.replace(config, model_params=updated_params)

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

    # DVC tracks `results/<engine>/<state>/config.json` as a metric file
    # (see dvc.yaml). This file is overwritten by every training run on
    # purpose — DVC consumes it as the "latest run snapshot" for dashboards.
    # The CANONICAL per-run config is written into the per-invocation
    # checkpoint dir (see _default_checkpoint_callbacks above), so each
    # individual run's config is preserved alongside its weights and is
    # auto-discovered by scripts/evaluate.py via parent / config.json.
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

    # Validate loss + gradient accumulation compatibility before training.
    _BACKWARD_ONLY_LOSSES = {"nash_mtl", "pcgrad", "gradnorm"}
    grad_accum = getattr(config, "gradient_accumulation_steps", 1) or 1
    if (
        task_key == "mtl"
        and grad_accum > 1
        and config.mtl_loss in _BACKWARD_ONLY_LOSSES
    ):
        logger.error(
            "--mtl-loss %r calls backward() internally and is incompatible "
            "with gradient_accumulation_steps=%d. Use "
            "gradient_accumulation_steps=1 or a loss with get_weighted_loss().",
            config.mtl_loss,
            grad_accum,
        )
        sys.exit(2)

    runner = _RUNNERS[task_key]
    runner(config, results_path, fold_results)

    logger.info("Done. Results written to: %s", results_path)


if __name__ == "__main__":
    main()
