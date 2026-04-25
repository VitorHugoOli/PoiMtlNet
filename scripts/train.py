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
from typing import Dict, Optional

# Ensure repo root and src/ are on sys.path when invoked directly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.experiment import DatasetSignature, ExperimentConfig
from configs.globals import CATEGORIES_MAP
from configs.paths import EmbeddingEngine, IoPaths
from data.folds import FoldCreator, TaskType, load_folds, rebuild_dataloaders
from tasks import (
    CHECK2HGI_NEXT_REGION,
    LEGACY_CATEGORY_NEXT,
    TaskSet,
    get_preset,
    resolve_task_set,
)
from tasks.presets import list_presets
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
_NO_CHECKPOINTS = False


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

    if _NO_CHECKPOINTS:
        cbs = []
    else:
        run_dir = _make_run_dir(results_path, task="mtl", config=config)
        cbs = _default_checkpoint_callbacks(run_dir, monitor="val_f1_category")
    with history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            config=config,
            results_path=results_path,
            callbacks=cbs,
        )

    return results


def _run_mtl_check2hgi(
    config: ExperimentConfig,
    results_path: Path,
    fold_results: dict,
    task_set: TaskSet,
) -> dict:
    """Dispatch MTL runs on the check2HGI preset (or any non-legacy task_set).

    Differs from ``_run_mtl`` on four axes:

    * ``MLHistory.tasks`` uses the preset's slot names
      (``next_category`` / ``next_region`` instead of ``category`` / ``next``).
    * ``DatasetHistory`` points at ``next.parquet`` + ``next_region.parquet``
      (there is no separate category.parquet on this track).
    * ``ModelCheckpoint`` watches ``val_joint_acc1`` — the metric emitted
      by ``mtl_cv.py`` alongside ``val_joint_score`` specifically for
      high-cardinality-head setups where macro-F1 is a weak summary.
    * ``train_with_cross_validation`` receives the resolved ``task_set``.
    """
    from training.runners.mtl_cv import train_with_cross_validation

    engine = EmbeddingEngine(config.embedding_engine)
    history = MLHistory(
        model_name="MTLNet",
        tasks={task_set.task_a.name, task_set.task_b.name},
        num_folds=len(fold_results),
        datasets={
            DatasetHistory(
                raw_data=IoPaths.get_next(config.state, engine),
                folds_signature=None,
                description="next_category input (shared X)",
            ),
            DatasetHistory(
                raw_data=IoPaths.get_next_region(config.state, engine),
                folds_signature=None,
                description="next_region input (shared X, region labels)",
            ),
        },
        label_map=CATEGORIES_MAP,
        save_path=results_path,
        verbose=True,
    )

    if _NO_CHECKPOINTS:
        cbs = []
    else:
        run_dir = _make_run_dir(results_path, task=f"mtl__{task_set.name}", config=config)
        # val_joint_geom_lift = sqrt((acc1_a/maj_a) * (acc1_b/maj_b))
        cbs = _default_checkpoint_callbacks(run_dir, monitor="val_joint_geom_lift")
    with history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            config=config,
            results_path=results_path,
            callbacks=cbs,
            task_set=task_set,
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

    if _NO_CHECKPOINTS:
        cbs = []
    else:
        run_dir = _make_run_dir(results_path, task="category", config=config)
        cbs = _default_checkpoint_callbacks(run_dir, monitor="val_f1")
    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=cbs,
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

    if _NO_CHECKPOINTS:
        cbs = []
    else:
        run_dir = _make_run_dir(results_path, task="next", config=config)
        cbs = _default_checkpoint_callbacks(run_dir, monitor="val_f1")
    with history:
        results = run_cv(
            history, folds, config,
            results_path=results_path,
            callbacks=cbs,
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
    "mtl_check2hgi": TaskType.MTL_CHECK2HGI,
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
        "--task-set",
        type=str,
        default=None,
        choices=list_presets(),
        help=(
            "MTL task-set preset. Only used with --task mtl. Defaults to "
            "legacy_category_next (bit-exact with the pre-parameterisation "
            "runner). check2hgi_next_region activates the 2-task "
            "{next_category, next_region} pair on check2HGI embeddings."
        ),
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
        "--use-class-weights",
        dest="use_class_weights",
        action="store_true",
        default=None,
        help=(
            "Pass class-balanced weights to per-head CrossEntropyLoss. "
            "Recommended for the check2HGI next_region head on Florida "
            "(22% majority-class region would otherwise dominate the "
            "loss and starve the next_category gradient under NashMTL). "
            "Absent classes get weight 1.0 (see src/training/helpers.py)."
        ),
    )
    parser.add_argument(
        "--no-class-weights",
        dest="use_class_weights",
        action="store_false",
        default=None,
        help="Force-disable class-balanced CE weighting even if the config default is on.",
    )
    parser.add_argument(
        "--task-a-input-type",
        type=str,
        choices=("checkin", "region", "concat"),
        default="checkin",
        help=(
            "Check2HGI MTL only: input modality for task-a slot (next_category). "
            "'checkin' (default, bit-exact-legacy) = 9-window of check-in emb; "
            "'region' = 9-window of region emb via placeid→region lookup; "
            "'concat' = [checkin ⊕ region] stacked on feature axis. Used by the "
            "P4 per-task-modality ablation (CH03)."
        ),
    )
    parser.add_argument(
        "--task-b-input-type",
        type=str,
        choices=("checkin", "region", "concat"),
        default="checkin",
        help=(
            "Check2HGI MTL only: input modality for task-b slot (next_region). "
            "See --task-a-input-type. P1 showed region-emb input is the right "
            "modality for the region head (53% Acc@10 vs 20% on check-in)."
        ),
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
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size. Overrides config value.",
    )
    parser.add_argument(
        "--max-lr",
        type=float,
        default=None,
        help="OneCycleLR max_lr. Overrides config value. STL next uses 0.01, STL region GRU 0.003, MTL default 0.001.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=("onecycle", "constant", "cosine", "warmup_constant"),
        default=None,
        help=(
            "LR scheduler type. 'onecycle' (default) preserves legacy "
            "behaviour. 'constant' holds --max-lr fixed (disentangles "
            "'more epochs' from 'stretched OneCycleLR schedule', F45). "
            "'cosine' decays from --max-lr to 0 without warmup. "
            "'warmup_constant' (F48-H2) linearly warms LR over "
            "--pct-start of total steps, then holds --max-lr forever."
        ),
    )
    parser.add_argument(
        "--pct-start",
        type=float,
        default=None,
        help=(
            "OneCycleLR pct_start (warmup fraction). PyTorch default 0.3. "
            "Smaller values push peak LR earlier and leave more epochs in "
            "annealing. Used by F46."
        ),
    )
    parser.add_argument(
        "--cat-lr",
        type=float,
        default=None,
        help=(
            "Per-head LR (F48-H3) — LR for the cat encoder + cat head "
            "param group. When --cat-lr, --reg-lr and --shared-lr are "
            "ALL set, the optimizer is built with three distinct param "
            "groups and --max-lr is ignored. Pair with --scheduler "
            "constant so the per-group LRs survive."
        ),
    )
    parser.add_argument(
        "--reg-lr",
        type=float,
        default=None,
        help="Per-head LR — LR for the next encoder + next head group (F48-H3).",
    )
    parser.add_argument(
        "--shared-lr",
        type=float,
        default=None,
        help=(
            "Per-head LR — LR for the cross-attn + final_ln group (F48-H3). "
            "Default recommendation: shared_lr = reg_lr (cross-attn is in "
            "the reg gradient path; throttling it reproduces F44 not H3)."
        ),
    )
    parser.add_argument(
        "--reg-head-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override region head hyperparameter (task_b_head_params) in the MTL task_set. Repeatable. E.g. --reg-head-param hidden_dim=512 --reg-head-param num_layers=3",
    )
    parser.add_argument(
        "--reg-head",
        type=str,
        default=None,
        help=(
            "Override the region-task head factory (task_b.head_factory) in the MTL task_set. "
            "Default preserves the preset's choice (next_gru for CHECK2HGI_NEXT_REGION). "
            "Use e.g. --reg-head next_stan to swap in the STAN head."
        ),
    )
    parser.add_argument(
        "--cat-head-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override category head hyperparameter (task_a_head_params) in the MTL task_set. Repeatable. Symmetric to --reg-head-param.",
    )
    parser.add_argument(
        "--cat-head",
        type=str,
        default=None,
        help=(
            "Override the category-task head factory (task_a.head_factory) in the MTL task_set. "
            "Default preserves the preset's choice (None → MTLnet's CategoryHeadTransformer default). "
            "Use e.g. --cat-head next_gru or --cat-head category_ensemble for the F27 cat-head ablation."
        ),
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
    parser.add_argument(
        "--folds-path",
        type=str,
        default=None,
        help=(
            "Path to a frozen fold_indices_{task}.pt file (see "
            "scripts/study/freeze_folds.py). When omitted, the canonical "
            "path output/{engine}/{state}/folds/fold_indices_{task}.pt is "
            "auto-loaded if present and its input signatures still match; "
            "otherwise folds are generated from scratch and a warning is "
            "logged."
        ),
    )
    parser.add_argument(
        "--no-folds-cache",
        action="store_true",
        default=False,
        help="Ignore any cached folds and always regenerate. Use only for debugging.",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        default=False,
        help="Skip saving model checkpoints. Saves disk for disposable runs (screen, promote).",
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
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("--batch-size must be > 0")
        config = dataclasses.replace(config, batch_size=args.batch_size)
    if args.max_lr is not None:
        if args.max_lr <= 0:
            raise ValueError("--max-lr must be > 0")
        config = dataclasses.replace(config, max_lr=args.max_lr)
    if args.scheduler is not None:
        config = dataclasses.replace(config, scheduler_type=args.scheduler)
    if args.pct_start is not None:
        if not (0 < args.pct_start < 1):
            raise ValueError("--pct-start must be in (0, 1)")
        config = dataclasses.replace(config, pct_start=args.pct_start)
    # Per-head LR (F48-H3). Validate as a triple — partial sets are an
    # error since the runner only switches to per-head mode when all
    # three are present.
    _per_head_set = {
        "cat_lr": args.cat_lr,
        "reg_lr": args.reg_lr,
        "shared_lr": args.shared_lr,
    }
    _set_keys = [k for k, v in _per_head_set.items() if v is not None]
    if _set_keys and len(_set_keys) != 3:
        raise ValueError(
            f"--cat-lr / --reg-lr / --shared-lr must all be set together, "
            f"got only {_set_keys}. Per-head LR is an all-or-nothing mode."
        )
    if len(_set_keys) == 3:
        for k, v in _per_head_set.items():
            if v <= 0:
                raise ValueError(f"--{k.replace('_', '-')} must be > 0")
        config = dataclasses.replace(
            config,
            cat_lr=args.cat_lr, reg_lr=args.reg_lr, shared_lr=args.shared_lr,
        )
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
    if args.use_class_weights is not None:
        config = dataclasses.replace(config, use_class_weights=args.use_class_weights)

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
    if getattr(args, "replace_model_params", False) and not model_param_overrides:
        raise ValueError("--replace-model-params requires at least one --model-param")
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


def _signatures_match(meta_path: Path, task: str, state: str, engine: EmbeddingEngine) -> bool:
    """Check if cached meta.json's input signatures still match the current parquets.

    Loud mismatch beats silent cache hit: if inputs changed since freeze, force
    regeneration rather than serve stale splits.
    """
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False
    cached = meta.get("inputs_signatures", {})
    expected_files: list[tuple[str, Path]] = []
    if task in ("mtl", "category"):
        expected_files.append(("category.parquet", IoPaths.get_category(state, engine)))
    if task in ("mtl", "next"):
        expected_files.append(("next.parquet", IoPaths.get_next(state, engine)))
    for name, path in expected_files:
        if name not in cached or not path.exists():
            return False
        live = DatasetSignature.from_path(path)
        if live.sha256 != cached[name].get("sha256"):
            return False
    return True


def _resolve_folds(
    args: argparse.Namespace,
    config: ExperimentConfig,
    engine: EmbeddingEngine,
    task_key: str,
    task_set: Optional[TaskSet] = None,
) -> dict:
    """Return the fold_results dict — either loaded from a frozen cache or
    generated on the fly.

    Precedence:
      1. Explicit --folds-path (fail loud if missing or unreadable)
      2. --no-folds-cache forces regeneration
      3. Canonical cache at output/{engine}/{state}/folds/fold_indices_{task}.pt
         IF its input signatures still match the current parquets
      4. Generate from scratch, with a warning pointing at freeze_folds.py
    """
    task_type = _TASK_TYPES[task_key]

    def _from_scratch(reason: str) -> dict:
        logger.warning(
            "Generating folds on the fly (%s). "
            "Run `python scripts/study/freeze_folds.py --state %s --engine %s --task %s` "
            "to freeze them; paired statistical tests require frozen splits.",
            reason, config.state, engine.value, task_key,
        )
        creator = FoldCreator(
            task_type=task_type,
            n_splits=config.k_folds,
            batch_size=config.batch_size,
            seed=config.seed,
            use_weighted_sampling=False,
            task_set=task_set,
            task_a_input_type=getattr(args, "task_a_input_type", "checkin"),
            task_b_input_type=getattr(args, "task_b_input_type", "checkin"),
        )
        return creator.create_folds(config.state, engine)

    if args.no_folds_cache:
        return _from_scratch("--no-folds-cache set")

    if args.folds_path is not None:
        cache_path = Path(args.folds_path)
        if not cache_path.exists():
            raise SystemExit(f"--folds-path {cache_path} does not exist")
        logger.info("Loading folds from %s (explicit --folds-path)", cache_path)
        serialized = load_folds(cache_path)
        return rebuild_dataloaders(serialized, batch_size=config.batch_size, use_weighted_sampling=False)

    canonical = IoPaths.get_folds_dir(config.state, engine) / f"fold_indices_{task_key}.pt"
    meta = canonical.with_suffix(".meta.json")
    if not canonical.exists():
        return _from_scratch(f"no cache at {canonical}")
    if not _signatures_match(meta, task_key, config.state, engine):
        return _from_scratch(
            f"cache {canonical} is stale (input parquet signature changed since freeze)"
        )
    logger.info("Loading frozen folds from %s", canonical)
    serialized = load_folds(canonical)
    # Warn if cached n_splits < requested k_folds (can't expand)
    cached_n = len(next(iter(serialized.fold_indices.values())))
    if cached_n < config.k_folds:
        raise SystemExit(
            f"cached folds have n_splits={cached_n} but config requests "
            f"k_folds={config.k_folds}; re-freeze with --n-splits or reduce --folds"
        )
    return rebuild_dataloaders(serialized, batch_size=config.batch_size, use_weighted_sampling=False)


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

    # Resolve task_set for MTL runs. Default preset keeps legacy
    # {category, next} behaviour bit-exact; non-legacy preset activates
    # the check2HGI task pair and switches fold-creation to MTL_CHECK2HGI.
    task_set: Optional[TaskSet] = None
    is_check2hgi_track = False
    if task_key == "mtl":
        preset_name = args.task_set or LEGACY_CATEGORY_NEXT.name
        task_set = get_preset(preset_name)
        is_check2hgi_track = preset_name == CHECK2HGI_NEXT_REGION.name
        if is_check2hgi_track and engine != EmbeddingEngine.CHECK2HGI:
            print(
                f"error: --task-set {preset_name} requires --engine check2hgi "
                f"(got {engine.value}).",
                file=sys.stderr,
            )
            sys.exit(2)
        # Apply --reg-head / --reg-head-param / --cat-head / --cat-head-param
        # overrides BEFORE fold creation so FoldCreator sees the final
        # ``task_b.head_factory`` when deciding which dataloader path to use
        # (e.g. aux-publishing for the B5 head ``next_getnext_hard``). Cat
        # overrides don't currently affect fold creation but are applied here
        # symmetrically so the early task_set is the authoritative source
        # of head choice. The later ``resolve_task_set`` call after fold
        # creation additionally sets ``task_b_num_classes``, which can only
        # be known once labels are loaded — that pass is kept.
        if is_check2hgi_track and (
            args.reg_head or args.reg_head_param
            or args.cat_head or args.cat_head_param
        ):
            _early_reg_head_params = _parse_key_value_overrides(
                args.reg_head_param or [], "--reg-head-param"
            )
            _early_cat_head_params = _parse_key_value_overrides(
                args.cat_head_param or [], "--cat-head-param"
            )
            task_set = resolve_task_set(
                task_set,
                task_a_head_factory=args.cat_head,
                task_a_head_params=_early_cat_head_params or None,
                task_b_head_factory=args.reg_head,
                task_b_head_params=_early_reg_head_params or None,
            )

    # Resolve folds: prefer the frozen cache under
    # output/{engine}/{state}/folds/fold_indices_{task}.pt (see
    # scripts/study/freeze_folds.py). Falls back to on-the-fly generation
    # with a warning — required for paired statistical tests across
    # ablation runs.
    fold_resolve_key = "mtl_check2hgi" if is_check2hgi_track else task_key
    fold_results = _resolve_folds(
        args, config, engine, fold_resolve_key,
        task_set=task_set if is_check2hgi_track else None,
    )

    # For the check2HGI track: resolve task_b.num_classes from the
    # full next_region label space (the preset stores 0 as a
    # placeholder) and inject the resolved ``task_set`` into model_params.
    #
    # Computing ``n_regions`` from a single fold's train labels is a
    # subtle bug: val folds can contain regions absent from any given
    # train fold (cross-fold user partition doesn't preserve region
    # support). If the model is sized for a too-small n_regions and
    # val/other-fold labels exceed that range, bincount-based metrics
    # fail at runtime. Using the max across EVERY fold's (train ∪ val)
    # labels yields the true label-space size.
    if is_check2hgi_track:
        assert task_set is not None
        # Resolve task_b (region) num_classes from the union of train
        # and val labels across every fold. Cross-fold user partition
        # doesn't preserve region support, so a single fold's train
        # labels can miss classes present in val or other folds; a
        # too-small n_regions breaks bincount-based metrics at runtime.
        # task_a (next_category) is always 7 — no runtime resolution needed.
        max_b = -1
        for fr in fold_results.values():
            max_b = max(
                max_b,
                int(fr.next.train.y.max().item()),
                int(fr.next.val.y.max().item()),
            )
        n_regions = max_b + 1
        reg_head_params = _parse_key_value_overrides(
            args.reg_head_param or [], "--reg-head-param"
        )
        cat_head_params = _parse_key_value_overrides(
            args.cat_head_param or [], "--cat-head-param"
        )
        task_set = resolve_task_set(
            task_set,
            task_a_head_factory=args.cat_head,
            task_a_head_params=cat_head_params if cat_head_params else None,
            task_b_num_classes=n_regions,
            task_b_head_params=reg_head_params if reg_head_params else None,
            task_b_head_factory=args.reg_head,
        )
        logger.info(
            "check2HGI task_set resolved: task_a=%s/%d, task_b=%s/%d",
            task_set.task_a.name, task_set.task_a.num_classes,
            task_set.task_b.name, task_set.task_b.num_classes,
        )
        updated_params = dict(config.model_params)
        updated_params["task_set"] = task_set
        config = dataclasses.replace(config, model_params=updated_params)

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

    global _NO_CHECKPOINTS
    _NO_CHECKPOINTS = getattr(args, "no_checkpoints", False)

    if is_check2hgi_track:
        assert task_set is not None
        _run_mtl_check2hgi(config, results_path, fold_results, task_set)
    else:
        runner = _RUNNERS[task_key]
        runner(config, results_path, fold_results)

    logger.info("Done. Results written to: %s", results_path)


if __name__ == "__main__":
    main()
