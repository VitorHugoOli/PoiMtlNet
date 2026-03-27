import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure src/ is on sys.path so project imports work when invoked directly
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.paths import IoPaths, RESULTS_ROOT, EmbeddingEngine
from configs.experiment import ExperimentConfig
from data.folds import FoldCreator, TaskType
from train.category.cross_validation import run_cv
from common.ml_history import MLHistory, DatasetHistory
from configs.globals import CATEGORIES_MAP


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_category_model(
    state: str,
    embedd_engine: EmbeddingEngine,
    save_folds: bool = False,
    folds_chkpt: Optional[str] = None
) -> dict:
    """
    Train Category prediction model for a specific state and embedding engine.

    Args:
        state: State name (e.g., "florida", "california")
        embedd_engine: Embedding engine to use (DGI, HGI, HMRM)
        save_folds: Whether to save folds to disk
        folds_chkpt: Path to checkpoint file for folds

    Returns:
        Dictionary with training results
    """
    config = ExperimentConfig.default_category(
        name=f"category_{state}_{embedd_engine.value}",
        state=state,
        embedding_engine=embedd_engine.value,
    )

    logger.info(f"{'='*80}")
    logger.info(f"Starting Category training: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}")

    # Get paths
    data_input = IoPaths.get_category(state, embedd_engine)
    output_dir = IoPaths.get_results_dir(state, embedd_engine)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Creating folds
    logger.info("Creating folds...")
    folds = None
    folds_save_path = None

    if folds_chkpt is not None:
        folds_chkpt_path = Path(output_dir) / folds_chkpt
        logger.info(f"Loading folds from checkpoint: {folds_chkpt_path}")
        fold_results = FoldCreator.load(folds_chkpt_path)
        folds = [
            (fold_results[i].category.train.dataloader, fold_results[i].category.val.dataloader)
            for i in range(len(fold_results))
        ]
    else:
        creator = FoldCreator(
            task_type=TaskType.CATEGORY,
            n_splits=config.k_folds,
            batch_size=config.batch_size,
            seed=config.seed,
            use_weighted_sampling=False,
        )
        fold_results = creator.create_folds(state, embedd_engine)
        folds = [
            (fold_results[i].category.train.dataloader, fold_results[i].category.val.dataloader)
            for i in range(len(fold_results))
        ]

        if save_folds:
            folds_pth = Path(output_dir) / 'folds'
            folds_pth.mkdir(parents=True, exist_ok=True)
            folds_save_path = creator.save(folds_pth)
            logger.info(f"Saving folds to: {folds_save_path}")

    # Creating history
    history: MLHistory = MLHistory(
        model_name="Category",
        model_type="Single-Task",
        tasks='category',
        num_folds=config.k_folds,
        datasets={
            DatasetHistory(
                raw_data=str(data_input),
                folds_signature=str(folds_save_path) if folds_save_path else folds_chkpt,
                description="POI Category Classification",
            )
        },
        label_map=CATEGORIES_MAP,
        save_path=str(output_dir),
        verbose=True,
        display_report=True
    )

    # Running cross-validation
    logger.info(f"Starting cross-validation training...")
    with history:
        results = run_cv(history, folds, config, results_path=output_dir)

    history.display.end_training()
    logger.info(f"Completed Category training: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}\n")

    return results


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train Category model with cross-validation")
    parser.add_argument(
        "--state", type=str, nargs="+",
        default=["california"],
        help="State(s) to train on (e.g. alabama florida texas)",
    )
    parser.add_argument(
        "--engine", type=str, nargs="+",
        default=["poi2hgi"],
        choices=[e.value for e in EmbeddingEngine],
        help="Embedding engine(s) to use",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    TRAINING_CONFIGS: List[Tuple[str, EmbeddingEngine]] = [
        (state, EmbeddingEngine(engine))
        for state in args.state
        for engine in args.engine
    ]

    logger.info(f"Starting Category training pipeline")
    logger.info(f"Total configurations to train: {len(TRAINING_CONFIGS)}")
    logger.info(f"Configurations: {[(s, e.value) for s, e in TRAINING_CONFIGS]}\n")

    # Train each configuration sequentially
    all_results = {}
    for idx, (state, embedd_engine) in enumerate(TRAINING_CONFIGS, 1):
        logger.info(f"Training configuration {idx}/{len(TRAINING_CONFIGS)}")

        try:
            results = train_category_model(
                state=state,
                embedd_engine=embedd_engine,
                save_folds=False,
                folds_chkpt=None
            )
            all_results[f"{state}_{embedd_engine.value}"] = results
        except Exception as e:
            logger.error(f"Failed to train {state} with {embedd_engine.value}: {str(e)}", exc_info=True)
            continue

    logger.info(f"\n{'='*80}")
    logger.info(f"All training completed!")
    logger.info(f"Successfully trained: {len(all_results)}/{len(TRAINING_CONFIGS)} configurations")
    logger.info(f"{'='*80}")
