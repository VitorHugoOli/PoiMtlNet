import sys
from pathlib import Path

# Ensure src/ is on sys.path so project imports work when invoked directly
_src = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.paths import IoPaths, RESULTS_ROOT, EmbeddingEngine
from configs.experiment import ExperimentConfig

import logging
from typing import List, Tuple

from data.folds import FoldCreator, TaskType
from common.ml_history import MLHistory, DatasetHistory
from configs.globals import CATEGORIES_MAP
from training.runners.mtl_cv import train_with_cross_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_mtl_model(state: str, embedd_engine: EmbeddingEngine) -> dict:
    """
    Train MTL model for a specific state and embedding engine.

    Args:
        state: State name (e.g., "florida", "california")
        embedd_engine: Embedding engine to use (DGI, HGI, HMRM)

    Returns:
        Dictionary with training results
    """
    config = ExperimentConfig.default_mtl(
        name=f"mtl_{state}_{embedd_engine.value}",
        state=state,
        embedding_engine=embedd_engine.value,
    )

    logger.info(f"{'='*80}")
    logger.info(f"Starting training for: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}")

    # Create folds
    logger.info(f'Creating {config.k_folds}-fold cross-validation splits...')
    creator = FoldCreator(
        task_type=TaskType.MTL,
        n_splits=config.k_folds,
        batch_size=config.batch_size,
        use_weighted_sampling=False,
    )
    fold_results = creator.create_folds(state, embedd_engine)
    folds_path = None  # Can use creator.save(output_dir) if needed

    # Initialize ML History
    results_path = IoPaths.get_results_dir(state, embedd_engine)

    history = MLHistory(
        model_name='MTLNet',
        tasks={'next', 'category'},
        num_folds=config.k_folds,
        datasets={
            DatasetHistory(
                raw_data=IoPaths.get_next(state, embedd_engine),
                folds_signature=folds_path,
                description="Data related to next POI prediction. Data with 107 features",
            ),
            DatasetHistory(
                raw_data=IoPaths.get_category(state, embedd_engine),
                folds_signature=folds_path,
                description="Data related to category prediction. Data with 107 features",
            )
        },
        label_map=CATEGORIES_MAP,
        save_path=results_path,
        verbose=True,
    )

    # Train with cross-validation
    logger.info(f'Starting cross-validation training...')
    with history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            config=config,
            results_path=results_path,
        )

    logger.info(f"Completed training for: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}\n")

    return results


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train MTL model with cross-validation")
    parser.add_argument(
        "--state", type=str, nargs="+",
        default=["alabama"],
        help="State(s) to train on (e.g. alabama florida texas)",
    )
    parser.add_argument(
        "--engine", type=str, nargs="+",
        default=["hgi"],
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

    logger.info(f"Starting MTL training pipeline")
    logger.info(f"Total configurations to train: {len(TRAINING_CONFIGS)}")
    logger.info(f"Configurations: {[(s, e.value) for s, e in TRAINING_CONFIGS]}\n")

    # Train each configuration sequentially
    all_results = {}
    for idx, (state, embedd_engine) in enumerate(TRAINING_CONFIGS, 1):
        logger.info(f"Training configuration {idx}/{len(TRAINING_CONFIGS)}")

        try:
            results = train_mtl_model(state, embedd_engine)
            all_results[f"{state}_{embedd_engine.value}"] = results
        except Exception as e:
            logger.error(f"Failed to train {state} with {embedd_engine.value}: {str(e)}", exc_info=True)
            continue

    logger.info(f"\n{'='*80}")
    logger.info(f"All training completed!")
    logger.info(f"Successfully trained: {len(all_results)}/{len(TRAINING_CONFIGS)} configurations")
    logger.info(f"{'='*80}")
