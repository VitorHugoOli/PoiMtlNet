from configs.model import MTLModelConfig
from configs.paths import IoPaths, RESULTS_ROOT, EmbeddingEngine

import logging
from typing import List, Tuple

from etl.mtl.create_fold import create_folds
from common.ml_history.metrics import MLHistory
from common.ml_history.utils.dataset import DatasetHistory
from train.mtlnet.mtl_train import train_with_cross_validation

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
    logger.info(f"{'='*80}")
    logger.info(f"Starting training for: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}")

    # Create folds
    logger.info(f'Creating {MTLModelConfig.K_FOLDS}-fold cross-validation splits...')
    fold_results, folds_path = create_folds(
        state,
        embedd_engine,
        k_splits=MTLModelConfig.K_FOLDS,
        save_folder=None,
    )

    # Initialize ML History
    history = MLHistory(
        model_name='MTLNet',
        tasks={'next', 'category'},
        num_folds=MTLModelConfig.K_FOLDS,
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
        }
    )

    # Train with cross-validation
    logger.info(f'Starting cross-validation training...')
    with history.context() as history:
        results = train_with_cross_validation(
            dataloaders=fold_results,
            history=history,
            num_classes=MTLModelConfig.NUM_CLASSES,
            num_epochs=MTLModelConfig.EPOCHS,
            learning_rate=MTLModelConfig.LEARNING_RATE
        )

    # Save results
    results_path = IoPaths.get_results_dir(state, embedd_engine)
    logger.info(f'Saving results to: {results_path}')
    history.storage.save(path=results_path)

    logger.info(f"Completed training for: {state.upper()} with {embedd_engine.value.upper()}")
    logger.info(f"{'='*80}\n")

    return results


if __name__ == '__main__':
    # Define configurations to train: [(state, embedding_engine), ...]
    TRAINING_CONFIGS: List[Tuple[str, EmbeddingEngine]] = [
        # ("florida", EmbeddingEngine.DGI),
        # ("florida", EmbeddingEngine.HGI),
        ("alabama", EmbeddingEngine.DGI),
        # ("arizona", EmbeddingEngine.DGI),
        # ("georgia", EmbeddingEngine.DGI),
        # ("florida", EmbeddingEngine.DGI),
        # ("california", EmbeddingEngine.DGI),
        # ("texas", EmbeddingEngine.DGI),
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
