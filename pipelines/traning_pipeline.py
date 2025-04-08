import torch
from calflops import calculate_flops

from configs.model import ModelConfig
from configs.paths import IO_ROOT, OUTPUT_ROOT, RESULTS_PATH
from data.create_fold import create_folds
from modeling.mtl_train import train_with_cross_validation

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    state = "florida"  # Replace with the desired state

    # Define parameters
    next_data_path = f'{OUTPUT_ROOT}/{state}/pre-processing/next-input.csv'  # Replace with actual path
    category_data_path = f'{OUTPUT_ROOT}/{state}/pre-processing/category-input.csv'  # Replace with actual path

    logging.info(f'Creating folds')

    # Process data and create dataloaders
    fold_results = create_folds(
        next_data_path,
        category_data_path,
        k_splits=ModelConfig.K_FOLDS,
    )

    logging.info(f'Fold results: {len(fold_results)}')

    results = train_with_cross_validation(
        dataloaders=fold_results,
        num_classes=ModelConfig.NUM_CLASSES,
        num_epochs=ModelConfig.EPOCHS,
        learning_rate=ModelConfig.LEARNING_RATE
    )

    results.export_to_csv(output_dir=RESULTS_PATH + f'/{state}')
