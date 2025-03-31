from calflops import calculate_flops

from configs.model import ModelConfig
from configs.paths import IO_ROOT, OUTPUT_ROOT
from data.create_fold import create_folds
from modeling.mtl_train import train_with_cross_validation

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    # Define parameters
    next_data_path = f'/Users/vitor/Desktop/mestrado/ingred/data/ori/pre-processing/alabama/nextpoi-input.csv'  # Replace with actual path
    category_data_path = f'/Users/vitor/Desktop/mestrado/ingred/data/ori/pre-processing/alabama/categorypoi-input.csv'  # Replace with actual path

    logging.info(f'Creating folds')

    # Process data and create dataloaders
    fold_results = create_folds(
        next_data_path,
        category_data_path,
        k_splits=5
    )

    logging.info(f'Fold results: {len(fold_results)}')

    results = train_with_cross_validation(
        dataloaders=fold_results,
        num_classes=ModelConfig.NUM_CLASSES,
        num_epochs=ModelConfig.EPOCHS,
        learning_rate=ModelConfig.LEARNING_RATE
    )
