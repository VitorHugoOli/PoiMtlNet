import os
import pickle

import torch
from numpy.distutils.lib2def import output_def

from configs.model import MTLModelConfig
from configs.paths import IO_ROOT, OUTPUT_ROOT, RESULTS_PATH
from data.create_fold import create_folds
from modeling.mtl_train import train_with_cross_validation

import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':

    state = "florida_new"  # Replace with the desired state
    output_dir = f'{OUTPUT_ROOT}/{state}/pre-processing'
    # output_dir = f'/Users/vitor/Desktop/mestrado/ingred/data/ori/pre-processing/chicago'

    # Define parameters
    next_data_path = f'{output_dir}/next-input.csv'  # Replace with actual path
    category_data_path = f'{output_dir}/category-input.csv'  # Replace with actual path

    logging.info(f'Creating folds')

    # Process data and create dataloaders
    fold_results = create_folds(
        next_data_path,
        category_data_path,
        k_splits=MTLModelConfig.K_FOLDS,
    )

    #save the folds in pickle format
    with open(f'{output_dir}/folds.pkl', 'wb') as f:
        pickle.dump(fold_results, f)

    logging.info(f'Fold results: {len(fold_results)}')

    results = train_with_cross_validation(
        dataloaders=fold_results,
        num_classes=MTLModelConfig.NUM_CLASSES,
        num_epochs=MTLModelConfig.EPOCHS,
        learning_rate=MTLModelConfig.LEARNING_RATE
    )

    results.export_to_csv(output_dir=RESULTS_PATH + f'/{state}')
