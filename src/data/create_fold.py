import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from dataclasses import dataclass
from typing import Dict

from tqdm import tqdm

from configs.globals import DEVICE, CATEGORIES_MAP


# Define dataclasses for organizing data
@dataclass
class InputData:
    """Class for organizing model input data"""
    dataloader: DataLoader
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class SuperInputData:
    """Class for organizing all splits of data"""
    train: InputData
    test: InputData
    val: InputData


class CustomDataset(Dataset):
    def __init__(self, input_features, true_y):
        self.input_features = input_features
        self.true_y = true_y

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        sample = {
            'x': self.input_features[idx],
            'y': self.true_y[idx]
        }
        return sample


# Define the mapping function for categories
def map_categories(y):
    inv_categories = {v: k for k, v in CATEGORIES_MAP.items()}

    try:
        y_encoded = np.array([[inv_categories[yi] for yi in row] for _, row in y.iterrows()])
    except:
        y_encoded = y.map(inv_categories)

    return y_encoded


# Define function to convert data to tensors
def x_y_to_tensor(x, y, task_type):
    """
    Convert pandas dataframes/series to PyTorch tensors
    """
    x_tensor = torch.tensor(x.values, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y.values, dtype=torch.long).to(DEVICE)

    if task_type == 'next':
        x_tensor = x_tensor.view(-1, 9, 100)
    else:
        x_tensor = x_tensor.view(-1, 1, 100)

    return x_tensor, y_tensor


# Define function to create dataloaders
def input_to_dataloader(x, y, x_cut=None, y_cut=None, batch_size=64, shuffle=True):
    """
    Create PyTorch DataLoader from tensor data
    """
    # if x_cut is not none, y_cut should be none and vice versa
    assert (x_cut is not None) == (y_cut is not None), "x_cut and y_cut should be both None or both not None"

    if x_cut is not None:
        x = x[x_cut]
        y = y[y_cut]

    dataset = CustomDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def create_folds(path_next_input, path_category_input, k_splits=5):
    """
    Main function to process the POI data and create dataloaders

    Args:
        path_next_input: Path to the next POI input CSV
        path_category_input: Path to the category POI input CSV

    Returns:
        Dictionary containing SuperInputData objects for each fold and task
    """
    print("Loading and processing Next POI data...")
    df_next_input = pd.read_csv(path_next_input)
    print(f"Next POI data shape: {df_next_input.shape}")

    # Get unique user IDs
    userids = df_next_input['userid'].unique()
    print(f"Number of unique users: {len(userids)}")

    # Extract target columns (columns from 900 to the end except the last one)
    target = df_next_input.columns[900:-1]

    # Create features and targets for next POI
    x_next = df_next_input.drop(target, axis=1)
    y_next = df_next_input[target]
    y_next = y_next.fillna('None')
    y_next = map_categories(y_next)
    y_next = pd.DataFrame(y_next)
    print(f"Next POI feature shape: {x_next.shape}, target shape: {y_next.shape}")

    print("\nLoading and processing Category POI data...")
    # Read Category POI data
    category_input = pd.read_csv(path_category_input)
    print(f"Category POI data shape: {category_input.shape}")

    # Get unique place IDs
    places_ids = category_input['placeid'].unique()
    places_ids = places_ids.astype(int)
    print(f"Number of unique places: {len(places_ids)}")

    # Set place ID as index
    category_input = category_input.set_index('placeid')

    # Create features and targets for category POI
    x_category = category_input.drop('category', axis=1)
    y_category = category_input['category']
    y_category = map_categories(y_category)
    print(f"Category POI feature shape: {x_category.shape}, target shape: {y_category.shape}")

    # Ensure correct data types
    try:
        x_next.userid = x_next.userid.astype(int)
        userids = x_next['userid'].unique()
    except:
        pass

    # K-fold setup
    kf = KFold(n_splits=k_splits, shuffle=True, random_state=42)

    # Store results for each fold
    fold_results: Dict[int, Dict[str, SuperInputData]] = {}

    io_input = enumerate(
        zip(kf.split(userids), kf.split(range(len(x_category)))))

    for fold, ((train_user_index, test_user_index), (train_place_index, test_place_index)) in tqdm(io_input,
                                                                                                   desc="Processing folds"):
        # Make copies of the data for this fold
        x_next_fold = x_next.copy()
        y_next_fold = y_next.copy()
        x_category_fold = x_category.copy()
        y_category_fold = y_category.copy()

        # Split train data into train and validation sets
        train_user_index, val_user_index = train_test_split(train_user_index, test_size=0.25, random_state=42)
        train_place_index, val_place_index = train_test_split(train_place_index, test_size=0.25, random_state=42)

        # Get user IDs for each split
        train_users = userids[train_user_index]
        val_users = userids[val_user_index]
        test_users = userids[test_user_index]

        # Get indices for Next POI data splits
        train_index_next = x_next_fold[x_next_fold['userid'].isin(train_users)].index
        val_index_next = x_next_fold[x_next_fold['userid'].isin(val_users)].index
        test_index_next = x_next_fold[x_next_fold['userid'].isin(test_users)].index

        # Drop user ID column as it's no longer needed for modeling
        x_next_fold = x_next_fold.drop('userid', axis=1)

        # Convert Next POI data to tensors
        x_next_tensor, y_next_tensor = x_y_to_tensor(x_next_fold, y_next_fold, 'next')

        # Create dataloaders for Next POI
        next_dataloader_train = input_to_dataloader(
            x_next_tensor, y_next_tensor,
            x_cut=train_index_next, y_cut=train_index_next)
        next_dataloader_val = input_to_dataloader(
            x_next_tensor, y_next_tensor,
            x_cut=val_index_next, y_cut=val_index_next)
        next_dataloader_test = input_to_dataloader(
            x_next_tensor, y_next_tensor,
            x_cut=test_index_next, y_cut=test_index_next)

        # Convert Category POI data to tensors
        x_category_tensor, y_category_tensor = x_y_to_tensor(x_category_fold, y_category_fold, 'category')

        # Create dataloaders for Category POI
        category_dataloader_train = input_to_dataloader(
            x_category_tensor, y_category_tensor,
            x_cut=train_place_index, y_cut=train_place_index)
        category_dataloader_val = input_to_dataloader(
            x_category_tensor, y_category_tensor,
            x_cut=val_place_index, y_cut=val_place_index)
        category_dataloader_test = input_to_dataloader(
            x_category_tensor, y_category_tensor,
            x_cut=test_place_index, y_cut=test_place_index)

        # Create SuperInputData for Next POI
        next_data = SuperInputData(
            train=InputData(
                dataloader=next_dataloader_train,
                x=x_next_tensor[train_index_next],
                y=y_next_tensor[train_index_next]
            ),
            val=InputData(
                dataloader=next_dataloader_val,
                x=x_next_tensor[val_index_next],
                y=y_next_tensor[val_index_next]
            ),
            test=InputData(
                dataloader=next_dataloader_test,
                x=x_next_tensor[test_index_next],
                y=y_next_tensor[test_index_next]
            )
        )

        # Create SuperInputData for Category POI
        category_data = SuperInputData(
            train=InputData(
                dataloader=category_dataloader_train,
                x=x_category_tensor[train_index_next],
                y=y_category_tensor[train_index_next]
            ),
            val=InputData(
                dataloader=category_dataloader_val,
                x=x_category_tensor[val_place_index],
                y=y_category_tensor[val_place_index]
            ),
            test=InputData(
                dataloader=category_dataloader_test,
                x=x_category_tensor[test_place_index],
                y=y_category_tensor[test_place_index]
            )
        )

        # Store results for this fold
        fold_results[fold] = {
            'next': next_data,
            'category': category_data
        }

    return fold_results


# Example usage
if __name__ == "__main__":
    # Define parameters
    path_nextpoi_input_alabama = "path/to/nextpoi_input_alabama.csv"  # Replace with actual path
    path_categorypoi_input_alabama = "path/to/categorypoi_input_alabama.csv"  # Replace with actual path

    # Process data and create dataloaders
    fold_results = create_folds(
        path_nextpoi_input_alabama,
        path_categorypoi_input_alabama,
        k_splits=5
    )
