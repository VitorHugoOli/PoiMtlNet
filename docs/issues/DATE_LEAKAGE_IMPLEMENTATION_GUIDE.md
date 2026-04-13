# Implementation Guide: Fixing Data Leakage in Embedding Training

This document provides step-by-step technical instructions for implementing inductive training to fix the data leakage problem.

---

## Table of Contents

1. [Overview](#1-overview)
2. [File Modifications Summary](#2-file-modifications-summary)
3. [Step 1: Create Data Splitting Utilities](#3-step-1-create-data-splitting-utilities)
4. [Step 2: Modify Check2HGI Preprocessing](#4-step-2-modify-check2hgi-preprocessing)
5. [Step 3: Modify Check2HGI Training](#5-step-3-modify-check2hgi-training)
6. [Step 4: Implement Inference Mode](#6-step-4-implement-inference-mode)
7. [Step 5: Modify HGI for POI-based Split](#7-step-5-modify-hgi-for-poi-based-split)
8. [Step 6: Update Input Creation](#8-step-6-update-input-creation)
9. [Step 7: Update Fold Creator](#9-step-7-update-fold-creator)
10. [Step 8: Update Pipelines](#10-step-8-update-pipelines)
11. [Testing Plan](#11-testing-plan)
12. [Migration Guide](#12-migration-guide)

---

## 1. Overview

### Current Flow (With Leakage)
```
All Check-ins → Preprocess → Train Embedding → Generate ALL Embeddings
                                                      ↓
                                    MTL Fold Split (80/20) ← LEAKAGE!
```

### Target Flow (No Leakage)
```
All Check-ins → Split Users (80/10/10)
                     ↓
              Train Users → Preprocess → Train Embedding
                                              ↓
              Val/Test Users → Preprocess → Inference Only
                                              ↓
                           Separate Embedding Files
                                              ↓
                      MTL Training (uses pre-split data)
```

---

## 2. File Modifications Summary

| File | Status | Changes |
|------|--------|---------|
| `src/etl/data_split.py` | **NEW** | User/POI splitting utilities |
| `src/embeddings/check2hgi/preprocess.py` | Modify | Add `user_filter` parameter |
| `src/embeddings/check2hgi/check2hgi.py` | Modify | Add `inference_mode`, save model |
| `src/embeddings/check2hgi/model/Check2HGIModule.py` | Modify | Add `encode()` method for inference |
| `src/embeddings/hgi/preprocess.py` | Modify | Add `poi_filter` parameter |
| `src/embeddings/hgi/hgi.py` | Modify | Add inference mode |
| `src/etl/create_input.py` | Modify | Handle split-aware inputs |
| `src/etl/create_fold.py` | Modify | Use pre-split data |
| `pipelines/embedding/check2hgi.pipe.py` | Modify | Orchestrate split pipeline |
| `pipelines/train/mtl.pipe.py` | Modify | Use correct splits |

---

## 3. Step 1: Create Data Splitting Utilities

### Create new file: `src/etl/data_split.py`

```python
"""
Data splitting utilities for inductive training.

Usage:
    from etl.data_split import DataSplitter

    # For Next-POI task (user-based split)
    splitter = DataSplitter(seed=42)
    splits = splitter.split_by_users(checkins_df, train_ratio=0.8, val_ratio=0.1)

    # For Category task (POI-based split)
    splits = splitter.split_by_pois(pois_df, train_ratio=0.8, stratify_by='category')
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
from pathlib import Path
import json


@dataclass
class DataSplit:
    """Container for train/val/test splits."""
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    split_type: str  # 'user' or 'poi'
    seed: int

    def save(self, path: Path) -> None:
        """Save split to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'train_ids': self.train_ids.tolist(),
            'val_ids': self.val_ids.tolist(),
            'test_ids': self.test_ids.tolist(),
            'split_type': self.split_type,
            'seed': self.seed,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved split to {path}")

    @classmethod
    def load(cls, path: Path) -> 'DataSplit':
        """Load split from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            train_ids=np.array(data['train_ids']),
            val_ids=np.array(data['val_ids']),
            test_ids=np.array(data['test_ids']),
            split_type=data['split_type'],
            seed=data['seed'],
        )

    def get_train_set(self) -> Set:
        return set(self.train_ids.tolist())

    def get_val_set(self) -> Set:
        return set(self.val_ids.tolist())

    def get_test_set(self) -> Set:
        return set(self.test_ids.tolist())


class DataSplitter:
    """Utility for splitting data by users or POIs."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def split_by_users(
        self,
        checkins_df: pd.DataFrame,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        min_checkins_per_user: int = 10,
    ) -> DataSplit:
        """
        Split check-in data by users.

        Ensures entire user trajectories stay together.
        Users with fewer than min_checkins_per_user are excluded.

        Args:
            checkins_df: DataFrame with 'userid' column
            train_ratio: Fraction of users for training (default 0.8)
            val_ratio: Fraction of users for validation (default 0.1)
            min_checkins_per_user: Minimum check-ins required per user

        Returns:
            DataSplit with user IDs for train/val/test
        """
        # Filter users with enough check-ins
        user_counts = checkins_df.groupby('userid').size()
        valid_users = user_counts[user_counts >= min_checkins_per_user].index.values

        # Shuffle users
        np.random.shuffle(valid_users)

        # Calculate split points
        n_users = len(valid_users)
        n_train = int(n_users * train_ratio)
        n_val = int(n_users * val_ratio)

        train_users = valid_users[:n_train]
        val_users = valid_users[n_train:n_train + n_val]
        test_users = valid_users[n_train + n_val:]

        print(f"User split: train={len(train_users)}, val={len(val_users)}, test={len(test_users)}")
        print(f"Total valid users: {n_users}, excluded: {len(user_counts) - n_users}")

        return DataSplit(
            train_ids=train_users,
            val_ids=val_users,
            test_ids=test_users,
            split_type='user',
            seed=self.seed,
        )

    def split_by_pois(
        self,
        pois_df: pd.DataFrame,
        train_ratio: float = 0.8,
        stratify_by: Optional[str] = 'category',
    ) -> DataSplit:
        """
        Split POI data for category classification.

        Ensures truly unseen POIs in test set (cold-start scenario).

        Args:
            pois_df: DataFrame with 'placeid' and optionally 'category' columns
            train_ratio: Fraction of POIs for training (default 0.8)
            stratify_by: Column to stratify by (default 'category')

        Returns:
            DataSplit with POI IDs for train/val/test
        """
        if stratify_by and stratify_by in pois_df.columns:
            # Stratified split to maintain category distribution
            from sklearn.model_selection import train_test_split

            pois = pois_df['placeid'].values
            categories = pois_df[stratify_by].values

            # First split: train vs (val+test)
            train_pois, temp_pois, _, temp_cats = train_test_split(
                pois, categories,
                train_size=train_ratio,
                stratify=categories,
                random_state=self.seed
            )

            # Second split: val vs test (50/50 of remaining)
            val_pois, test_pois = train_test_split(
                temp_pois,
                test_size=0.5,
                stratify=temp_cats,
                random_state=self.seed
            )
        else:
            # Random split
            pois = pois_df['placeid'].values.copy()
            np.random.shuffle(pois)

            n_pois = len(pois)
            n_train = int(n_pois * train_ratio)
            n_val = int(n_pois * 0.1)

            train_pois = pois[:n_train]
            val_pois = pois[n_train:n_train + n_val]
            test_pois = pois[n_train + n_val:]

        print(f"POI split: train={len(train_pois)}, val={len(val_pois)}, test={len(test_pois)}")

        return DataSplit(
            train_ids=train_pois,
            val_ids=val_pois,
            test_ids=test_pois,
            split_type='poi',
            seed=self.seed,
        )

    def filter_checkins_by_users(
        self,
        checkins_df: pd.DataFrame,
        user_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Filter check-ins to only include specified users."""
        user_set = set(user_ids.tolist())
        return checkins_df[checkins_df['userid'].isin(user_set)].copy()

    def filter_pois_by_ids(
        self,
        pois_df: pd.DataFrame,
        poi_ids: np.ndarray,
    ) -> pd.DataFrame:
        """Filter POIs to only include specified IDs."""
        poi_set = set(poi_ids.tolist())
        return pois_df[pois_df['placeid'].isin(poi_set)].copy()
```

---

## 4. Step 2: Modify Check2HGI Preprocessing

### Modify: `src/embeddings/check2hgi/preprocess.py`

Add `user_filter` parameter to filter check-ins during preprocessing.

```python
# Add to Check2HGIPreprocess.__init__() (around line 19)
def __init__(self, checkins_file, boroughs_file, temp_path, edge_type='user_sequence',
             temporal_decay=3600.0, user_filter=None):
    """
    Initialize preprocessor.

    Args:
        checkins_file: Path to check-in parquet file
        boroughs_file: Path to boroughs/regions CSV file
        temp_path: Path to save intermediate files
        edge_type: Type of edges ('user_sequence', 'same_poi', 'both')
        temporal_decay: Decay parameter for temporal edge weights (seconds)
        user_filter: Optional set of user IDs to include (for inductive training)
    """
    self.checkins_file = checkins_file
    self.boroughs_file = boroughs_file
    self.temp_path = Path(temp_path)
    self.edge_type = edge_type
    self.temporal_decay = temporal_decay
    self.user_filter = set(user_filter) if user_filter is not None else None


# Modify _load_checkins() (around line 37)
def _load_checkins(self):
    """Load and prepare check-in data."""
    print("Loading check-in data...")
    self.checkins = pd.read_parquet(self.checkins_file)

    # Ensure required columns exist
    required = ['userid', 'placeid', 'datetime', 'category', 'latitude', 'longitude']
    missing = [c for c in required if c not in self.checkins.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ADDED: Filter by users if specified (for inductive training)
    if self.user_filter is not None:
        original_len = len(self.checkins)
        self.checkins = self.checkins[self.checkins['userid'].isin(self.user_filter)]
        print(f"Filtered to {len(self.checkins)} check-ins ({original_len - len(self.checkins)} excluded)")

    # Sort by user and time (essential for user_sequence edges)
    self.checkins = self.checkins.sort_values(['userid', 'datetime']).reset_index(drop=True)
    # ... rest of the method unchanged


# Modify preprocess_check2hgi() function (around line 348)
def preprocess_check2hgi(city, city_shapefile, edge_type='user_sequence',
                          temporal_decay=3600.0, cta_file=None, user_filter=None):
    """
    Main preprocessing function for Check2HGI.

    Args:
        city: City/state name
        city_shapefile: Path to census tract shapefile
        edge_type: Type of edges ('user_sequence', 'same_poi', 'both')
        temporal_decay: Decay parameter for temporal edge weights
        cta_file: Optional path to pre-computed boroughs file
        user_filter: Optional list/set of user IDs to include (for inductive training)
    """
    temp_folder = IoPaths.CHECK2HGI.get_temp_dir(city)
    temp_folder.mkdir(parents=True, exist_ok=True)

    output_folder = IoPaths.CHECK2HGI.get_output_dir(city)
    output_folder.mkdir(parents=True, exist_ok=True)

    checkins_file = IoPaths.get_city(city)
    boroughs_path = Path(cta_file) if cta_file else temp_folder / "boroughs_area.csv"

    # Create boroughs file if not exists
    if cta_file is None and not boroughs_path.exists():
        print(f"Creating boroughs file from shapefile: {city_shapefile}")
        census = gpd.read_file(city_shapefile).to_crs(4326)
        census[['GEOID', 'geometry']].to_csv(boroughs_path, index=False)

    pre = Check2HGIPreprocess(
        checkins_file=str(checkins_file),
        boroughs_file=str(boroughs_path),
        temp_path=temp_folder,
        edge_type=edge_type,
        temporal_decay=temporal_decay,
        user_filter=user_filter,  # ADDED
    )

    data = pre.get_data()

    # ADDED: Include split info in filename if filtered
    suffix = "_train" if user_filter is not None else ""
    output_path = IoPaths.CHECK2HGI.get_graph_data_file(city).with_suffix(f'{suffix}.pkl')

    with open(output_path, 'wb') as f:
        pkl.dump(data, f)

    print(f"Saved: {output_path}")
    print(f"Check-ins: {data['num_checkins']}, POIs: {data['num_pois']}, "
          f"Regions: {data['num_regions']}, Edges: {len(data['edge_weight'])}")

    return output_path
```

---

## 5. Step 3: Modify Check2HGI Training

### Modify: `src/embeddings/check2hgi/check2hgi.py`

Add model saving and inference mode support.

```python
# Add after imports (around line 30)
from pathlib import Path


# Modify train_check2hgi() to save model checkpoint
def train_check2hgi(city, args, output_suffix=""):
    """Train Check2HGI model and generate embeddings.

    Args:
        city: City/state name
        args: Training arguments
        output_suffix: Suffix for output files (e.g., "_train", "_val")

    Returns:
        model_path: Path to saved model checkpoint
    """
    output_folder = IoPaths.CHECK2HGI.get_state_dir(city)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    data_path = IoPaths.CHECK2HGI.get_graph_data_file(city)
    if output_suffix:
        data_path = data_path.with_suffix(f'{output_suffix}.pkl')

    print(f"Loading data: {data_path}")

    with open(data_path, 'rb') as handle:
        city_dict = pkl.load(handle)

    # ... (existing model initialization code unchanged until training loop)

    # Training loop
    t = trange(1, args.epoch + 1, desc="Training Check2HGI")
    lowest_loss = math.inf
    best_epoch = 0
    best_state = None

    for epoch in t:
        if use_mini_batch:
            loss = train_epoch_mini_batch(data, loader, model, optimizer, args, use_amp, device_type)
        else:
            loss = train_epoch_full_batch(data, model, optimizer, scheduler, args, use_amp, device_type)

        if loss < lowest_loss:
            lowest_loss = loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        t.set_postfix(loss=f'{loss:.4f}', best=f'{lowest_loss:.4f}', best_epoch=best_epoch)

    # Load best model and extract embeddings
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(best_state)

    # ADDED: Save model checkpoint for inference
    model_path = output_folder / f"model{output_suffix}.pt"
    torch.save({
        'model_state_dict': best_state,
        'args': vars(args),
        'in_channels': in_channels,
        'num_pois': num_pois,
        'num_regions': num_regions,
        'placeid_to_idx': city_dict['placeid_to_idx'],
        'region_to_idx': city_dict['region_to_idx'],
    }, model_path)
    print(f"Saved model checkpoint: {model_path}")

    # Final forward pass to get embeddings
    if data.x.device != model.checkin_encoder.convs[0].lin.weight.device:
        data = data.to(args.device)

    model.eval()
    with torch.no_grad():
        _ = model(data)
        checkin_emb, poi_emb, region_emb = model.get_embeddings()

    # Save embeddings with suffix
    output_path = IoPaths.get_embedd(city, EmbeddingEngine.CHECK2HGI)
    if output_suffix:
        output_path = output_path.with_stem(output_path.stem + output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_np = checkin_emb.numpy()
    df = pd.DataFrame(embeddings_np, columns=[f'{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'datetime', metadata['datetime'].values)
    df.insert(0, 'category', metadata['category'].values)
    df.insert(0, 'placeid', metadata['placeid'].values)
    df.insert(0, 'userid', metadata['userid'].values)
    df.to_parquet(output_path, index=False)
    print(f"Check-in embeddings: {output_path} {embeddings_np.shape}")

    return model_path
```

---

## 6. Step 4: Implement Inference Mode

### Add new function in: `src/embeddings/check2hgi/check2hgi.py`

```python
def inference_check2hgi(city, model_path, args, user_filter=None, output_suffix="_val"):
    """
    Run inference on Check2HGI model for new users (no training).

    This is used for generating embeddings for validation/test users
    using a model trained only on training users.

    Args:
        city: City/state name
        model_path: Path to trained model checkpoint
        args: Arguments (device, etc.)
        user_filter: Set of user IDs to generate embeddings for
        output_suffix: Suffix for output files (e.g., "_val", "_test")

    Returns:
        output_path: Path to saved embeddings
    """
    print(f"Running inference for {len(user_filter) if user_filter else 'all'} users")

    output_folder = IoPaths.CHECK2HGI.get_state_dir(city)

    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=args.device)
    in_channels = checkpoint['in_channels']
    model_args = argparse.Namespace(**checkpoint['args'])

    # Preprocess data for inference users
    preprocess_check2hgi(
        city=city,
        city_shapefile=args.shapefile,
        edge_type=model_args.edge_type,
        temporal_decay=model_args.temporal_decay,
        user_filter=user_filter,
    )

    # Load preprocessed data
    data_path = IoPaths.CHECK2HGI.get_graph_data_file(city)
    if user_filter:
        data_path = data_path.with_suffix('_train.pkl')  # Uses filtered data

    with open(data_path, 'rb') as handle:
        city_dict = pkl.load(handle)

    num_pois = city_dict['num_pois']
    num_regions = city_dict['num_regions']

    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(city_dict['node_features'], dtype=torch.float32),
        edge_index=torch.tensor(city_dict['edge_index'], dtype=torch.int64),
        edge_weight=torch.tensor(city_dict['edge_weight'], dtype=torch.float32),
        checkin_to_poi=torch.tensor(city_dict['checkin_to_poi'], dtype=torch.int64),
        poi_to_region=torch.tensor(city_dict['poi_to_region'], dtype=torch.int64),
        region_adjacency=torch.tensor(city_dict['region_adjacency'], dtype=torch.int64),
        region_area=torch.tensor(city_dict['region_area'], dtype=torch.float32),
        coarse_region_similarity=torch.tensor(city_dict['coarse_region_similarity'], dtype=torch.float32),
        num_pois=num_pois,
        num_regions=num_regions,
    ).to(args.device)

    metadata = city_dict['metadata']

    # Rebuild model architecture
    checkin_encoder = CheckinEncoder(in_channels, model_args.dim, num_layers=model_args.num_layers)
    checkin2poi = Checkin2POI(model_args.dim, model_args.attention_head)
    poi2region = POI2Region(model_args.dim, model_args.attention_head)

    def region2city(z, area):
        return torch.sigmoid((z.transpose(0, 1) * area).sum(dim=1))

    model = Check2HGI(
        hidden_channels=model_args.dim,
        checkin_encoder=checkin_encoder,
        checkin2poi=checkin2poi,
        poi2region=poi2region,
        region2city=region2city,
        corruption=corruption,
        alpha_c2p=model_args.alpha_c2p,
        alpha_p2r=model_args.alpha_p2r,
        alpha_r2c=model_args.alpha_r2c,
    ).to(args.device)

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Generate embeddings (inference only, no training)
    print("Generating embeddings (inference mode)...")
    with torch.no_grad():
        _ = model(data)
        checkin_emb, poi_emb, region_emb = model.get_embeddings()

    # Save embeddings
    output_path = IoPaths.get_embedd(city, EmbeddingEngine.CHECK2HGI)
    output_path = output_path.with_stem(output_path.stem + output_suffix)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embeddings_np = checkin_emb.numpy()
    df = pd.DataFrame(embeddings_np, columns=[f'{i}' for i in range(embeddings_np.shape[1])])
    df.insert(0, 'datetime', metadata['datetime'].values)
    df.insert(0, 'category', metadata['category'].values)
    df.insert(0, 'placeid', metadata['placeid'].values)
    df.insert(0, 'userid', metadata['userid'].values)
    df.to_parquet(output_path, index=False)

    print(f"Inference embeddings: {output_path} {embeddings_np.shape}")
    return output_path
```

---

## 7. Step 5: Modify HGI for POI-based Split

### Modify: `src/embeddings/hgi/preprocess.py`

Add `poi_filter` parameter for POI-based splits.

```python
# Modify HGIPreprocess.__init__() (around line 24)
def __init__(self, pois_filename, boroughs_filename, temp_path, poi_filter=None):
    self.pois_filename = pois_filename
    self.boroughs_filename = boroughs_filename
    self.temp_path = temp_path
    self.poi_filter = set(poi_filter) if poi_filter is not None else None


# Modify _read_poi_data() (around line 29)
def _read_poi_data(self):
    """Load and prepare POI data."""
    self.pois = pd.read_parquet(self.pois_filename)

    if "category" not in self.pois.columns:
        raise ValueError("Column 'category' missing from input data.")

    self.pois = self.pois.dropna(subset=["category"])

    # ADDED: Filter by POI IDs if specified (for inductive training)
    if self.poi_filter is not None:
        original_len = len(self.pois)
        self.pois = self.pois[self.pois['placeid'].isin(self.poi_filter)]
        print(f"Filtered to {len(self.pois)} POIs ({original_len - len(self.pois)} excluded)")

    # ... rest of the method unchanged


# Modify preprocess_hgi() (around line 226)
def preprocess_hgi(city, city_shapefile, poi_emb_path=None, cta_file=None, poi_filter=None):
    """Main preprocessing function for HGI.

    Args:
        city: City/state name
        city_shapefile: Path to census tract shapefile
        poi_emb_path: Optional path to pre-trained POI embeddings
        cta_file: Optional path to pre-computed boroughs file
        poi_filter: Optional set of POI IDs to include (for inductive training)
    """
    temp_folder = IoPaths.HGI.get_temp_dir(city)
    temp_folder.mkdir(parents=True, exist_ok=True)

    output_folder = IoPaths.HGI.get_output_dir(city)
    output_folder.mkdir(parents=True, exist_ok=True)

    checkins = IoPaths.get_city(city)
    boroughs_path = Path(cta_file) if cta_file else IoPaths.HGI.get_boroughs_file(city)

    # Create boroughs file if not exists
    if cta_file is None and not boroughs_path.exists():
        print(f"Creating boroughs file from shapefile: {city_shapefile}")
        census = gpd.read_file(city_shapefile).to_crs(4326)
        census[['GEOID', 'geometry']].to_csv(boroughs_path, index=False)

    pre = HGIPreprocess(
        str(checkins),
        str(boroughs_path),
        temp_folder,
        poi_filter=poi_filter,  # ADDED
    )
    data = pre.get_data_torch(poi_emb_path=poi_emb_path)

    # ADDED: Include split info in filename
    suffix = "_train" if poi_filter is not None else ""
    output_path = IoPaths.HGI.get_graph_data_file(city).with_suffix(f'{suffix}.pkl')

    with open(output_path, "wb") as f:
        pkl.dump(data, f)

    print(f"Saved: {output_path}")
    return output_path
```

---

## 8. Step 6: Update Input Creation

### Modify: `src/etl/create_input.py`

Add split-aware input generation.

```python
# Add new function after existing ones
def create_input_with_splits(
    state: str,
    embedding_engine: EmbeddingEngine,
    split: 'DataSplit',  # from data_split.py
    use_checkin_embeddings: bool = True,
):
    """
    Create input files respecting train/val/test splits.

    This ensures:
    - Train inputs use only train user embeddings
    - Val inputs use only val user embeddings (from inference)
    - Test inputs use only test user embeddings (from inference)

    Args:
        state: State name
        embedding_engine: Embedding engine type
        split: DataSplit object with train/val/test user IDs
        use_checkin_embeddings: Use check-in level embeddings
    """
    print(f"Creating split-aware inputs for {state}")

    for split_name, user_ids in [
        ('train', split.train_ids),
        ('val', split.val_ids),
        ('test', split.test_ids),
    ]:
        print(f"\nProcessing {split_name} split ({len(user_ids)} users)...")

        # Load embeddings for this split
        emb_path = IoPaths.get_embedd(state, embedding_engine)
        emb_path = emb_path.with_stem(emb_path.stem + f"_{split_name}")

        if not emb_path.exists():
            print(f"Warning: Embeddings not found at {emb_path}, skipping")
            continue

        embeddings_df = pd.read_parquet(emb_path)

        # Filter to only this split's users
        user_set = set(user_ids.tolist())
        embeddings_df = embeddings_df[embeddings_df['userid'].isin(user_set)]

        # Output paths with split suffix
        sequences_path = IoPaths.get_seq_next(state, embedding_engine)
        sequences_path = str(sequences_path).replace('.parquet', f'_{split_name}.parquet')

        next_input_path = IoPaths.get_next(state, embedding_engine)
        next_input_path = str(next_input_path).replace('.parquet', f'_{split_name}.parquet')

        category_input_path = IoPaths.get_category(state, embedding_engine)
        category_input_path = str(category_input_path).replace('.parquet', f'_{split_name}.parquet')

        # Create directories
        Path(sequences_path).parent.mkdir(parents=True, exist_ok=True)

        # Generate inputs
        if use_checkin_embeddings:
            generate_next_input_from_checkings(
                embeddings_df,
                sequences_path,
                next_input_path
            )
        else:
            checkins_df = IoPaths.load_city(state)
            checkins_df = checkins_df[checkins_df['userid'].isin(user_set)]
            generate_next_input_from_poi(
                embeddings_df,
                checkins_df,
                sequences_path,
                next_input_path
            )

        # Category input (if applicable)
        generate_category_input(embeddings_df.copy(), category_input_path)

        print(f"Created {split_name} inputs")
```

---

## 9. Step 7: Update Fold Creator

### Modify: `src/etl/create_fold.py`

Add support for pre-split data.

```python
# Add new method to FoldCreator class
def create_folds_from_splits(
    self,
    state: str,
    embedding_engine: EmbeddingEngine,
) -> Dict[int, FoldResult]:
    """
    Create folds using pre-split data files.

    Expects files with _train, _val, _test suffixes.
    Only creates train/val folds (test is held out for final evaluation).

    Args:
        state: State name
        embedding_engine: Embedding engine type

    Returns:
        Dictionary mapping fold_idx to FoldResult
    """
    logger.info(f"Creating folds from pre-split data for {state}")

    # Load train and val data
    X_train, y_train = self._load_split_data(state, embedding_engine, 'train')
    X_val, y_val = self._load_split_data(state, embedding_engine, 'val')

    # Convert to tensors
    x_train_tensor, y_train_tensor = _convert_to_tensors(X_train, y_train, self.task_type)
    x_val_tensor, y_val_tensor = _convert_to_tensors(X_val, y_val, self.task_type)

    # Create single fold (no cross-validation since split is already done)
    fold_results = {
        0: FoldResult()
    }

    if self.task_type in [TaskType.NEXT, TaskType.MTL]:
        fold_results[0].next = TaskFoldData(
            train=FoldData(
                _create_dataloader(x_train_tensor, y_train_tensor, self.batch_size, True, self.use_weighted_sampling, self.seed),
                x_train_tensor, y_train_tensor
            ),
            val=FoldData(
                _create_dataloader(x_val_tensor, y_val_tensor, self.batch_size, False, False, self.seed),
                x_val_tensor, y_val_tensor
            ),
        )

    if self.task_type in [TaskType.CATEGORY, TaskType.MTL]:
        # Load category data separately
        X_cat_train, y_cat_train = self._load_split_data(state, embedding_engine, 'train', task='category')
        X_cat_val, y_cat_val = self._load_split_data(state, embedding_engine, 'val', task='category')

        x_cat_train_tensor, y_cat_train_tensor = _convert_to_tensors(X_cat_train, y_cat_train, TaskType.CATEGORY)
        x_cat_val_tensor, y_cat_val_tensor = _convert_to_tensors(X_cat_val, y_cat_val, TaskType.CATEGORY)

        fold_results[0].category = TaskFoldData(
            train=FoldData(
                _create_dataloader(x_cat_train_tensor, y_cat_train_tensor, self.batch_size, True, self.use_weighted_sampling, self.seed),
                x_cat_train_tensor, y_cat_train_tensor
            ),
            val=FoldData(
                _create_dataloader(x_cat_val_tensor, y_cat_val_tensor, self.batch_size, False, False, self.seed),
                x_cat_val_tensor, y_cat_val_tensor
            ),
        )

    return fold_results


def _load_split_data(
    self,
    state: str,
    embedding_engine: EmbeddingEngine,
    split: str,  # 'train', 'val', 'test'
    task: str = 'next',  # 'next' or 'category'
) -> Tuple[np.ndarray, np.ndarray]:
    """Load data for a specific split."""
    if task == 'next':
        path = IoPaths.get_next(state, embedding_engine)
    else:
        path = IoPaths.get_category(state, embedding_engine)

    # Modify path with split suffix
    path = str(path).replace('.parquet', f'_{split}.parquet')
    df = pd.read_parquet(path)

    if task == 'next':
        df['label'] = _map_categories(df['next_category'])
        df = df.drop(columns=['userid', 'next_category'])
        expected_features = InputsConfig.EMBEDDING_DIM * InputsConfig.SLIDE_WINDOW
        feature_cols = df.columns[:expected_features]
    else:
        df['label'] = _map_categories(df['category'])
        df = df.drop(columns=['category'])
        feature_cols = list(map(str, range(InputsConfig.EMBEDDING_DIM)))

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    logger.info(f"Loaded {split} {task} data: {X.shape}")
    return X, y
```

---

## 10. Step 8: Update Pipelines

### Create new file: `pipelines/embedding/check2hgi_inductive.pipe.py`

```python
"""
Inductive Check2HGI Pipeline - No Data Leakage

This pipeline:
1. Splits users into train/val/test BEFORE embedding training
2. Trains embeddings ONLY on train users
3. Runs inference for val/test users
4. Creates separate input files for each split
"""

import argparse
from pathlib import Path

from configs.paths import IoPaths, EmbeddingEngine, Resources
from etl.data_split import DataSplitter, DataSplit
from embeddings.check2hgi.check2hgi import train_check2hgi, inference_check2hgi
from embeddings.check2hgi.preprocess import preprocess_check2hgi
from etl.create_input import create_input_with_splits


def run_inductive_pipeline(state: str, args):
    """
    Run the full inductive training pipeline.

    Steps:
    1. Load check-ins and split by users
    2. Preprocess and train on train users only
    3. Run inference for val/test users
    4. Create split-aware input files
    """
    print(f"\n{'='*60}")
    print(f"INDUCTIVE CHECK2HGI PIPELINE - {state}")
    print(f"{'='*60}\n")

    # Step 1: Split users
    print("Step 1: Splitting users...")
    checkins_df = IoPaths.load_city(state)

    splitter = DataSplitter(seed=args.seed)
    split = splitter.split_by_users(
        checkins_df,
        train_ratio=0.8,
        val_ratio=0.1,
        min_checkins_per_user=10,
    )

    # Save split for reproducibility
    split_path = IoPaths.CHECK2HGI.get_state_dir(state) / "user_split.json"
    split.save(split_path)

    # Step 2: Preprocess train data
    print("\nStep 2: Preprocessing train users...")
    preprocess_check2hgi(
        city=state,
        city_shapefile=str(args.shapefile),
        edge_type=args.edge_type,
        temporal_decay=args.temporal_decay,
        user_filter=split.get_train_set(),
    )

    # Step 3: Train on train users
    print("\nStep 3: Training on train users...")
    model_path = train_check2hgi(state, args, output_suffix="_train")

    # Step 4: Inference for val users
    print("\nStep 4: Running inference for val users...")
    inference_check2hgi(
        city=state,
        model_path=model_path,
        args=args,
        user_filter=split.get_val_set(),
        output_suffix="_val",
    )

    # Step 5: Inference for test users
    print("\nStep 5: Running inference for test users...")
    inference_check2hgi(
        city=state,
        model_path=model_path,
        args=args,
        user_filter=split.get_test_set(),
        output_suffix="_test",
    )

    # Step 6: Create input files
    print("\nStep 6: Creating split-aware input files...")
    create_input_with_splits(
        state=state,
        embedding_engine=EmbeddingEngine.CHECK2HGI,
        split=split,
        use_checkin_embeddings=True,
    )

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Split saved: {split_path}")
    print(f"Model saved: {model_path}")
    print(f"Train users: {len(split.train_ids)}")
    print(f"Val users: {len(split.val_ids)}")
    print(f"Test users: {len(split.test_ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inductive Check2HGI Pipeline')

    # Data
    parser.add_argument('--state', type=str, default='Alabama')
    parser.add_argument('--shapefile', type=str, default=str(Resources.TL_AL))
    parser.add_argument('--seed', type=int, default=42)

    # Preprocessing
    parser.add_argument('--edge_type', type=str, default='user_sequence')
    parser.add_argument('--temporal_decay', type=float, default=3600.0)

    # Model
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--attention_head', type=int, default=4)
    parser.add_argument('--alpha_c2p', type=float, default=0.4)
    parser.add_argument('--alpha_p2r', type=float, default=0.3)
    parser.add_argument('--alpha_r2c', type=float, default=0.3)

    # Training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_norm', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_amp', action='store_true', default=True)

    # Mini-batch
    parser.add_argument('--mini_batch_threshold', type=int, default=5_000_000)
    parser.add_argument('--batch_size', type=int, default=2**13)
    parser.add_argument('--num_neighbors', type=int, default=10)

    args = parser.parse_args()
    run_inductive_pipeline(args.state, args)
```

---

## 11. Testing Plan

### Unit Tests

Create `tests/test_data_split.py`:

```python
import pytest
import pandas as pd
import numpy as np
from etl.data_split import DataSplitter, DataSplit


def test_user_split_ratios():
    """Test that user split respects ratios."""
    # Create mock data
    checkins = pd.DataFrame({
        'userid': np.repeat(range(100), 20),  # 100 users, 20 check-ins each
        'placeid': np.random.randint(0, 50, 2000),
        'datetime': pd.date_range('2020-01-01', periods=2000, freq='H'),
    })

    splitter = DataSplitter(seed=42)
    split = splitter.split_by_users(checkins, train_ratio=0.8, val_ratio=0.1)

    assert len(split.train_ids) == 80
    assert len(split.val_ids) == 10
    assert len(split.test_ids) == 10


def test_no_user_overlap():
    """Test that splits have no overlapping users."""
    checkins = pd.DataFrame({
        'userid': np.repeat(range(100), 20),
        'placeid': np.random.randint(0, 50, 2000),
        'datetime': pd.date_range('2020-01-01', periods=2000, freq='H'),
    })

    splitter = DataSplitter(seed=42)
    split = splitter.split_by_users(checkins)

    train_set = set(split.train_ids)
    val_set = set(split.val_ids)
    test_set = set(split.test_ids)

    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0


def test_poi_split_stratified():
    """Test that POI split maintains category distribution."""
    pois = pd.DataFrame({
        'placeid': range(1000),
        'category': np.random.choice(['Food', 'Shop', 'Travel'], 1000),
    })

    splitter = DataSplitter(seed=42)
    split = splitter.split_by_pois(pois, stratify_by='category')

    # Check that all splits have all categories
    train_pois = pois[pois['placeid'].isin(split.train_ids)]
    val_pois = pois[pois['placeid'].isin(split.val_ids)]
    test_pois = pois[pois['placeid'].isin(split.test_ids)]

    assert set(train_pois['category'].unique()) == {'Food', 'Shop', 'Travel'}
    assert set(val_pois['category'].unique()) == {'Food', 'Shop', 'Travel'}
    assert set(test_pois['category'].unique()) == {'Food', 'Shop', 'Travel'}
```

### Integration Test

```python
def test_full_inductive_pipeline():
    """Test full pipeline runs without errors."""
    # Use smallest state for testing
    state = 'Alabama'

    # Run pipeline with minimal epochs
    args = Namespace(
        state=state,
        shapefile=str(Resources.TL_AL),
        seed=42,
        edge_type='user_sequence',
        temporal_decay=3600.0,
        dim=32,  # Smaller for testing
        num_layers=1,
        attention_head=2,
        epoch=5,  # Minimal epochs
        device='cpu',
        # ... other args
    )

    run_inductive_pipeline(state, args)

    # Verify outputs exist
    assert (IoPaths.CHECK2HGI.get_state_dir(state) / "user_split.json").exists()
    assert (IoPaths.CHECK2HGI.get_state_dir(state) / "model_train.pt").exists()

    # Verify separate embedding files
    emb_base = IoPaths.get_embedd(state, EmbeddingEngine.CHECK2HGI)
    assert emb_base.with_stem(emb_base.stem + "_train").exists()
    assert emb_base.with_stem(emb_base.stem + "_val").exists()
    assert emb_base.with_stem(emb_base.stem + "_test").exists()
```

---

## 12. Migration Guide

### For Existing Experiments

1. **Keep old results**: Don't delete existing results; they serve as baseline
2. **Run new pipeline**: Execute inductive pipeline for comparison
3. **Compare metrics**: Expect ~5-15% drop in validation metrics (this is realistic)

### Backwards Compatibility

The new code is backwards compatible:
- If `user_filter=None`, behaves like original (all data)
- Old pipelines continue to work unchanged
- New `*_inductive.pipe.py` files for new approach

### Recommended Workflow

```bash
# 1. Run inductive pipeline
python pipelines/embedding/check2hgi_inductive.pipe.py --state Alabama

# 2. Train MTL with split-aware data
python pipelines/train/mtl_inductive.pipe.py --state Alabama

# 3. Compare with old results
python scripts/compare_results.py --old results/old --new results/new
```

---

## Checklist

- [ ] Create `src/etl/data_split.py`
- [ ] Modify `src/embeddings/check2hgi/preprocess.py`
- [ ] Modify `src/embeddings/check2hgi/check2hgi.py`
- [ ] Add inference function to Check2HGI
- [ ] Modify `src/embeddings/hgi/preprocess.py`
- [ ] Modify `src/embeddings/hgi/hgi.py`
- [ ] Modify `src/etl/create_input.py`
- [ ] Modify `src/etl/create_fold.py`
- [ ] Create `pipelines/embedding/check2hgi_inductive.pipe.py`
- [ ] Create unit tests
- [ ] Run integration tests
- [ ] Compare results with baseline
