import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.globals import DEVICE
from configs.model import InputsConfig
from configs.paths import IoPaths, EmbeddingEngine
from embeddings.time2vec.model.Time2VecModule import Time2VecContrastiveModel
from embeddings.time2vec.model.dataset import TemporalContrastiveDataset


def worker_init_fn(worker_id):
    """Initialize each worker with optimized settings."""
    worker_seed = torch.initial_seed() % 2 ** 32 + worker_id
    np.random.seed(worker_seed)


def train_epoch(model, dataloader, optimizer, device, tau):
    """Train for one epoch."""
    model.train()
    total_loss = torch.tensor(0.0, device=device)

    for t_i, t_j, label in dataloader:
        t_i = t_i.to(device=device, dtype=torch.float32)
        t_j = t_j.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        z_i, z_j = model(t_i, t_j)
        loss = model.contrastive_loss(z_i, z_j, label, tau=tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    return total_loss.item() / len(dataloader)


def create_embedding(state: str, args):
    """
    Train Time2Vec model and generate embeddings for a state.

    Args:
        state: State name
        args: Command line arguments with training parameters
    """
    # Setup output folder
    output_folder = IoPaths.TIME2VEC.get_state_dir(state)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load check-ins
    checkins_path = IoPaths.get_city(state)
    print(f"Loading check-ins from: {checkins_path}")
    checkins = pd.read_parquet(checkins_path)
    print(f"Loaded {len(checkins)} check-ins")

    # Validate required columns
    required_cols = ["local_datetime", "category", "placeid"]
    for col in required_cols:
        if col not in checkins.columns:
            raise ValueError(f"Required column missing: {col}")

    # Extract time features
    print("Extracting temporal features...")
    time_hours, time_feats = TemporalContrastiveDataset.extract_time_features(checkins)

    # Filter checkins to match valid datetime entries
    valid_mask = pd.to_datetime(
        checkins["local_datetime"].astype(str), utc=True, errors="coerce"
    ).notna()
    checkins_valid = checkins[valid_mask].reset_index(drop=True)

    categories = checkins_valid["category"].astype(str).values
    placeids = checkins_valid["placeid"].astype(str).values
    userids = checkins_valid["userid"].values
    datetimes = checkins_valid["datetime"].values
    latitudes = checkins_valid["latitude"].values
    longitudes = checkins_valid["longitude"].values

    print(f"Valid check-ins: {len(checkins_valid)}")
    print(f"Time features shape: {time_feats.shape}")

    # Create contrastive dataset
    print("Creating contrastive dataset...")
    dataset = TemporalContrastiveDataset(
        time_hours=time_hours,
        time_feats=time_feats,
        r_pos_hours=args.r_pos_hours,
        r_neg_hours=args.r_neg_hours,
        max_pairs=args.max_pairs,
        k_neg_per_i=args.k_neg_per_i,
        max_pos_per_i=args.max_pos_per_i,
        seed=args.seed,
    )

    # Optimized DataLoader configuration
    num_workers = min(8, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory_device=str(DEVICE) if hasattr(DEVICE, 'index') else None,
        persistent_workers=True,
        prefetch_factor=5,
        worker_init_fn=worker_init_fn,
    )

    # Initialize model
    print(f"Initializing model: activation={args.activation}, "
          f"out_features={args.out_features}, embed_dim={args.dim}")

    model = Time2VecContrastiveModel(
        activation=args.activation,
        out_features=args.out_features,
        embed_dim=args.dim,
        in_features=2,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Training for {args.epoch} epochs on {args.device}...")
    bar = tqdm(range(1, args.epoch + 1), desc="Training")

    for epoch in bar:
        avg_loss = train_epoch(model, dataloader, optimizer, args.device, args.tau)
        bar.set_postfix(loss=f"{avg_loss:.4f}")

    # Save trained model
    model_path = IoPaths.TIME2VEC.get_model_file(state)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Generate embeddings for all check-ins
    print("Generating embeddings...")
    model.eval()
    with torch.no_grad():
        t_tensor = torch.from_numpy(time_feats).to(args.device)
        embeddings = model.encode(t_tensor).cpu().numpy()

    # Save embeddings
    output_path = IoPaths.get_embedd(state, EmbeddingEngine.TIME2VEC)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(embeddings, columns=[f'{i}' for i in range(embeddings.shape[1])])
    df.insert(0, 'userid', userids)
    df.insert(1, 'placeid', placeids)
    df.insert(2, 'datetime', datetimes)
    df.insert(3, 'category', categories)
    df.insert(4, 'latitude', latitudes)
    df.insert(5, 'longitude', longitudes)
    df.to_parquet(output_path, index=False)

    print(f"Embeddings saved to: {output_path}")
    print(f"Embedding dimensions: {embeddings.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Time2Vec embeddings for POI data')
    parser.add_argument('--state', type=str, default='Texas',
                        help='State name for processing')
    parser.add_argument('--dim', type=int, default=InputsConfig.EMBEDDING_DIM,
                        help='Dimension of output embedding')
    parser.add_argument('--out_features', type=int, default=64,
                        help='Dimension of Time2Vec layer output')
    parser.add_argument('--activation', type=str, default='sin',
                        choices=['sin', 'cos'],
                        help='Periodic activation function')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--r_pos_hours', type=float, default=1.0,
                        help='Radius in hours for positive pairs')
    parser.add_argument('--r_neg_hours', type=float, default=24.0,
                        help='Minimum distance in hours for negative pairs')
    parser.add_argument('--max_pairs', type=int, default=2_000_000,
                        help='Maximum number of pairs to generate')
    parser.add_argument('--k_neg_per_i', type=int, default=5,
                        help='Number of negative pairs per anchor')
    parser.add_argument('--max_pos_per_i', type=int, default=20,
                        help='Maximum positive pairs per anchor')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--tau', type=float, default=0.3,
                        help='Temperature for contrastive loss')
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='Device to use for training')

    args = parser.parse_args()
    create_embedding(state=args.state, args=args)