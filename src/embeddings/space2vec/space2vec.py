"""
Space2Vec: Spatial contrastive learning for location embeddings.

This module trains a Space2Vec model using contrastive learning on
spatial proximity pairs and generates location embeddings for POIs.
"""

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
from embeddings.space2vec.model.SpaceEncoder import SpaceContrastiveModel
from embeddings.space2vec.model.dataset import (
    PairsMemmapDataset,
    build_pairs_memmap,
    to_xy_km,
)


def worker_init_fn(worker_id: int):
    """Initialize each worker with a unique seed."""
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)


def train_epoch(
    model: SpaceContrastiveModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    tau: float,
) -> float:
    """
    Train for one epoch.

    Args:
        model: SpaceContrastiveModel instance
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Device for training
        tau: Temperature for contrastive loss

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for coord_i, coord_j, label in dataloader:
        # Convert to numpy for coordinate transformation
        ci = coord_i.numpy()
        cj = coord_j.numpy()

        # Convert lat/lon to XY in kilometers
        XY_i = to_xy_km(ci)
        XY_j = to_xy_km(cj)

        # Compute deltas and add noise for augmentation
        deltas_km = torch.from_numpy(XY_j - XY_i).float().to(device)
        noise = torch.randn_like(deltas_km) * 0.01

        # Forward pass with augmented deltas
        z_i = model(deltas_km + noise)
        z_j = model(deltas_km - noise)

        # Compute loss
        loss = model.contrastive_loss(z_i, z_j, label.to(device), tau=tau)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def create_embedding(state: str, args):
    """
    Train Space2Vec model and generate embeddings for a state.

    Args:
        state: State name
        args: Command line arguments with training parameters
    """
    # Setup output directories
    output_folder = IoPaths.SPACE2VEC.get_state_dir(state)
    output_folder.mkdir(parents=True, exist_ok=True)

    pairs_dir = IoPaths.SPACE2VEC.get_pairs_dir(state)
    pairs_dir.mkdir(parents=True, exist_ok=True)

    # Load check-ins
    checkins_path = IoPaths.get_city(state)
    print(f"Loading check-ins from: {checkins_path}")
    checkins = pd.read_parquet(checkins_path)
    print(f"Loaded {len(checkins)} check-ins")

    # Validate required columns
    required_cols = ["latitude", "longitude", "category", "placeid"]
    for col in required_cols:
        if col not in checkins.columns:
            raise ValueError(f"Required column missing: {col}")

    # Extract coordinates and metadata
    coords = checkins[["latitude", "longitude"]].values.astype(np.float32)
    categories = checkins["category"].astype(str).values
    placeids = checkins["placeid"].astype(str).values

    print(f"Coordinates shape: {coords.shape}")

    # Generate spatial proximity pairs
    pairs_count_file = pairs_dir / "pairs_count.npy"
    if pairs_count_file.exists() and not args.force_pairs:
        print(f"Loading existing pairs from: {pairs_dir}")
        n_pairs = int(np.load(pairs_count_file)[0])
    else:
        print("Generating spatial proximity pairs...")
        n_pairs = build_pairs_memmap(
            coords_deg=coords,
            out_dir=pairs_dir,
            max_pairs=args.max_pairs,
            r_pos_km=args.r_pos_km,
            r_neg_km=args.r_neg_km,
            k_pos_per_i=args.k_pos_per_i,
            k_neg_per_i=args.k_neg_per_i,
            seed=args.seed,
        )

    print(f"Total pairs: {n_pairs}")

    # Create dataset and dataloader
    dataset = PairsMemmapDataset(coords, pairs_dir)
    num_workers = min(8, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )

    # Initialize model
    print(
        f"Initializing model: embed_dim={args.dim}, spa_embed_dim={args.spa_embed_dim}, "
        f"freq_num={args.freq_num}"
    )

    model = SpaceContrastiveModel(
        embed_dim=args.dim,
        spa_embed_dim=args.spa_embed_dim,
        coord_dim=2,
        frequency_num=args.freq_num,
        max_radius=args.max_radius,
        min_radius=args.min_radius,
        ffn_act="leakyrelu",
        freq_init="geometric",
        ffn_num_hidden_layers=1,
        ffn_dropout_rate=0.5,
        ffn_use_layernormalize=True,
        ffn_skip_connection=True,
        ffn_hidden_dim=512,
        device=str(args.device),
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"Training for {args.epoch} epochs on {args.device}...")
    bar = tqdm(range(1, args.epoch + 1), desc="Training")

    for epoch in bar:
        avg_loss = train_epoch(model, dataloader, optimizer, args.device, args.tau)
        bar.set_postfix(loss=f"{avg_loss:.4f}")

    # Save trained model
    model_path = IoPaths.SPACE2VEC.get_model_file(state)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Generate embeddings for all coordinates
    # NOTE: The notebook uses raw lat/lon coordinates for embedding generation,
    # even though training uses XY deltas. We match this for compatibility.
    print("Generating embeddings...")
    model.eval()

    # Process in batches for memory efficiency
    batch_size = 10000
    all_embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(coords), batch_size), desc="Embedding"):
            end = min(start + batch_size, len(coords))
            batch_coords = coords[start:end]

            # Use raw lat/lon coordinates (same as notebook cell 14)
            batch_tensor = torch.from_numpy(batch_coords).float().to(args.device)

            embeddings = model.encode(batch_tensor).cpu().numpy()
            all_embeddings.append(embeddings)

    embeddings = np.vstack(all_embeddings)

    # Save embeddings
    output_path = IoPaths.get_embedd(state, EmbeddingEngine.SPACE2VEC)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(embeddings, columns=[f"{i}" for i in range(embeddings.shape[1])])
    df.insert(0, "placeid", placeids)
    df.insert(1, "category", categories)
    df.to_parquet(output_path, index=False)

    print(f"Embeddings saved to: {output_path}")
    print(f"Embedding dimensions: {embeddings.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Space2Vec embeddings for POI data"
    )
    parser.add_argument(
        "--state", type=str, default="Florida", help="State name for processing"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=InputsConfig.EMBEDDING_DIM,
        help="Dimension of output embedding",
    )
    parser.add_argument(
        "--spa_embed_dim",
        type=int,
        default=128,
        help="Spatial encoder hidden dimension",
    )
    parser.add_argument(
        "--freq_num", type=int, default=16, help="Number of frequency bands"
    )
    parser.add_argument(
        "--max_radius",
        type=float,
        default=50,
        help="Max radius for position encoding",
    )
    parser.add_argument(
        "--min_radius",
        type=float,
        default=0.02,
        help="Min radius for position encoding",
    )
    parser.add_argument(
        "--epoch", type=int, default=40, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--r_pos_km",
        type=float,
        default=10.0,
        help="Positive pair radius in kilometers",
    )
    parser.add_argument(
        "--r_neg_km",
        type=float,
        default=70.0,
        help="Negative pair minimum distance in kilometers",
    )
    parser.add_argument(
        "--k_pos_per_i", type=int, default=8, help="Max positive pairs per anchor"
    )
    parser.add_argument(
        "--k_neg_per_i", type=int, default=16, help="Max negative pairs per anchor"
    )
    parser.add_argument(
        "--max_pairs", type=int, default=2_000_000, help="Maximum pairs to generate"
    )
    parser.add_argument(
        "--tau", type=float, default=0.15, help="Temperature for contrastive loss"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default=str(DEVICE),
        help="Device to use for training (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--force_pairs",
        action="store_true",
        help="Force regeneration of pairs even if they exist",
    )

    args = parser.parse_args()

    # Convert device string to torch device
    args.device = torch.device(args.device)

    create_embedding(state=args.state, args=args)
