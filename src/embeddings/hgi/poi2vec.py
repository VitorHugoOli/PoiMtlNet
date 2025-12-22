"""POI2Vec: Pre-training POI embeddings using Node2Vec for HGI."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.nn import Node2Vec
from tqdm import tqdm

from configs.paths import IoPaths


class POI2Vec:
    """POI2Vec embedding generator using Node2Vec on spatial POI graphs."""

    def __init__(self, edges_file, pois_file, embedding_dim=64, walk_length=10,
                 context_size=5, walks_per_node=5, num_negative_samples=2,
                 p=0.5, q=0.5, device=None):
        self.embedding_dim = embedding_dim

        # Load data
        self.edges = pd.read_csv(edges_file)
        self.pois = pd.read_csv(pois_file)
        print(f"Loaded {len(self.edges)} edges, {len(self.pois)} POIs")

        # Create edge index tensor
        edge_index = torch.tensor(self.edges[["source", "target"]].T.values, dtype=torch.long)

        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Initialize Node2Vec model
        self.model = Node2Vec(
            edge_index,
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            sparse=True,
        ).to(self.device)

    def train(self, epochs=100, batch_size=128, lr=0.01, num_workers=0):
        """Train POI2Vec embeddings."""
        print(f"Training POI2Vec for {epochs} epochs...")

        loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0
            num_batches = 0

            with tqdm(loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
                for pos_rw, neg_rw in pbar:
                    optimizer.zero_grad()
                    loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            print(f"Epoch {epoch}/{epochs} - Avg loss: {total_loss / num_batches:.4f}")

        # Get embeddings
        self.model.eval()
        with torch.no_grad():
            return self.model().cpu().numpy()

    def save_embeddings(self, embeddings, output_path):
        """Save POI embeddings to file."""
        output_path = Path(output_path)
        print(f"Saving embeddings to: {output_path}")

        save_dict = {"in_embed.weight": torch.tensor(embeddings, dtype=torch.float32)}
        if "placeid" in self.pois.columns:
            save_dict["placeids"] = self.pois["placeid"].tolist()

        torch.save(save_dict, output_path)
        print(f"Saved embeddings: {embeddings.shape}")


def train_poi2vec(city, epochs=100, embedding_dim=64, batch_size=128, lr=0.01,device=None):
    """Train POI2Vec embeddings for a city."""
    temp_folder = IoPaths.HGI.get_temp_dir(city)

    if not temp_folder.exists():
        raise FileNotFoundError(f"HGI temp folder not found: {temp_folder}. Run preprocessing first.")

    edges_file = temp_folder / "edges.csv"
    pois_file = temp_folder / "pois.csv"

    if not edges_file.exists() or not pois_file.exists():
        raise FileNotFoundError("edges.csv or pois.csv not found. Run preprocessing first.")

    poi2vec = POI2Vec(edges_file=edges_file, pois_file=pois_file, embedding_dim=embedding_dim,device=device)
    embeddings = poi2vec.train(epochs=epochs, batch_size=batch_size, lr=lr)

    output_path = IoPaths.HGI.get_poi_emb_file(city)
    poi2vec.save_embeddings(embeddings, output_path)

    print(f"POI2Vec training completed. Embeddings saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train POI2Vec embeddings for HGI")
    parser.add_argument("--city", type=str, default="Alabama")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)

    args = parser.parse_args()

    print(f"POI2Vec: {args.city} | dim={args.dim} | epochs={args.epochs}")
    train_poi2vec(city=args.city, epochs=args.epochs, embedding_dim=args.dim,
                  batch_size=args.batch_size, lr=args.lr)
