"""Check2HGI preprocessing module for creating check-in graph structures."""

import argparse
import pickle as pkl
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt
from sklearn.preprocessing import LabelEncoder

from configs.paths import IoPaths, Resources


class Check2HGIPreprocess:
    """Preprocessing pipeline for Check2HGI embeddings."""

    def __init__(self, checkins_file, boroughs_file, temp_path, edge_type='user_sequence',
                 temporal_decay=3600.0):
        """
        Initialize preprocessor.

        Args:
            checkins_file: Path to check-in parquet file
            boroughs_file: Path to boroughs/regions CSV file
            temp_path: Path to save intermediate files
            edge_type: Type of edges ('user_sequence', 'same_poi', 'both')
            temporal_decay: Decay parameter for temporal edge weights (seconds)
        """
        self.checkins_file = checkins_file
        self.boroughs_file = boroughs_file
        self.temp_path = Path(temp_path)
        self.edge_type = edge_type
        self.temporal_decay = temporal_decay

    def _load_checkins(self):
        """Load and prepare check-in data."""
        print("Loading check-in data...")
        self.checkins = pd.read_parquet(self.checkins_file)

        # Ensure required columns exist
        required = ['userid', 'placeid', 'datetime', 'category', 'latitude', 'longitude']
        missing = [c for c in required if c not in self.checkins.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by user and time (essential for user_sequence edges)
        self.checkins = self.checkins.sort_values(['userid', 'datetime']).reset_index(drop=True)

        # Encode categories
        self.le_category = LabelEncoder()
        self.checkins['category_encoded'] = self.le_category.fit_transform(
            self.checkins['category'].fillna('Unknown')
        )

        # Create geometry for spatial join
        self.checkins['geometry'] = self.checkins.apply(
            lambda row: f"POINT({row['longitude']} {row['latitude']})"
            if pd.notnull(row['longitude']) and pd.notnull(row['latitude'])
            else None,
            axis=1
        )

        print(f"Loaded {len(self.checkins)} check-ins, {self.checkins['placeid'].nunique()} POIs")

    def _load_boroughs(self):
        """Load boroughs/regions data."""
        print("Loading boroughs data...")
        self.boroughs = pd.read_csv(self.boroughs_file)
        self.boroughs['geometry'] = self.boroughs['geometry'].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry='geometry', crs='EPSG:4326')

    def _assign_regions(self):
        """Assign each check-in and POI to a region via spatial join."""
        print("Assigning regions...")

        # Create unique POI dataframe with representative coordinates
        pois = self.checkins.groupby('placeid').agg({
            'latitude': 'first',
            'longitude': 'first',
            'category_encoded': 'first',
            'geometry': 'first'
        }).reset_index()

        # Convert to GeoDataFrame
        pois['geometry'] = pois['geometry'].apply(wkt.loads)
        pois = gpd.GeoDataFrame(pois, geometry='geometry', crs='EPSG:4326')

        # Spatial join with boroughs
        pois = pois.sjoin(self.boroughs, how='left', predicate='intersects')

        # Handle POIs outside any region (assign to nearest or drop)
        pois = pois.dropna(subset=['GEOID'])

        # Create POI to region mapping
        unique_regions = pois['GEOID'].unique()
        self.region_to_idx = {r: i for i, r in enumerate(unique_regions)}
        pois['region_idx'] = pois['GEOID'].map(self.region_to_idx)

        # Store POI info
        self.pois = pois
        self.num_regions = len(unique_regions)

        # Create placeid to POI index mapping
        valid_placeids = set(pois['placeid'].tolist())
        self.placeid_to_idx = {pid: i for i, pid in enumerate(pois['placeid'].tolist())}

        # Filter check-ins to only those with valid POIs
        self.checkins = self.checkins[self.checkins['placeid'].isin(valid_placeids)].reset_index(drop=True)

        # Map check-ins to POI indices
        self.checkins['poi_idx'] = self.checkins['placeid'].map(self.placeid_to_idx)

        print(f"Assigned to {self.num_regions} regions, {len(self.pois)} POIs")

    def _build_user_sequence_edges(self):
        """Build edges connecting consecutive check-ins by the same user."""
        print("Building user sequence edges...")

        edges = []
        weights = []

        # Group by user and iterate
        for userid, group in self.checkins.groupby('userid'):
            if len(group) < 2:
                continue

            indices = group.index.tolist()
            times = group['datetime'].tolist()

            for i in range(len(indices) - 1):
                src = indices[i]
                tgt = indices[i + 1]

                # Compute temporal weight (exponential decay)
                time_gap = (times[i + 1] - times[i]).total_seconds()
                weight = np.exp(-time_gap / self.temporal_decay)

                # Bidirectional edges
                edges.append([src, tgt])
                edges.append([tgt, src])
                weights.append(weight)
                weights.append(weight)

        if edges:
            edge_index = np.array(edges).T
            edge_weight = np.array(weights)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros(0, dtype=np.float32)

        print(f"Built {len(weights)} user sequence edges")
        return edge_index, edge_weight

    def _build_same_poi_edges(self, max_edges_per_poi=50):
        """Build edges connecting check-ins at the same POI."""
        print("Building same-POI edges...")

        edges = []
        weights = []

        # Group by POI
        for placeid, group in self.checkins.groupby('placeid'):
            if len(group) < 2:
                continue

            indices = group.index.tolist()

            # Limit edges per POI to avoid explosion
            if len(indices) > max_edges_per_poi:
                # Sample subset
                sampled = np.random.choice(indices, max_edges_per_poi, replace=False).tolist()
            else:
                sampled = indices

            # Connect all pairs in sample
            for i, src in enumerate(sampled):
                for tgt in sampled[i + 1:]:
                    edges.append([src, tgt])
                    edges.append([tgt, src])
                    weights.append(1.0)
                    weights.append(1.0)

        if edges:
            edge_index = np.array(edges).T
            edge_weight = np.array(weights)
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
            edge_weight = np.zeros(0, dtype=np.float32)

        print(f"Built {len(weights)} same-POI edges")
        return edge_index, edge_weight

    def _build_edges(self):
        """Build edges based on edge_type."""
        if self.edge_type == 'user_sequence':
            return self._build_user_sequence_edges()
        elif self.edge_type == 'same_poi':
            return self._build_same_poi_edges()
        elif self.edge_type == 'both':
            e1, w1 = self._build_user_sequence_edges()
            e2, w2 = self._build_same_poi_edges()
            edge_index = np.concatenate([e1, e2], axis=1)
            edge_weight = np.concatenate([w1, w2])
            return edge_index, edge_weight
        else:
            raise ValueError(f"Unknown edge_type: {self.edge_type}")

    def _build_node_features(self):
        """Build check-in node features: category one-hot + temporal encoding."""
        print("Building node features...")

        num_checkins = len(self.checkins)
        num_categories = len(self.le_category.classes_)

        # Category one-hot
        category_onehot = np.zeros((num_checkins, num_categories), dtype=np.float32)
        category_onehot[np.arange(num_checkins), self.checkins['category_encoded'].values] = 1.0

        # Temporal encoding (sin/cos for hour and day of week)
        # Parse datetime
        dt = pd.to_datetime(self.checkins['datetime'])
        hour = dt.dt.hour.values
        dow = dt.dt.dayofweek.values

        temporal = np.zeros((num_checkins, 4), dtype=np.float32)
        temporal[:, 0] = np.sin(2 * np.pi * hour / 24)  # hour_sin
        temporal[:, 1] = np.cos(2 * np.pi * hour / 24)  # hour_cos
        temporal[:, 2] = np.sin(2 * np.pi * dow / 7)    # dow_sin
        temporal[:, 3] = np.cos(2 * np.pi * dow / 7)    # dow_cos

        # Concatenate
        node_features = np.concatenate([category_onehot, temporal], axis=1)

        print(f"Node features shape: {node_features.shape}")
        return node_features

    def _compute_region_features(self):
        """Compute region-level features: areas, adjacency, and similarity."""
        print("Computing region features...")

        # Unique regions with POIs
        unique_regions = list(self.region_to_idx.keys())

        # Compute region areas
        region_area = np.zeros(self.num_regions, dtype=np.float32)
        for geoid, idx in self.region_to_idx.items():
            region_row = self.boroughs[self.boroughs['GEOID'] == geoid]
            if len(region_row) > 0:
                region_area[idx] = region_row.geometry.iloc[0].area

        # Compute adjacency
        region_adjacency = self._compute_adjacency_matrix(unique_regions)

        # Compute similarity (simple: based on POI counts)
        coarse_similarity = self._compute_similarity_matrix()

        return region_area, region_adjacency, coarse_similarity

    def _compute_adjacency_matrix(self, unique_regions):
        """Compute spatial adjacency matrix between regions."""
        valid_regions = self.boroughs[self.boroughs['GEOID'].isin(unique_regions)].copy()
        valid_regions['idx'] = valid_regions['GEOID'].map(self.region_to_idx)

        joined = gpd.sjoin(valid_regions, valid_regions, how='inner', predicate='intersects')
        edges = joined[joined['idx_left'] != joined['idx_right']]

        if len(edges) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.vstack([
            edges['idx_left'].values.astype(np.int64),
            edges['idx_right'].values.astype(np.int64)
        ])

    def _compute_similarity_matrix(self):
        """Compute region similarity based on shared POI categories."""
        similarity = np.eye(self.num_regions, dtype=np.float32) * 0.5

        # Count category distribution per region
        poi_region = self.pois[['region_idx', 'category_encoded']].values
        num_categories = len(self.le_category.classes_)

        region_cat_counts = np.zeros((self.num_regions, num_categories))
        for region_idx, cat_idx in poi_region:
            region_cat_counts[int(region_idx), int(cat_idx)] += 1

        # Normalize
        row_sums = region_cat_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        region_cat_dist = region_cat_counts / row_sums

        # Cosine similarity
        norms = np.linalg.norm(region_cat_dist, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = region_cat_dist / norms
        similarity = np.dot(normalized, normalized.T)

        return similarity.astype(np.float32)

    def get_data(self):
        """Execute full preprocessing pipeline and return data dict."""
        self._load_checkins()
        self._load_boroughs()
        self._assign_regions()

        # Build edges
        edge_index, edge_weight = self._build_edges()

        # Normalize edge weights
        if len(edge_weight) > 0:
            mi, ma = edge_weight.min(), edge_weight.max()
            if ma > mi:
                edge_weight = (edge_weight - mi) / (ma - mi)

        # Build node features
        node_features = self._build_node_features()

        # Build hierarchical mappings
        checkin_to_poi = self.checkins['poi_idx'].values.astype(np.int64)
        poi_to_region = self.pois['region_idx'].values.astype(np.int64)

        # Compute region features
        region_area, region_adjacency, coarse_similarity = self._compute_region_features()

        # Metadata for output
        metadata = self.checkins[['userid', 'placeid', 'datetime']].copy()

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weight': edge_weight.astype(np.float32),
            'checkin_to_poi': checkin_to_poi,
            'poi_to_region': poi_to_region,
            'region_adjacency': region_adjacency,
            'region_area': region_area,
            'coarse_region_similarity': coarse_similarity,
            'num_checkins': len(self.checkins),
            'num_pois': len(self.pois),
            'num_regions': self.num_regions,
            'metadata': metadata,
            'placeid_to_idx': self.placeid_to_idx,
            'region_to_idx': self.region_to_idx,
        }


def preprocess_check2hgi(city, city_shapefile, edge_type='user_sequence',
                          temporal_decay=3600.0, cta_file=None):
    """
    Main preprocessing function for Check2HGI.

    Args:
        city: City/state name
        city_shapefile: Path to census tract shapefile
        edge_type: Type of edges ('user_sequence', 'same_poi', 'both')
        temporal_decay: Decay parameter for temporal edge weights
        cta_file: Optional path to pre-computed boroughs file
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
    )

    data = pre.get_data()

    output_path = IoPaths.CHECK2HGI.get_graph_data_file(city)
    with open(output_path, 'wb') as f:
        pkl.dump(data, f)

    print(f"Saved: {output_path}")
    print(f"Check-ins: {data['num_checkins']}, POIs: {data['num_pois']}, "
          f"Regions: {data['num_regions']}, Edges: {len(data['edge_weight'])}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess check-in data for Check2HGI')
    parser.add_argument('--city', type=str, default='Texas')
    parser.add_argument('--shapefile', type=str, default=str(Resources.TL_TX))
    parser.add_argument('--edge_type', type=str, default='user_sequence',
                        choices=['user_sequence', 'same_poi', 'both'])
    parser.add_argument('--temporal_decay', type=float, default=3600.0,
                        help='Decay parameter for temporal edge weights (seconds)')
    parser.add_argument('--cta', type=str, default=None)

    args = parser.parse_args()
    preprocess_check2hgi(
        city=args.city,
        city_shapefile=args.shapefile,
        edge_type=args.edge_type,
        temporal_decay=args.temporal_decay,
        cta_file=args.cta
    )
