"""HGI preprocessing module for creating hierarchical graph structures."""

import argparse
import pickle as pkl
from itertools import combinations
from pathlib import Path
from warnings import simplefilter

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.spatial
import torch
from shapely import wkt
from sklearn.preprocessing import LabelEncoder

from configs.paths import IoPaths, Resources
from embeddings.hgi.utils import SpatialUtils, mode_or_first


class HGIPreprocess:
    """Preprocessing pipeline for HGI embeddings."""

    def __init__(self, pois_filename, boroughs_filename, temp_path):
        self.pois_filename = pois_filename
        self.boroughs_filename = boroughs_filename
        self.temp_path = temp_path

    def _read_poi_data(self):
        """Load and prepare POI data."""
        self.pois = pd.read_parquet(self.pois_filename)

        if "category" not in self.pois.columns:
            raise ValueError("Column 'category' missing from input data.")

        self.pois = self.pois.dropna(subset=["category"])

        # Create geometry column if not exists
        if 'geometry' not in self.pois.columns:
            self.pois['geometry'] = self.pois.apply(
                lambda x: f"POINT({x['longitude']} {x['latitude']})"
                if pd.notnull(x['longitude']) and pd.notnull(x['latitude'])
                else None,
                axis=1
            )

        # Handle duplicate placeids
        if not self.pois['placeid'].is_unique:
            agg_funcs = {
                'category': mode_or_first,
                'longitude': 'first',
                'latitude': 'first',
                'geometry': 'first'
            }
            for col in self.pois.columns:
                if col not in agg_funcs and col != 'placeid':
                    agg_funcs[col] = 'first'
            self.pois = self.pois.groupby('placeid').agg(agg_funcs).reset_index()

        # Convert to GeoDataFrame
        self.pois["geometry"] = self.pois["geometry"].apply(wkt.loads)
        self.pois = gpd.GeoDataFrame(self.pois, geometry="geometry", crs="EPSG:4326")
        self.pois["geometry"] = self.pois["geometry"].apply(
            lambda x: x if x.geom_type == "Point" else x.centroid
        )

    def _read_boroughs_data(self):
        """Load boroughs/regions data and encode categories."""
        self.boroughs = pd.read_csv(self.boroughs_filename)
        self.boroughs["geometry"] = self.boroughs["geometry"].apply(wkt.loads)
        self.boroughs = gpd.GeoDataFrame(self.boroughs, geometry="geometry", crs="EPSG:4326")

        # Encode categories
        le = LabelEncoder()
        self.pois["category"] = le.fit_transform(self.pois["category"].values)

        # Spatial join with boroughs
        self.pois = self.pois[['placeid', 'category', 'geometry']].sjoin(
            self.boroughs, how='inner', predicate='intersects'
        )

    def _create_graph(self):
        """Create spatial graph using Delaunay triangulation."""
        points = np.array(self.pois.geometry.apply(lambda x: [x.x, x.y]).tolist())
        D = SpatialUtils.diagonal_length_bbox(self.pois.geometry.unary_union.envelope.bounds)

        # Delaunay triangulation
        triangles = scipy.spatial.Delaunay(points, qhull_options="QJ QbB Pp").simplices

        # Build edge list with weights
        edges = []
        seen = set()
        for simplex in triangles:
            for x, y in combinations(simplex, 2):
                if (x, y) not in seen and (y, x) not in seen:
                    seen.add((x, y))
                    dist = SpatialUtils.haversine_np(*points[x], *points[y])
                    w1 = np.log((1 + D ** 1.5) / (1 + dist ** 1.5))
                    w2 = 1.0 if self.pois.iloc[x]["GEOID"] == self.pois.iloc[y]["GEOID"] else 0.4
                    edges.append({'source': x, 'target': y, 'weight': w1 * w2})

        self.edges = pd.DataFrame(edges)

        # Normalize weights
        mi, ma = self.edges['weight'].min(), self.edges['weight'].max()
        self.edges['weight'] = (self.edges['weight'] - mi) / (ma - mi)

        # Save for POI2Vec
        self.edges.to_csv(str(self.temp_path / 'edges.csv'), index=False)
        self.pois[['placeid']].to_csv(str(self.temp_path / 'pois.csv'), index=False)

    def _compute_region_features(self):
        """Compute region-level features: areas, adjacency, and similarity."""
        unique_regions = self.pois['GEOID'].unique()
        region_to_idx = {region: idx for idx, region in enumerate(unique_regions)}

        # Map POIs to region indices
        self.region_id = self.pois['GEOID'].map(region_to_idx).values

        # Compute region areas
        self.region_area = np.array([
            self.boroughs[self.boroughs['GEOID'] == r].geometry.iloc[0].area
            for r in unique_regions
        ], dtype=np.float32)

        # Compute adjacency and similarity
        self.region_adjacency = self._compute_adjacency_matrix(unique_regions, region_to_idx)
        self.coarse_region_similarity = self._compute_similarity_matrix(region_to_idx, len(unique_regions))

    def _compute_adjacency_matrix(self, unique_regions, region_to_idx):
        """Compute spatial adjacency matrix between regions."""
        valid_regions = self.boroughs[self.boroughs['GEOID'].isin(unique_regions)].copy()
        valid_regions['idx'] = valid_regions['GEOID'].map(region_to_idx)

        joined = gpd.sjoin(valid_regions, valid_regions, how='inner', predicate='intersects')
        edges = joined[joined['idx_left'] != joined['idx_right']]

        if len(edges) == 0:
            return np.zeros((2, 0), dtype=np.int64)

        return np.vstack([
            edges['idx_left'].values.astype(np.int64),
            edges['idx_right'].values.astype(np.int64)
        ])

    def _compute_similarity_matrix(self, region_to_idx, num_regions):
        """Compute region similarity based on edge co-occurrence."""
        similarity = np.zeros((num_regions, num_regions), dtype=np.float32)

        source_regions = self.region_id[self.edges['source'].values]
        target_regions = self.region_id[self.edges['target'].values]
        weights = self.edges['weight'].values

        df = pd.DataFrame({'src': source_regions, 'tgt': target_regions, 'w': weights})
        agg = df.groupby(['src', 'tgt'])['w'].sum().reset_index()

        for _, row in agg.iterrows():
            i, j = int(row['src']), int(row['tgt'])
            similarity[i, j] += row['w']
            if i != j:
                similarity[j, i] += row['w']

        # Normalize
        row_sums = similarity.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return similarity / row_sums

    def _load_poi_embeddings(self, poi_emb_path):
        """Load pre-trained POI embeddings or use one-hot encoding."""
        if poi_emb_path is None or not Path(poi_emb_path).exists():
            simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
            return pd.get_dummies(self.pois['category'], dtype=int).values.astype(np.float32)

        blob = torch.load(poi_emb_path, map_location="cpu")

        if "in_embed.weight" in blob:
            embeddings = blob["in_embed.weight"].detach().cpu().numpy()
            placeids = blob.get("placeids", [])
        elif "embeddings" in blob:
            embeddings = blob["embeddings"].detach().cpu().numpy()
            placeids = blob.get("placeids", [])
        else:
            raise ValueError("Unknown embedding format")

        placeid_to_idx = {str(pid): i for i, pid in enumerate(placeids)}
        X = np.zeros((len(self.pois), embeddings.shape[1]), dtype=np.float32)

        for i, pid in enumerate(self.pois['placeid'].tolist()):
            if str(pid) in placeid_to_idx:
                X[i] = embeddings[placeid_to_idx[str(pid)]]

        return X

    def get_data_torch(self, poi_emb_path=None):
        """Execute full preprocessing pipeline."""
        print("Reading POI data...")
        self._read_poi_data()

        print("Reading boroughs data...")
        self._read_boroughs_data()

        print("Creating spatial graph...")
        self._create_graph()

        print("Computing region features...")
        self._compute_region_features()

        print("Loading POI embeddings...")
        node_features = self._load_poi_embeddings(poi_emb_path)

        return {
            'node_features': node_features,
            'edge_index': self.edges[["source", "target"]].T.values,
            'edge_weight': self.edges["weight"].values,
            'region_id': self.region_id,
            'region_area': self.region_area,
            'coarse_region_similarity': self.coarse_region_similarity,
            'region_adjacency': self.region_adjacency,
            'number_pois': len(self.pois),
            'number_regions': len(np.unique(self.region_id)),
            'y': self.pois['category'].values,
            'place_id': self.pois['placeid'].values,
        }


def preprocess_hgi(city, city_shapefile, poi_emb_path=None, cta_file=None):
    """Main preprocessing function for HGI."""
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

    pre = HGIPreprocess(str(checkins), str(boroughs_path), temp_folder)
    data = pre.get_data_torch(poi_emb_path=poi_emb_path)

    output_path = IoPaths.HGI.get_graph_data_file(city)
    with open(output_path, "wb") as f:
        pkl.dump(data, f)

    print(f"Saved: {output_path}")
    print(f"POIs: {data['number_pois']}, Regions: {data['number_regions']}, Edges: {len(data['edge_weight'])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess POI data for HGI')
    parser.add_argument('--city', type=str, default='Texas')
    parser.add_argument('--shapefile', type=str, default=Resources.TL_TX)
    parser.add_argument('--poi_emb', type=str, default=None)
    parser.add_argument('--cta', type=str, default=None)

    args = parser.parse_args()
    preprocess_hgi(city=args.city, city_shapefile=args.shapefile, poi_emb_path=args.poi_emb, cta_file=args.cta)
