import os
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame


def get_parent_of_src(project_root: Path) -> Optional[Path]:
    """
    If `src` appears in `project_root` path (or any ancestor), return the parent directory of that `src`.
    Otherwise return None.
    """
    for p in (project_root, *project_root.parents):
        if p.name == "src":
            return p.parent
    return None

# Base directories
PROJECT_ROOT = get_parent_of_src(Path(__file__).parent.parent)
DATA_ROOT = Path(os.environ.get('DATA_ROOT', PROJECT_ROOT / 'data'))
OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', PROJECT_ROOT / 'output'))
RESULTS_ROOT = Path(os.environ.get('RESULTS_ROOT', PROJECT_ROOT / 'results'))

# Input data paths
IO_CHECKINS = DATA_ROOT / 'checkins'


class EmbeddingEngine(Enum):
    DGI = "dgi"
    HGI = "hgi"
    HMRM = "hmrm"
    TIME2VEC = "time2vec"
    SPACE2VEC = "space2vec"
    SPHERE2VEC = "sphere2vec"
    CHECK2HGI = "check2hgi"
    CHECK2HGI_POOLED = "check2hgi_pooled"  # POI-mean-pooled C4 counterfactual
    POI2HGI = "poi2hgi"
    FUSION = "fusion"  # Multi-embedding fusion


class Resources:
    """
    Static paths for resource files.
    """
    _miscellaneous_dir = DATA_ROOT / "miscellaneous"
    _gowalla_dir = DATA_ROOT / "gowalla"
    # US Census TIGER tract shapefiles (Gowalla states)
    TL_AL: Path = _miscellaneous_dir / "tl_2022_01_tract_AL" / "tl_2022_01_tract.shp"
    TL_AZ: Path = _miscellaneous_dir / "tl_2022_04_tract_AZ" / "tl_2022_04_tract.shp"
    TL_GA: Path = _miscellaneous_dir / "tl_2022_13_tract_GA" / "tl_2022_13_tract.shp"
    TL_FL: Path = _miscellaneous_dir / "tl_2022_12_tract_FL" / "tl_2022_12_tract.shp"
    TL_CA: Path = _miscellaneous_dir / "tl_2022_06_tract_CA" / "tl_2022_06_tract.shp"
    TL_TX: Path = _miscellaneous_dir / "tl_2022_48_tract_TX" / "tl_2022_48_tract.shp"
    # New York state tracts — covers NYC (FSQ + STEPS).
    # Download: https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_36_tract.zip
    TL_NY: Path = _miscellaneous_dir / "tl_2022_36_tract_NY" / "tl_2022_36_tract.shp"
    # Sentinel for grid-based synthetic boroughs (international cities: Tokyo, etc.)
    # When set to None in the HGI pipeline, grid_boroughs.create_grid_boroughs() is used.
    GRID: None = None

    # ── Gowalla raw inputs (consumed by src/etl/gowalla/) ──────────────────
    # See pipelines/etl/gowalla.pipe.py for a turnkey wrapper.
    CHECKINS_PARQUET: Path = _gowalla_dir / "gowalla_checkins.parquet"
    CHECKINS: Path = _gowalla_dir / "gowalla_checkins.csv"
    SPOTS: Path = _gowalla_dir / "gowalla_spots_subset1.csv"
    SPOTS_2: Path = _gowalla_dir / "gowalla_spots_subset2.csv"
    CATEGORIES_STRUCTURE: Path = _gowalla_dir / "gowalla_category_structure.json"
    CATEGORIES_CALLBACK: Path = _gowalla_dir / "callback_categories.json"
    EXTRA_CATEGORIES_CALLBACK: Path = _gowalla_dir / "extra_categories.json"
    # US states shapefile — download from Census TIGER:
    # https://www2.census.gov/geo/tiger/TIGER2022/STATE/tl_2022_us_state.zip
    STATES_US: Path = _miscellaneous_dir / "tl_2022_us_state" / "tl_2022_us_state.shp"
    # Timezone polygons — used by stage 2 to compute local_datetime per checkin.
    # Download from https://github.com/evansiroky/timezone-boundary-builder/releases
    # (combined-shapefile-with-oceans variant).
    TIMEZONES: Path = _miscellaneous_dir / "combined-shapefile-with-oceans" / "combined-shapefile-with-oceans.shp"



class _DGIIoPath:
    # File name constants
    CTA_FILE: str = "cta.csv"
    INTER_DATA_FILE: str = "data_inter.pkl"

    # DGI embeddings output
    _dgi_dir: Path = OUTPUT_DIR / EmbeddingEngine.DGI.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the DGI output directory for a specific state."""
        return cls._dgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_inter_file(cls, state: str) -> Path:
        """Get the intermediate data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.INTER_DATA_FILE

    @classmethod
    def get_cta_file(cls, state: str) -> Path:
        """Get the CTA file path for a specific state."""
        return cls.get_temp_dir(state) / cls.CTA_FILE


class _HGIIoPath:
    # File name constants
    POIS_FILE: str = "pois_gowalla.csv"
    BOROUGHS_FILE: str = "boroughs_area.csv"
    POI_EMB_FILE: str = "poi-encoder.tensor"
    GRAPH_DATA_FILE: str = "gowalla.pt"
    POI_INDEX_FILE: str = "poi_index.csv"
    POI_REGION_MAP_FILE: str = "poi_region_map.csv"
    EDGES_FILE: str = "edges.csv"
    POIS_PROCESSED_FILE: str = "pois.csv"

    # HGI embeddings output
    _hgi_dir: Path = OUTPUT_DIR / EmbeddingEngine.HGI.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the HGI output directory for a specific state."""
        return cls._hgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_pois_file(cls, state: str) -> Path:
        """Get the POIs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POIS_FILE

    @classmethod
    def get_boroughs_file(cls, state: str) -> Path:
        """Get the boroughs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.BOROUGHS_FILE

    @classmethod
    def get_poi_emb_file(cls, state: str) -> Path:
        """Get the POI embeddings file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_EMB_FILE

    @classmethod
    def get_graph_data_file(cls, state: str) -> Path:
        """Get the graph data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.GRAPH_DATA_FILE

    @classmethod
    def get_poi_index_file(cls, state: str) -> Path:
        """Get the POI index file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_INDEX_FILE

    @classmethod
    def get_poi_region_map_file(cls, state: str) -> Path:
        """Get the POI region map file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POI_REGION_MAP_FILE

    @classmethod
    def get_edges_file(cls, state: str) -> Path:
        """Get the edges file path for a specific state."""
        return cls.get_temp_dir(state) / cls.EDGES_FILE

    @classmethod
    def get_pois_processed_file(cls, state: str) -> Path:
        """Get the processed POIs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.POIS_PROCESSED_FILE


class _Time2VecIoPath:
    # File name constants
    MODEL_FILE: str = "time2vec_model.pt"

    # Time2Vec embeddings output
    _time2vec_dir: Path = OUTPUT_DIR / EmbeddingEngine.TIME2VEC.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the Time2Vec output directory for a specific state."""
        return cls._time2vec_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_model_file(cls, state: str) -> Path:
        """Get the trained model file path for a specific state."""
        return cls.get_output_dir(state) / cls.MODEL_FILE


class _Space2VecIoPath:
    # File name constants
    MODEL_FILE: str = "space2vec_model.pt"
    PAIRS_I_FILE: str = "pairs_i.int32"
    PAIRS_J_FILE: str = "pairs_j.int32"
    PAIRS_Y_FILE: str = "pairs_y.uint8"
    PAIRS_COUNT_FILE: str = "pairs_count.npy"

    # Space2Vec embeddings output
    _space2vec_dir: Path = OUTPUT_DIR / EmbeddingEngine.SPACE2VEC.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the Space2Vec output directory for a specific state."""
        return cls._space2vec_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for memmap pair files."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_model_file(cls, state: str) -> Path:
        """Get the trained model file path for a specific state."""
        return cls.get_output_dir(state) / cls.MODEL_FILE

    @classmethod
    def get_pairs_dir(cls, state: str) -> Path:
        """Get the directory for memmap pair files."""
        return cls.get_temp_dir(state)


class _Sphere2VecIoPath:
    # File name constants
    MODEL_FILE: str = "sphere2vec_model.pt"

    # Sphere2Vec embeddings output
    _sphere2vec_dir: Path = OUTPUT_DIR / EmbeddingEngine.SPHERE2VEC.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the Sphere2Vec output directory for a specific state."""
        return cls._sphere2vec_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_model_file(cls, state: str) -> Path:
        """Get the trained model file path for a specific state."""
        return cls.get_output_dir(state) / cls.MODEL_FILE


class _Check2HGIIoPath:
    # File name constants
    GRAPH_DATA_FILE: str = "checkin_graph.pt"
    BOROUGHS_FILE: str = "boroughs_area.csv"

    # Check2HGI embeddings output
    _check2hgi_dir: Path = OUTPUT_DIR / EmbeddingEngine.CHECK2HGI.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the Check2HGI output directory for a specific state."""
        return cls._check2hgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_graph_data_file(cls, state: str) -> Path:
        """Get the graph data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.GRAPH_DATA_FILE

    @classmethod
    def get_boroughs_file(cls, state: str) -> Path:
        """Get the boroughs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.BOROUGHS_FILE


class _POI2HGIIoPath:
    # File name constants
    GRAPH_DATA_FILE: str = "poi_graph.pt"
    BOROUGHS_FILE: str = "boroughs_area.csv"

    # POI2HGI embeddings output
    _poi2hgi_dir: Path = OUTPUT_DIR / EmbeddingEngine.POI2HGI.value

    @classmethod
    def get_state_dir(cls, state: str) -> Path:
        """Get the POI2HGI output directory for a specific state."""
        return cls._poi2hgi_dir / state.lower()

    @classmethod
    def get_output_dir(cls, state: str) -> Path:
        return cls.get_state_dir(state)

    @classmethod
    def get_temp_dir(cls, state: str) -> Path:
        """Get the temp directory for a specific state."""
        return cls.get_output_dir(state) / "temp"

    @classmethod
    def get_graph_data_file(cls, state: str) -> Path:
        """Get the graph data file path for a specific state."""
        return cls.get_temp_dir(state) / cls.GRAPH_DATA_FILE

    @classmethod
    def get_boroughs_file(cls, state: str) -> Path:
        """Get the boroughs file path for a specific state."""
        return cls.get_temp_dir(state) / cls.BOROUGHS_FILE


class IoPaths:
    """
    Static paths for I/O operations.
    """
    EMBEDDINGS_FILE: str = "embeddings.parquet"

    # ── Gowalla ETL artefacts (raw → labelled → localised → per-state) ─────
    _gowalla_etl_dir: Path = DATA_ROOT / "temp" / "gowalla"
    CHECKINS_ETL_STEP_1: Path = _gowalla_etl_dir / "stage1_categorised.parquet"
    CHECKINS_ETL_STEP_2: Path = _gowalla_etl_dir / "stage2_localised.parquet"
    CHECKINS_ETL_STEP_3: Path = _gowalla_etl_dir / "stage3_states.parquet"
    CHECKINS_ETL_STEP_3_CSV: Path = _gowalla_etl_dir / "stage3_states.csv"
    CHECKINS_ETL_STATES: Path = DATA_ROOT / "checkins"
    CHECKINS_ETL_STATES_PARQUET: Path = DATA_ROOT / "checkins_parquet"

    @classmethod
    def validate(cls) -> None:
        """Check that required data directories exist and create output directories.

        Call this at pipeline startup, not at import time.
        Raises FileNotFoundError if checkins directory is missing.
        """
        if not IO_CHECKINS.exists() or not IO_CHECKINS.is_symlink():
            raise FileNotFoundError(f"Checkins directory not found: {IO_CHECKINS}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    DGI = _DGIIoPath
    HGI = _HGIIoPath
    TIME2VEC = _Time2VecIoPath
    SPACE2VEC = _Space2VecIoPath
    SPHERE2VEC = _Sphere2VecIoPath
    CHECK2HGI = _Check2HGIIoPath
    POI2HGI = _POI2HGIIoPath


    @classmethod
    def get_embedd(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        """Get the embeddings file for a specific state and engine."""
        if embedd_engine == EmbeddingEngine.FUSION:
            raise ValueError(
                "FUSION is a synthetic engine (task-specific). "
                "Use get_fusion_category() or get_fusion_next() instead."
            )
        return OUTPUT_DIR / embedd_engine.value / state.lower() / cls.EMBEDDINGS_FILE

    @classmethod
    def load_embedd(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        """Load embeddings for a specific state and embedding engine."""
        return pd.read_parquet(cls.get_embedd(state, embedd_engine))

    @classmethod
    def get_city(cls, state: str, ext: str = 'parquet') -> Path:
        return DATA_ROOT / 'checkins' / f"{state.capitalize()}.{ext}"

    @classmethod
    def load_city(cls, state: str, ext: str = 'parquet') -> DataFrame:
        """Load check-in data for a specific state."""
        return pd.read_parquet(cls.get_city(state, ext))

    @classmethod
    def get_input_dir(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        return OUTPUT_DIR / embedd_engine.value / state.lower() / "input"

    @classmethod
    def get_category(cls, state: str, embedd_engine: EmbeddingEngine):
        """Get category input path (routes FUSION automatically)."""
        if embedd_engine == EmbeddingEngine.FUSION:
            return cls.get_fusion_category(state)
        return cls.get_input_dir(state, embedd_engine) / "category.parquet"

    @classmethod
    def load_category(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        """Load category input data for a specific state and engine."""
        return pd.read_parquet(cls.get_category(state, embedd_engine))

    @classmethod
    def get_next(cls, state: str, embedd_engine: EmbeddingEngine):
        """Get next-POI input path (routes FUSION automatically)."""
        if embedd_engine == EmbeddingEngine.FUSION:
            return cls.get_fusion_next(state)
        return cls.get_input_dir(state, embedd_engine) / "next.parquet"

    @classmethod
    def get_next_region(cls, state: str, embedd_engine: EmbeddingEngine) -> Path:
        """Next-region input path.

        The next-region *label space* is derived from the check2HGI
        preprocessing graph artifact (``poi_to_region`` tensor) — that
        labelling is substrate-independent. Each engine that wants to
        run a region task must publish its own ``input/next_region.parquet``
        with substrate-specific embedding columns (built by
        ``scripts/probe/build_hgi_next_region.py`` and friends).

        Currently supported: CHECK2HGI (canonical), HGI (built for the
        Phase-1 MTL counterfactual; see SUBSTRATE_COMPARISON_PLAN §5).
        Other engines: port the builder + pre-stage the parquet first.
        """
        supported = (EmbeddingEngine.CHECK2HGI, EmbeddingEngine.HGI)
        if embedd_engine not in supported:
            raise ValueError(
                f"next_region not yet built for {embedd_engine}. Supported: "
                f"{[e.name for e in supported]}. Build with "
                f"scripts/probe/build_hgi_next_region.py (or analogous)."
            )
        return cls.get_input_dir(state, embedd_engine) / "next_region.parquet"

    @classmethod
    def load_next_region(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        """Load next-region input data for a state (CHECK2HGI only)."""
        return pd.read_parquet(cls.get_next_region(state, embedd_engine))

    @classmethod
    def load_next(cls, state: str, embedd_engine: EmbeddingEngine) -> DataFrame:
        """Load next-POI input data for a specific state and engine."""
        return pd.read_parquet(cls.get_next(state, embedd_engine))

    @classmethod
    def get_seq_next(cls, state: str, embedd_engine: EmbeddingEngine):
        if embedd_engine == EmbeddingEngine.FUSION:
            return cls.get_fusion_seq_next(state)
        return OUTPUT_DIR / embedd_engine.value / state.lower() / "temp" / "sequences_next.parquet"

    @classmethod
    def get_results_dir(cls, state, embedd_engine: EmbeddingEngine) -> Path:
        """Get results directory (routes FUSION automatically)."""
        if embedd_engine == EmbeddingEngine.FUSION:
            return cls.get_fusion_results_dir(state)
        return RESULTS_ROOT / embedd_engine.value / state.lower()

    @classmethod
    def get_folds_dir(cls, state, embedd_engine: EmbeddingEngine) -> Path:
        """Get folds directory (routes FUSION automatically)."""
        if embedd_engine == EmbeddingEngine.FUSION:
            return cls.get_fusion_folds_dir(state)
        return OUTPUT_DIR / embedd_engine.value / state.lower() / "folds"

    # ========================================================================
    # Multi-Embedding Fusion Paths
    # ========================================================================

    @staticmethod
    def get_fusion_input_dir(state: str) -> Path:
        """Get directory for fusion-based inputs."""
        return OUTPUT_DIR / "fusion" / state.lower() / "input"

    @staticmethod
    def get_fusion_category(state: str) -> Path:
        """Get path to fused category input."""
        return IoPaths.get_fusion_input_dir(state) / "category.parquet"

    @staticmethod
    def load_fusion_category(state: str) -> DataFrame:
        """Load fused category input."""
        return pd.read_parquet(IoPaths.get_fusion_category(state))

    @staticmethod
    def get_fusion_next(state: str) -> Path:
        """Get path to fused next-POI input."""
        return IoPaths.get_fusion_input_dir(state) / "next.parquet"

    @staticmethod
    def load_fusion_next(state: str) -> DataFrame:
        """Load fused next-POI input."""
        return pd.read_parquet(IoPaths.get_fusion_next(state))

    @staticmethod
    def get_fusion_seq_next(state: str) -> Path:
        """Get path to intermediate sequences for fusion."""
        return IoPaths.get_fusion_input_dir(state) / "temp" / "sequences_next.parquet"

    @staticmethod
    def get_fusion_results_dir(state: str) -> Path:
        """Get results directory for fusion experiments."""
        return RESULTS_ROOT / "fusion" / state.lower()

    @staticmethod
    def get_fusion_folds_dir(state: str) -> Path:
        """Get folds directory for fusion experiments."""
        return OUTPUT_DIR / "fusion" / state.lower() / "folds"



