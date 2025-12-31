from concurrent.futures import ProcessPoolExecutor

from configs.model import InputsConfig
from configs.paths import IoPaths, EmbeddingEngine

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Union, Optional
from tqdm import tqdm
import gc
from functools import lru_cache

from etl.create_input import generate_category_input, generate_next_input_from_poi


def create_input(state: str, embedding_engine: EmbeddingEngine):
    """
    Create input files for a given state and embedding engine.

    Generates:
    - Category input file (embeddings with categories)
    - Next-POI input file (sequences with embeddings)
    """
    print(f"Processing state: {state} with embedding engine: {embedding_engine.value}")

    # Load data
    hgi = IoPaths.load_embedd(state, EmbeddingEngine.HGI)
    spave = IoPaths.load_embedd(state, EmbeddingEngine.SPACE2VEC)
    time = IoPaths.load_embedd(state, EmbeddingEngine.TIME2VEC)
    checkins = IoPaths.load_city(state)

    # Get output paths
    sequences_path = IoPaths.get_seq_next(state, embedding_engine)
    next_input_path = IoPaths.get_next(state, embedding_engine)
    category_input_path = IoPaths.get_category(state, embedding_engine)

    # Create directories
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    next_input_path.parent.mkdir(parents=True, exist_ok=True)
    category_input_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge embeddings
    chechings_embedd, pois_embedd = combine_embeddings(hgi, spave, time)

    # Generate inputs
    generate_category_input(embeddings_df, str(category_input_path))
    generate_next_input_from_poi(embeddings_df, checkins_df, str(sequences_path), str(next_input_path))


if __name__ == '__main__':
    STATE_NAME = [
        ("florida", EmbeddingEngine.CHECK2HGI),
        # ("texas", EmbeddingEngine.DGI),
        # ("alabama", EmbeddingEngine.DGI),
        # ("arizona", EmbeddingEngine.DGI),
        # ("georgia", EmbeddingEngine.DGI),
    ]
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(create_input, state, engine) for state, engine in STATE_NAME]
        results = [future.result() for future in futures]
