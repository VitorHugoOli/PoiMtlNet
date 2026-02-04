"""
Input generation builders for MTL tasks.

This module contains the business logic for generating category and next-POI inputs
from embeddings. These are pure transformation functions used by pipeline orchestration.

Moved from pipelines/create_inputs.pipe.py to follow the pattern where pipes only
handle orchestration and modules contain business logic.
"""
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm

from configs.paths import IoPaths, EmbeddingEngine
from configs.model import InputsConfig
from .core import (
    generate_sequences,
    create_embedding_lookup,
    create_category_lookup,
    convert_sequences_to_poi_embeddings,
    save_next_input_dataframe,
    save_parquet,
    get_zero_embedding,
    PADDING_VALUE,
    DEFAULT_BATCH_SIZE,
    MISSING_CATEGORY_VALUE,
)


def generate_category_input(state: str, engine: EmbeddingEngine) -> None:
    """
    Generate category task input from embeddings.

    Simply copies embeddings with placeid as index.

    Args:
        state: State name (e.g., 'alabama')
        engine: Embedding engine
    """
    embeddings_df = IoPaths.load_embedd(state, engine)
    output_path = IoPaths.get_category(state, engine)

    # Simple copy with placeid as index
    output_df = embeddings_df.copy()
    save_parquet(output_df, output_path)


def generate_next_input_from_poi(
    state: str,
    engine: EmbeddingEngine,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """
    Generate next-POI input from POI-level embeddings.

    Uses embeddings that are constant per POI (e.g., HGI, Space2Vec, DGI).

    Args:
        state: State name
        engine: Embedding engine
        batch_size: Batch size for processing sequences
    """
    # Load data
    embeddings_df = IoPaths.load_embedd(state, engine)
    checkins_df = IoPaths.load_city(state)

    # Build lookups
    embedding_dim = InputsConfig.EMBEDDING_DIM
    embedding_lookup = create_embedding_lookup(embeddings_df, embedding_dim)
    category_lookup = create_category_lookup(checkins_df)

    # Generate sequences per user
    checkins_df = checkins_df.sort_values(['userid', 'datetime'])
    user_sequences = checkins_df.groupby('userid')['placeid'].apply(
        lambda places: generate_sequences(places.tolist())
    )

    # Flatten sequences into DataFrame
    sequences_data = []
    for userid, seqs in user_sequences.items():
        for seq in seqs:
            sequences_data.append(seq + [userid])

    window_size = InputsConfig.SLIDE_WINDOW
    seq_cols = [f'poi_{i}' for i in range(window_size)] + ['target_poi', 'userid']
    sequences_df = pd.DataFrame(sequences_data, columns=seq_cols)

    # Save intermediate sequences
    sequences_path = IoPaths.get_seq_next(state, engine)
    save_parquet(sequences_df, sequences_path)

    # Convert sequences to embeddings using shared logic
    all_results = convert_sequences_to_poi_embeddings(
        sequences_df, embedding_lookup, category_lookup,
        window_size, embedding_dim, batch_size
    )

    # Save output
    save_next_input_dataframe(all_results, window_size, embedding_dim, state, engine)


def generate_next_input_from_checkins(
    state: str,
    engine: EmbeddingEngine,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> None:
    """
    Generate next-POI input from check-in-level embeddings.

    Uses embeddings that vary per check-in (e.g., Time2Vec).
    Each check-in already has its contextual embedding.

    Args:
        state: State name
        engine: Embedding engine
        batch_size: Batch size for processing sequences
    """
    # Load data
    embeddings_df = IoPaths.load_embedd(state, engine)

    # Embeddings are already at check-in level with category
    # No need for separate category lookup

    # Generate sequences per user
    embeddings_df = embeddings_df.sort_values(['userid', 'datetime'])

    # Get embedding columns
    embedding_dim = InputsConfig.EMBEDDING_DIM
    emb_cols = [str(i) for i in range(embedding_dim)]

    # Build sequences with embeddings attached
    user_groups = embeddings_df.groupby('userid')
    all_sequences = []

    for userid, user_df in tqdm(user_groups, desc="Processing users"):
        user_df = user_df.reset_index(drop=True)
        places = user_df['placeid'].tolist()
        sequences = generate_sequences(places)

        if not sequences:
            continue

        window_size = InputsConfig.SLIDE_WINDOW

        for seq_idx, seq in enumerate(sequences):
            history_pois = seq[:window_size]
            target_poi = seq[window_size]

            # Calculate starting position for this sequence in user's history
            # Non-overlapping sequences: seq 0 starts at 0, seq 1 at window_size, etc.
            history_start_idx = seq_idx * window_size

            # Build embeddings using POSITION-based lookup (not POI ID search)
            seq_embeddings = []
            for i, poi in enumerate(history_pois):
                if poi == PADDING_VALUE:
                    seq_embeddings.append(get_zero_embedding(embedding_dim))
                else:
                    # Use position in user's chronological history
                    row_idx = history_start_idx + i
                    if row_idx < len(user_df):
                        emb = user_df.iloc[row_idx][emb_cols].values.astype(np.float32)
                        seq_embeddings.append(emb)
                    else:
                        seq_embeddings.append(get_zero_embedding(embedding_dim))

            # Get target category - try position first, fall back to POI ID lookup
            target_idx = history_start_idx + window_size
            if target_idx < len(user_df):
                target_category = user_df.iloc[target_idx]['category']
            else:
                # Fallback: look up target POI's category by POI ID
                target_matches = user_df[user_df['placeid'] == target_poi]
                target_category = target_matches.iloc[0]['category'] if len(target_matches) > 0 else MISSING_CATEGORY_VALUE

            # Flatten and append
            flattened = np.vstack(seq_embeddings).ravel()
            all_sequences.append(np.concatenate([
                flattened, [target_category, userid]
            ]))

    # Save output DataFrame
    save_next_input_dataframe(all_sequences, window_size, embedding_dim, state, engine)
