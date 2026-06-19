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

# Engines that produce check-in-level embeddings (one per visit, not per POI).
# Category task requires POI-level embeddings (one per POI).
_CHECKIN_LEVEL_ENGINES = {
    EmbeddingEngine.CHECK2HGI,
    EmbeddingEngine.CHECK2HGI_POI2VEC,
    EmbeddingEngine.C2HGI_HGI_CONCAT,  # Design A: 128-dim late concat (per-step, NOT POI-level)
    EmbeddingEngine.CHECK2HGI_DESIGN_E,
    EmbeddingEngine.CHECK2HGI_DESIGN_B,
    EmbeddingEngine.CHECK2HGI_DESIGN_H,
    EmbeddingEngine.CHECK2HGI_DESIGN_D,
    EmbeddingEngine.CHECK2HGI_DESIGN_I,
    EmbeddingEngine.CHECK2HGI_DESIGN_J,
    EmbeddingEngine.CHECK2HGI_DESIGN_M,
    EmbeddingEngine.CHECK2HGI_DESIGN_L,
    EmbeddingEngine.CHECK2HGI_LEVER4_CANONICAL,
    EmbeddingEngine.CHECK2HGI_LEVER4_DESIGN_B,
    EmbeddingEngine.CHECK2HGI_RESLN,
    EmbeddingEngine.CHECK2HGI_RESLN_DESIGN_B,
    EmbeddingEngine.CHECK2HGI_RESLN_DESIGN_J,
    EmbeddingEngine.CHECK2HGI_CTLE,  # [ENUM-MERGE] B1 CTLE contextual per-visit substrate
    EmbeddingEngine.TIME2VEC,
}

from .core import (
    generate_sequences,
    create_embedding_lookup,
    create_category_lookup,
    convert_sequences_to_poi_embeddings,
    convert_user_checkins_to_sequences,
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
    Rejects check-in-level engines (Time2Vec, Check2HGI) — category task
    requires one embedding per POI.

    Args:
        state: State name (e.g., 'alabama')
        engine: Embedding engine

    Raises:
        ValueError: If engine produces check-in-level embeddings.
    """
    if engine in _CHECKIN_LEVEL_ENGINES:
        raise ValueError(
            f"Engine {engine.value} produces check-in-level embeddings, which are "
            f"invalid for the category task (requires one embedding per POI). "
            f"Use a POI-level engine (HGI, DGI, Space2Vec, POI2HGI, HMRM) or "
            f"fusion mode with POI-level category embeddings."
        )

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

    # Build lookups — infer dimension from the embedding artifact
    numeric_cols = [c for c in embeddings_df.columns if c.isdigit()]
    embedding_dim = len(numeric_cols)
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
    batch_size: int = DEFAULT_BATCH_SIZE,
    stride: int = None,
) -> None:
    """
    Generate next-POI input from check-in-level embeddings.

    Uses embeddings that vary per check-in (e.g., Time2Vec).
    Each check-in already has its contextual embedding.

    Args:
        state: State name
        engine: Embedding engine
        batch_size: Batch size for processing sequences (unused, kept for API compatibility)
    """
    # Load data
    embeddings_df = IoPaths.load_embedd(state, engine)

    # Embeddings are already at check-in level with category
    # No need for separate category lookup

    # Sort by user and time to ensure chronological order
    embeddings_df = embeddings_df.sort_values(['userid', 'datetime'])

    # Detect embedding dimension from data
    numeric_cols = [c for c in embeddings_df.columns if c.isdigit()]
    embedding_dim = len(numeric_cols)
    emb_cols = [str(i) for i in range(embedding_dim)]
    window_size = InputsConfig.SLIDE_WINDOW

    # Process each user. ``convert_user_checkins_to_sequences`` returns rows as
    # ``<U32`` arrays (the float embeddings are concatenated with the string
    # target_category + userid, which upcasts the whole 578-elem row to 128
    # bytes/elem = 32x bloat). At stride=1 the FL row count is ~8.5x the stride=9
    # case (~1.27M rows), so accumulating those <U32 rows + the np.array() copy
    # in save_next_input_dataframe peaks at ~168 GB and OOM-kills the box. Fix
    # (memory-safe, byte-identical output): convert each user's small batch of
    # rows to float32 immediately and split off the cat/userid metadata, so the
    # global accumulation is a compact float32 list instead of <U32. Peak ~5 GB.
    num_features = window_size * embedding_dim
    out_cols = [str(i) for i in range(num_features)]
    emb_rows: list = []          # float32 (num_features,) per window
    cat_meta: list = []          # target_category (str) — same value as before
    uid_meta: list = []          # userid (str, as the <U32 path produced)
    all_sequences = []

    for userid, user_df in tqdm(embeddings_df.groupby('userid'), desc="Processing users"):
        user_df = user_df.reset_index(drop=True)

        results, sequences = convert_user_checkins_to_sequences(
            user_df, emb_cols, window_size, embedding_dim, stride=stride
        )

        all_sequences.extend(sequences)
        for row in results:
            # row is a <U32 array; row[:num_features] are the float values as
            # strings — parsing back to float32 is lossless (numpy's str repr of
            # a float32 round-trips exactly), so the parquet is byte-identical to
            # the legacy save_next_input_dataframe path.
            emb_rows.append(np.asarray(row[:num_features], dtype=np.float32))
            cat_meta.append(row[num_features])
            uid_meta.append(row[num_features + 1])
        del results  # free this user's <U32 batch before the next group

    # Save intermediate sequences (for debugging/analysis)
    seq_cols = [f'poi_{i}' for i in range(window_size)] + ['target_poi', 'userid']
    sequences_df = pd.DataFrame(all_sequences, columns=seq_cols)
    # POI columns may mix str placeids with int padding (-1); unify to str for parquet
    poi_cols = [f'poi_{i}' for i in range(window_size)] + ['target_poi']
    for col in poi_cols:
        sequences_df[col] = sequences_df[col].astype(str)
    sequences_path = IoPaths.get_seq_next(state, engine)
    save_parquet(sequences_df, sequences_path)

    # Save output DataFrame — preallocate ONE float32 matrix (no <U32, no extra
    # np.array copy). Columns/dtypes/order match save_next_input_dataframe exactly.
    if emb_rows:
        mat = np.empty((len(emb_rows), num_features), dtype=np.float32)
        for i, e in enumerate(emb_rows):
            mat[i] = e
        output_df = pd.DataFrame(mat, columns=out_cols)
        output_df["next_category"] = cat_meta
        output_df["userid"] = uid_meta
    else:
        output_df = pd.DataFrame(columns=out_cols + ["next_category", "userid"])
    save_parquet(output_df, IoPaths.get_next(state, engine))
