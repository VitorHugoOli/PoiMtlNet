"""
Pure logic functions for MTL input generation (no I/O).

This module contains stateless, testable functions extracted from create_input.py.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from configs.model import InputsConfig


# Constants
PADDING_VALUE = -1
MIN_SEQUENCE_LENGTH = 5
DEFAULT_BATCH_SIZE = 100000
MISSING_CATEGORY_VALUE = 'None'


def generate_sequences(
    places_visited: List[int],
    window_size: int = InputsConfig.SLIDE_WINDOW,
    pad_value: int = PADDING_VALUE,
) -> List[List[int]]:
    """
    Generate non-overlapping sequences of fixed length for next-POI prediction.

    Each sequence contains:
    - First (window_size) positions: historical visits (padded if needed)
    - Final position: target POI to predict

    Args:
        places_visited: List of place IDs in chronological order
        window_size: Number of historical visits per sequence (default: SLIDE_WINDOW)
        pad_value: Value for padding short sequences (default: -1)

    Returns:
        List of sequences, each of length (window_size + 1).
        Empty list if insufficient data.
    """
    if not places_visited or len(places_visited) < MIN_SEQUENCE_LENGTH:
        return []

    sequences: List[List[int]] = []
    step = window_size
    total_visits = len(places_visited)

    for start_idx in range(0, total_visits, step):
        # Extract history window
        history = places_visited[start_idx:start_idx + step]

        # Pad if history is shorter than window_size
        if len(history) < step:
            history = history + [pad_value] * (step - len(history))

        # Determine target POI
        target_idx = start_idx + step
        if target_idx < total_visits:
            target_poi = places_visited[target_idx]
        else:
            # Use last real visit as target, shift history
            for j in range(len(history) - 1, -1, -1):
                if history[j] != pad_value:
                    target_poi = history[j]
                    history = history[:j] + history[j + 1:] + [pad_value]
                    break
            else:
                target_poi = pad_value

        # Skip all-padding sequences
        if all(x == pad_value for x in history) or target_poi == pad_value:
            continue

        sequences.append(history + [target_poi])

    return sequences


def create_embedding_lookup(
    embeddings_df: pd.DataFrame,
    embedding_dim: int
) -> Dict[int, np.ndarray]:
    """
    Build POI → embedding lookup dictionary.

    Consolidates patterns from:
    - create_input.py:360-372
    - embedding_fusion.py:413-417

    Args:
        embeddings_df: DataFrame with 'placeid' and embedding columns
        embedding_dim: Dimension of embeddings

    Returns:
        Dictionary mapping POI ID to embedding vector (np.ndarray)
    """
    emb_cols = [str(i) for i in range(embedding_dim)]

    # Set placeid as index if not already
    if 'placeid' in embeddings_df.columns:
        emb_df = embeddings_df.set_index('placeid')[emb_cols]
    else:
        emb_df = embeddings_df[emb_cols]

    # Build lookup dictionary
    lookup = {
        poi_id: row.values.astype(np.float32)
        for poi_id, row in emb_df.iterrows()
    }

    # Add zero embedding for padding
    lookup[PADDING_VALUE] = np.zeros(embedding_dim, dtype=np.float32)

    return lookup


def create_category_lookup(
    checkins_df: pd.DataFrame,
    default_value: str = None
) -> Dict[int, str]:
    """
    Build POI → category lookup dictionary.

    Extracted from create_input.py:350-357
    Updated to support default value for missing categories.

    Args:
        checkins_df: DataFrame with 'placeid' and 'category' columns
        default_value: Value to use for missing categories (default: MISSING_CATEGORY_VALUE)

    Returns:
        Dictionary mapping POI ID to category label
    """
    if default_value is None:
        default_value = MISSING_CATEGORY_VALUE

    # Handle duplicate placeids by taking first occurrence
    unique_checkins = checkins_df.drop_duplicates(subset=['placeid'], keep='first')
    lookup = dict(zip(unique_checkins['placeid'], unique_checkins['category']))

    # Add explicit padding entry
    lookup[PADDING_VALUE] = default_value

    return lookup


def get_zero_embedding(embedding_dim: int) -> np.ndarray:
    """
    Get zero embedding vector for padding.

    Centralizes pattern from lines 226, 373, 410, 449 across multiple files.

    Args:
        embedding_dim: Dimension of embedding

    Returns:
        Zero vector of specified dimension
    """
    return np.zeros(embedding_dim, dtype=np.float32)


def parse_and_sort_checkins(checkin_timestamps: List[str]) -> List:
    """
    Parse and sort checkin timestamps using vectorized operations.

    Extracted from create_input.py:19-29

    Args:
        checkin_timestamps: List of timestamp strings

    Returns:
        List of sorted datetime objects
    """
    return sorted(pd.to_datetime(checkin_timestamps))


def save_parquet(
    df: pd.DataFrame,
    output_path,
    create_dirs: bool = True,
    index: bool = False
) -> None:
    """
    Save DataFrame to parquet with automatic directory creation.

    Consolidates pattern from 5 locations across builders.py and fusion.py.

    Args:
        df: DataFrame to save
        output_path: Output file path (str or Path)
        create_dirs: If True, create parent directories
        index: If True, include index in output
    """
    from pathlib import Path

    path = Path(output_path)

    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(path, index=index)


def save_next_input_dataframe(
    results: List[np.ndarray],
    window_size: int,
    embedding_dim: int,
    state: str,
    engine
) -> None:
    """
    Save next-POI input results to parquet.

    Consolidates DataFrame creation and saving from:
    - builders.py:113-121 (generate_next_input_from_poi)
    - builders.py:196-209 (generate_next_input_from_checkins)
    - fusion.py:430-438 (_convert_sequences_to_fused_embeddings)

    Args:
        results: List of numpy arrays, each containing flattened embeddings
                + target category + userid
        window_size: Number of historical steps in sequence
        embedding_dim: Embedding dimension
        state: State name
        engine: Embedding engine (EmbeddingEngine enum)
    """
    from configs.paths import IoPaths

    # Create column names
    num_features = window_size * embedding_dim
    columns = list(map(str, range(num_features))) + ['next_category', 'userid']

    # Create DataFrame
    output_df = pd.DataFrame(results, columns=columns)

    # Save to output path
    output_path = IoPaths.get_next(state, engine)
    save_parquet(output_df, output_path)


def convert_sequences_to_poi_embeddings(
    sequences_df: pd.DataFrame,
    embedding_lookup: Dict[int, np.ndarray],
    category_lookup: Dict[int, str],
    window_size: int,
    embedding_dim: int,
    batch_size: int = None,
    show_progress: bool = True
) -> List[np.ndarray]:
    """
    Convert POI sequences to flattened embedding sequences.

    Consolidates ~33 lines of duplicated logic from:
    - builders.py:89-111 (generate_next_input_from_poi)
    - fusion.py:403-428 (_convert_sequences_to_fused_embeddings)

    Process:
    1. For each sequence row: [poi_0, poi_1, ..., poi_8, target_poi, userid]
    2. Map each POI ID to its embedding vector via lookup
    3. Stack window embeddings: shape (window_size, embedding_dim)
    4. Flatten to 1D: shape (window_size * embedding_dim,)
    5. Append target category and userid

    **IMPORTANT**: This function is ONLY for POI-level embeddings (dictionary-based lookup).
    It is NOT used by generate_next_input_from_checkins which has a different algorithm
    for check-in-level embeddings.

    Args:
        sequences_df: DataFrame with columns [poi_0, ..., poi_N, target_poi, userid]
        embedding_lookup: POI ID → embedding vector (dictionary)
        category_lookup: POI ID → category label
        window_size: Number of POIs in history window
        embedding_dim: Dimension of each embedding
        batch_size: Batch size for progress bar (default: DEFAULT_BATCH_SIZE)
        show_progress: If True, show tqdm progress bar

    Returns:
        List of numpy arrays, each containing:
        [flattened_embeddings..., target_category, userid]
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    from tqdm import tqdm

    all_results = []
    iterator = range(0, len(sequences_df), batch_size)

    if show_progress:
        iterator = tqdm(iterator, desc="Processing batches")

    for start_idx in iterator:
        batch = sequences_df.iloc[start_idx:start_idx + batch_size]

        for _, row in batch.iterrows():
            # Extract history POI IDs (first window_size columns)
            history_pois = row.iloc[:window_size].values.astype(int)
            target_poi = int(row.iloc[window_size])
            userid = row['userid']

            # Build sequence embeddings using lookup
            sequence_embeddings = np.vstack([
                embedding_lookup.get(int(poi_id), embedding_lookup[PADDING_VALUE])
                for poi_id in history_pois
            ])

            # Flatten to 1D: [window_size × embedding_dim] features
            flattened = sequence_embeddings.ravel()

            # Get target category
            target_category = category_lookup.get(target_poi, MISSING_CATEGORY_VALUE)

            # Append target category and userid
            all_results.append(np.concatenate([
                flattened,
                [target_category, userid]
            ]))

    return all_results


def convert_user_checkins_to_sequences(
    user_df: pd.DataFrame,
    embedding_cols: List[str],
    window_size: int,
    embedding_dim: int,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Convert a single user's check-in DataFrame to embedding sequences using position-based lookup.

    This function handles check-in-level embeddings where each row in user_df corresponds
    to a specific check-in event. Unlike POI-level embeddings where same POI = same embedding,
    check-in-level embeddings (Time2Vec, Check2HGI) have unique embeddings per visit.

    The position-based lookup ensures that sequence N starts at position N * window_size
    in the user's chronological history, preserving the temporal context of each visit.

    Args:
        user_df: DataFrame for a single user, sorted by datetime, with reset index.
                 Must contain columns: userid, placeid, category, and all embedding_cols.
        embedding_cols: List of column names containing embedding values.
                       Example: ['0', '1', ..., '63'] or ['fused_0', 'fused_1', ..., 'fused_127']
        window_size: Number of historical check-ins per sequence.
        embedding_dim: Dimension of each embedding vector.

    Returns:
        Tuple of:
        - embedding_results: List of numpy arrays, each containing:
          [flattened_window_embeddings, target_category, userid]
        - poi_sequences: List of POI ID sequences (for intermediate file saving)
          Each sequence is [poi_0, ..., poi_{window-1}, target_poi, userid]

    Notes:
        - Uses non-overlapping windows: sequence N starts at position N * window_size
        - Padding positions (PADDING_VALUE) get zero embeddings
        - Target category uses position-based lookup with fallback to POI ID search
    """
    embedding_results = []
    poi_sequences = []

    # Get user ID (same for all rows in user_df)
    userid = user_df['userid'].iloc[0]

    # Generate POI sequences using shared function
    places = user_df['placeid'].tolist()
    sequences = generate_sequences(places, window_size=window_size)

    if not sequences:
        return [], []

    for seq_idx, seq in enumerate(sequences):
        history_pois = seq[:window_size]
        target_poi = seq[window_size]

        # Save POI sequence for intermediate output
        poi_sequences.append(seq + [userid])

        # Calculate starting position in user's chronological history
        # Non-overlapping sequences: seq 0 starts at 0, seq 1 at window_size, etc.
        history_start_idx = seq_idx * window_size

        # Build embeddings using POSITION-based lookup (not POI ID search)
        seq_embeddings = []
        for i, poi in enumerate(history_pois):
            if poi == PADDING_VALUE:
                seq_embeddings.append(get_zero_embedding(embedding_dim))
            else:
                row_idx = history_start_idx + i
                if row_idx < len(user_df):
                    emb = user_df.iloc[row_idx][embedding_cols].values.astype(np.float32)
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
            target_category = (
                target_matches.iloc[0]['category']
                if len(target_matches) > 0
                else MISSING_CATEGORY_VALUE
            )

        # Flatten embeddings and append metadata
        flattened = np.vstack(seq_embeddings).ravel()
        embedding_results.append(np.concatenate([
            flattened, [target_category, userid]
        ]))

    return embedding_results, poi_sequences
