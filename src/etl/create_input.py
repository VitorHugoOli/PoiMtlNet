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

# Constants
SEQUENCE_LENGTH = 10
PADDING_VALUE = -1
MIN_SEQUENCE_LENGTH = 5


def parse_and_sort_checkins(checkin_timestamps: List[str]) -> List[datetime]:
    """
    Parse and sort checkin timestamps using vectorized operations.

    Args:
        checkin_timestamps: List of timestamp strings

    Returns:
        List of sorted datetime objects
    """
    return sorted(pd.to_datetime(checkin_timestamps))


def previus_generate_sequences(places_visited: List[int], max_sequences_per_user: Optional[int] = None) -> Union[
    List[List[int]], int]:
    """
    [DEPRECATED] Generate fixed-length sequences from a list of places visited.
    Use generate_sequences() instead.
    """
    if len(places_visited) < MIN_SEQUENCE_LENGTH:
        return 0

    sequences = []
    for i in range(0, len(places_visited), InputsConfig.EMBEDDING_DIM):
        if len(places_visited) >= i + SEQUENCE_LENGTH:
            seq = places_visited[i:i + SEQUENCE_LENGTH]
        else:
            remaining = places_visited[i:]
            seq = [PADDING_VALUE] * (SEQUENCE_LENGTH - len(remaining)) + remaining

        sequences.append(seq)

        if max_sequences_per_user and len(sequences) >= max_sequences_per_user:
            break

    return sequences


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


def flatten_user_sequences(
        user_ids: List[int],
        user_sequences_df: pd.DataFrame,
        output_path: str
) -> pd.DataFrame:
    """
    Flatten user sequences into a DataFrame where each row is one sequence.

    Converts from:
        user_id -> [[seq1], [seq2], ...]
    To:
        userid | pos_1 | pos_2 | ... | pos_N | target
        1      | 100   | 200   | ... | 800   | 900

    Args:
        user_ids: List of user IDs to process
        user_sequences_df: DataFrame indexed by userid with 'visit_sequence' column
            containing lists of sequences
        output_path: Path to save the resulting parquet file

    Returns:
        DataFrame with columns: ['userid', 1, 2, ..., SLIDE_WINDOW+1]
    """
    rows = []

    for user_id in tqdm(user_ids, desc="Flattening user sequences"):
        try:
            if user_id not in user_sequences_df.index:
                continue

            # Get all sequences for this user
            visit_sequences = user_sequences_df.loc[user_id].iloc[0]
            if not isinstance(visit_sequences, list):
                continue

            # Add each sequence as a row
            for sequence in visit_sequences:
                rows.append([user_id] + sequence)

        except (KeyError, IndexError) as e:
            print(f"Warning: Could not process user {user_id}: {e}")
            continue

    # Create DataFrame
    sequence_length = InputsConfig.SLIDE_WINDOW + 1  # history + target
    column_names = ['userid'] + list(range(1, sequence_length + 1))
    sequences_df = pd.DataFrame(rows, columns=column_names)

    print(f'Sequences DataFrame shape: {sequences_df.shape}')

    # Save to parquet
    try:
        sequences_df.to_parquet(output_path, index=False)
        print(f'Saved sequences to {output_path}\n')
    except IOError as e:
        print(f"Error saving to {output_path}: {e}")

    return sequences_df


def convert_sequences_to_embeddings(
        sequences_df: pd.DataFrame,
        output_path: str,
        category_lookup: Dict[int, str],
        embedding_lookup: Dict[int, np.ndarray],
        batch_size: int = 100_000
) -> pd.DataFrame:
    """
    Convert POI ID sequences to embedding sequences.

    For each sequence row:
    - Replace each POI ID with its embedding vector
    - Flatten the embedding matrix into a single row
    - Add the target category

    Args:
        sequences_df: DataFrame with columns [userid, 1, 2, ..., N, target]
            where values are POI IDs
        output_path: Path to save the resulting parquet file
        category_lookup: Dict mapping POI ID to category string
        embedding_lookup: Dict mapping POI ID to embedding numpy array
        batch_size: Number of sequences to process per batch

    Returns:
        DataFrame with columns: ['0', '1', ..., 'N*EMB_DIM-1', 'next_category', 'userid']
    """
    embedding_dim = InputsConfig.EMBEDDING_DIM
    window_size = InputsConfig.SLIDE_WINDOW

    # Output column names: embedding indices + metadata
    num_embedding_cols = embedding_dim * window_size
    column_names = list(map(str, range(num_embedding_cols))) + ['next_category', 'userid']

    # Check for existing file (resume support)
    existing_batches = []
    start_idx = 0

    if os.path.exists(output_path):
        try:
            existing_df = pd.read_parquet(output_path)
            start_idx = len(existing_df)
            existing_batches.append(existing_df)
            print(f"Resuming from row {start_idx}")
        except Exception as e:
            print(f"Warning: Could not read existing file ({e}), starting fresh.")
            start_idx = 0
    else:
        print("No existing file found; starting fresh.")

    # Zero embedding for padding
    zero_embedding = np.zeros(embedding_dim, dtype=np.float32)

    @lru_cache(maxsize=None)
    def get_embedding(poi_id: int) -> np.ndarray:
        """Get embedding for a POI, returning zeros for padding or unknown POIs."""
        if poi_id == PADDING_VALUE:
            return zero_embedding
        return embedding_lookup.get(poi_id, zero_embedding)

    # Process in batches
    processed_batches = []
    total_sequences = len(sequences_df)
    progress_bar = tqdm(
        range(start_idx, total_sequences, batch_size),
        desc="Converting to embeddings"
    )

    for batch_start in progress_bar:
        batch_end = min(batch_start + batch_size, total_sequences)
        batch = sequences_df.iloc[batch_start:batch_end]

        batch_rows = []
        for _, row in batch.iterrows():
            # Extract history POIs (first window_size columns after userid)
            history_pois = row.iloc[:window_size].values
            target_poi = row.iloc[window_size]

            # Look up target category
            target_category = category_lookup.get(target_poi, 'None')

            # Build embedding matrix and flatten
            sequence_embeddings = np.vstack([
                get_embedding(int(poi_id)) for poi_id in history_pois
            ])
            flattened_embeddings = sequence_embeddings.ravel()

            # Assemble output row
            batch_rows.append(
                list(flattened_embeddings) + [target_category, str(row.name)]
            )

        # Create batch DataFrame
        batch_df = pd.DataFrame(batch_rows, columns=column_names)
        processed_batches.append(batch_df)
        progress_bar.set_postfix(rows_processed=batch_end)

    # Concatenate all batches
    print("Concatenating batches...")
    all_batches = existing_batches + processed_batches
    result_df = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame(columns=column_names)

    # Write final result
    print(f"Writing {len(result_df)} sequences to {output_path}...")
    result_df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Success: Wrote {len(result_df)} sequences")

    gc.collect()
    return result_df


def generate_next_input_from_poi(
        embeddings_df: pd.DataFrame,
        checkins_df: pd.DataFrame,
        sequences_path: str,
        next_input_path: str
) -> pd.DataFrame:
    """
    Generate input data for next-POI prediction model.

    Pipeline:
    1. Create visit sequences from user check-ins (sliding window)
    2. Build lookup dictionaries for categories and embeddings
    3. Flatten sequences into a DataFrame
    4. Replace POI IDs with embedding vectors

    Args:
        embeddings_df: DataFrame with POI embeddings.
            Columns: 'placeid' (or index), 'category', '0'...'N' (embedding dims)
        checkins_df: DataFrame with user check-ins.
            Columns: 'userid', 'placeid', 'local_datetime', 'category'
        sequences_path: Path for intermediate sequences parquet file
        next_input_path: Path for final output parquet file

    Returns:
        DataFrame with flattened embeddings and target categories
    """
    # === Step 1: Parse datetime ===
    try:
        checkins_df['local_datetime'] = pd.to_datetime(checkins_df['local_datetime'])
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return pd.DataFrame()

    # === Step 2: Generate visit sequences per user ===
    print("Generating sequences for each user...")

    # Sort by user and time
    checkins_sorted = checkins_df.sort_values(by=['userid', 'local_datetime'])

    # Filter users with enough check-ins
    min_checkins = InputsConfig.SLIDE_WINDOW + 1
    user_counts = checkins_sorted.groupby('userid', sort=False).size()
    valid_users = user_counts[user_counts >= min_checkins].index

    # Generate sequences for valid users
    user_sequences = (
        checkins_sorted[checkins_sorted['userid'].isin(valid_users)]
        .groupby('userid', sort=False)['placeid']
        .apply(lambda places: generate_sequences(places.tolist()))
        .reset_index(name='visit_sequence')
    )

    # Remove users with empty sequences
    user_sequences = user_sequences[user_sequences.visit_sequence.map(bool)]

    sequences_by_user = user_sequences[['userid', 'visit_sequence']].set_index('userid')
    unique_user_ids = sequences_by_user.index.unique()

    total_sequences = sum(len(seq) for seq in user_sequences['visit_sequence'])
    print(f'Valid users: {len(user_sequences)}, Total sequences: {total_sequences}')

    # === Step 3: Build lookup dictionaries ===
    print("Building lookup dictionaries...")

    # Category lookup from checkins (source of truth)
    category_lookup = (
        checkins_df[['placeid', 'category']]
        .drop_duplicates('placeid')
        .set_index('placeid')['category']
        .to_dict()
    )
    category_lookup[PADDING_VALUE] = 'None'

    # Embedding lookup
    embedding_cols = [str(i) for i in range(InputsConfig.EMBEDDING_DIM)]

    if 'placeid' in embeddings_df.columns:
        emb_df = embeddings_df.set_index('placeid')[embedding_cols]
    else:
        emb_df = embeddings_df[embedding_cols]

    emb_df.index = emb_df.index.astype(int)

    embedding_lookup = {
        poi_id: row.values.astype(np.float32)
        for poi_id, row in emb_df.iterrows()
    }
    embedding_lookup[PADDING_VALUE] = np.zeros(InputsConfig.EMBEDDING_DIM, dtype=np.float32)

    # === Step 4: Flatten sequences to DataFrame ===
    print("Flattening sequences to DataFrame...")
    sequences_df = flatten_user_sequences(unique_user_ids, sequences_by_user, sequences_path)
    sequences_df.set_index('userid', inplace=True)

    # === Step 5: Convert to embeddings ===
    print("Converting sequences to embeddings...")
    result = convert_sequences_to_embeddings(
        sequences_df,
        next_input_path,
        category_lookup,
        embedding_lookup
    )

    print("Processing complete!")
    return result


def generate_category_input(embeddings_df: pd.DataFrame, category_path: str):
    """Save embeddings with category for category prediction task."""
    embeddings_df.set_index('placeid', inplace=True)
    embeddings_df.to_parquet(category_path)


def generate_next_input_from_checkings(
        embeddings_df: pd.DataFrame,
        sequences_path: str,
        next_input_path: str
) -> pd.DataFrame:
    """
    Generate input data for next-POI prediction using check-in level embeddings.

    Uses ONLY the embeddings file as the single source of truth.
    This is the most robust approach since:
    - No alignment/merge issues between different data sources
    - Every row is guaranteed to have an embedding
    - Self-contained and future-proof

    Unlike generate_next_input_from_poi() which uses POI-level embeddings (same
    embedding for same POI), this method uses check-in level embeddings where
    each visit has its own unique embedding based on temporal/contextual features.

    Args:
        embeddings_df: DataFrame with check-in embeddings from Check2HGI.
            Columns: 'userid', 'placeid', 'datetime', 'category', '0'...'63'
        sequences_path: Path for intermediate sequences parquet file
        next_input_path: Path for final output parquet file

    Returns:
        DataFrame with columns: ['0', '1', ..., 'N*EMB_DIM-1', 'next_category', 'userid']
        Same format as generate_next_input_from_poi() output.
    """
    print("Using embeddings file as single source of truth...")

    # === Step 1: Validate and prepare data ===
    embeddings_df = embeddings_df.copy()
    embeddings_df['datetime'] = pd.to_datetime(embeddings_df['datetime'])

    # Validate required columns
    embedding_cols = [str(i) for i in range(InputsConfig.EMBEDDING_DIM)]
    required_cols = ['userid', 'datetime', 'category'] + embedding_cols
    missing = set(required_cols) - set(embeddings_df.columns)
    if missing:
        print(f"Error: Missing columns in embeddings file: {missing}")
        return pd.DataFrame()

    print(f"Total check-ins with embeddings: {len(embeddings_df)}")

    # === Step 2: Sort and add row indices ===
    embeddings_df = embeddings_df.sort_values(['userid', 'datetime']).reset_index(drop=True)
    embeddings_df['row_idx'] = embeddings_df.index

    # Pre-extract embedding matrix for vectorized lookup
    embedding_matrix = embeddings_df[embedding_cols].values.astype(np.float32)
    zero_embedding = np.zeros(InputsConfig.EMBEDDING_DIM, dtype=np.float32)

    # === Step 3: Generate sequences ===
    print("Generating sequences for each user...")
    min_checkins = InputsConfig.SLIDE_WINDOW + 1
    user_counts = embeddings_df.groupby('userid', sort=False).size()
    valid_users = user_counts[user_counts >= min_checkins].index

    user_sequences = (
        embeddings_df[embeddings_df['userid'].isin(valid_users)]
        .groupby('userid', sort=False)['row_idx']
        .apply(lambda indices: generate_sequences(indices.tolist()))
        .reset_index(name='visit_sequence')
    )
    user_sequences = user_sequences[user_sequences.visit_sequence.map(bool)]

    if user_sequences.empty:
        print("No valid sequences generated.")
        return pd.DataFrame()

    total_sequences = sum(len(seq) for seq in user_sequences['visit_sequence'])
    print(f'Valid users: {len(user_sequences)}, Total sequences: {total_sequences}')

    # === Step 4: Collect all sequences ===
    print("Collecting sequences...")
    all_sequences = []
    all_userids = []
    for _, row in tqdm(user_sequences.iterrows(), total=len(user_sequences), desc="Processing users"):
        for seq in row['visit_sequence']:
            all_sequences.append(seq)
            all_userids.append(row['userid'])

    # === Step 5: Vectorized embedding extraction ===
    print("Converting sequences to embeddings (vectorized)...")
    sequences_array = np.array(all_sequences)
    history_indices = sequences_array[:, :-1]  # [N, SLIDE_WINDOW]
    target_indices = sequences_array[:, -1]  # [N]

    # Vectorized embedding lookup with padding handling
    # Add zero row at index 0 for padding
    embedding_matrix_padded = np.vstack([zero_embedding, embedding_matrix])
    # Shift indices: -1 (padding) -> 0 (zero_row), others -> idx + 1
    history_indices_safe = np.where(history_indices == PADDING_VALUE, 0, history_indices + 1)
    all_embeddings = embedding_matrix_padded[history_indices_safe]
    flattened_embeddings = all_embeddings.reshape(len(all_sequences), -1)

    # Get target categories from embeddings_df
    category_lookup = embeddings_df['category'].to_dict()
    category_lookup[PADDING_VALUE] = 'None'
    target_categories = [category_lookup.get(idx, 'None') for idx in target_indices]

    # === Step 6: Save intermediate sequences ===
    print("Saving intermediate sequences...")
    seq_column_names = ['userid'] + list(range(1, InputsConfig.SLIDE_WINDOW + 2))
    sequences_df = pd.DataFrame(
        [[uid] + seq for uid, seq in zip(all_userids, all_sequences)],
        columns=seq_column_names
    )
    sequences_df.to_parquet(sequences_path, index=False)
    print(f'Saved {len(sequences_df)} sequences to {sequences_path}')

    # === Step 7: Create and save output DataFrame ===
    print("Creating output DataFrame...")
    num_embedding_cols = InputsConfig.EMBEDDING_DIM * InputsConfig.SLIDE_WINDOW
    result_df = pd.DataFrame(flattened_embeddings, columns=[str(i) for i in range(num_embedding_cols)])
    result_df['next_category'] = target_categories
    result_df['userid'] = [str(uid) for uid in all_userids]

    print(f"Writing {len(result_df)} sequences to {next_input_path}...")
    result_df.to_parquet(next_input_path, index=False, engine='pyarrow')
    print(f"Success: Wrote {len(result_df)} sequences")

    gc.collect()
    print("Processing complete!")
    return result_df


from concurrent.futures import ProcessPoolExecutor


def create_input(state: str, embedding_engine: EmbeddingEngine, use_checkin_embeddings: bool = False):
    """
    Create input files for a given state and embedding engine.

    Generates:
    - Category input file (embeddings with categories)
    - Next-POI input file (sequences with embeddings)

    Args:
        state: State name (e.g., "florida", "alabama")
        embedding_engine: Embedding engine type (DGI, HGI, CHECK2HGI, etc.)
        use_checkin_embeddings: If True and engine is CHECK2HGI, use check-in level
            embeddings where each visit has its own unique embedding. If False,
            uses POI-level embeddings (same embedding for same POI).
            Only applicable for CHECK2HGI engine.
    """
    print(f"Processing state: {state} with embedding engine: {embedding_engine.value}")
    if use_checkin_embeddings:
        print("Using check-in level embeddings (each visit has unique embedding)")

    # Load data
    embeddings_df = IoPaths.load_embedd(state, embedding_engine)
    checkins_df = IoPaths.load_city(state)

    # Get output paths
    sequences_path = IoPaths.get_seq_next(state, embedding_engine)
    next_input_path = IoPaths.get_next(state, embedding_engine)
    category_input_path = IoPaths.get_category(state, embedding_engine)

    # Create directories
    sequences_path.parent.mkdir(parents=True, exist_ok=True)
    next_input_path.parent.mkdir(parents=True, exist_ok=True)
    category_input_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate category input (same for both methods)
    generate_category_input(embeddings_df.copy(), str(category_input_path))

    # Generate next-POI input using appropriate method
    if use_checkin_embeddings:
        generate_next_input_from_checkings(embeddings_df, str(sequences_path), str(next_input_path))
    else:
        generate_next_input_from_poi(embeddings_df, checkins_df, str(sequences_path), str(next_input_path))


if __name__ == '__main__':
    STATE_NAME = [
        ("florida", EmbeddingEngine.CHECK2HGI, True),  # Check-in level embeddings
        # ("texas", EmbeddingEngine.DGI, False),
        # ("alabama", EmbeddingEngine.DGI, False),
        # ("arizona", EmbeddingEngine.DGI, False),
        # ("georgia", EmbeddingEngine.DGI, False),
    ]
    with ProcessPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(create_input, state, engine, use_checkin_emb)
            for state, engine, use_checkin_emb in STATE_NAME
        ]
        results = [future.result() for future in futures]
