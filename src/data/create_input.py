from configs.model import InputsConfig
from configs.paths import OUTPUT_ROOT

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Union, Dict, Optional, Any
from tqdm import tqdm  # For progress tracking
import gc
from functools import lru_cache

# Define constants instead of magic numbers
SEQUENCE_LENGTH = 10
PADDING_VALUE = -1
MIN_SEQUENCE_LENGTH = 5


class PathConfig:
    """Class to handle file paths configuration."""

    def __init__(self, base_dir: str, checkins_file: str, sequences_file: str, output_file: str):
        self.base_dir = base_dir
        self.checkins_path = os.path.join(base_dir, checkins_file)
        self.sequences_path = os.path.join(base_dir, sequences_file)
        self.output_path = os.path.join(base_dir, output_file)

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.base_dir, exist_ok=True)

    def as_list(self) -> List[str]:
        """Return paths as a list for backward compatibility."""
        return ["", self.checkins_path, self.sequences_path, self.output_path]


def parse_and_sort_checkins(checkin_timestamps: List[str]) -> List[datetime]:
    """
    Parse and sort checkin timestamps using vectorized operations.

    Args:
        checkin_timestamps: List of timestamp strings

    Returns:
        List of sorted datetime objects
    """
    return sorted(pd.to_datetime(checkin_timestamps))


def generate_sequences(places_visited: List[int], max_sequences_per_user: Optional[int] = None) -> Union[
    List[List[int]], int]:
    """
    Generate fixed-length sequences from a list of places visited.

    Args:
        places_visited: List of place IDs representing sequential visits
        max_sequences_per_user: Maximum number of sequences to generate per user

    Returns:
        List of fixed-length sequences or 0 if input is too short
    """
    if len(places_visited) < MIN_SEQUENCE_LENGTH:
        return 0

    sequences = []
    for i in range(0, len(places_visited), InputsConfig.EMBEDDING_DIM):
        if len(places_visited) >= i + SEQUENCE_LENGTH:
            # Full sequence
            seq = places_visited[i:i + SEQUENCE_LENGTH]
        else:
            # Pad the last partial sequence
            remaining = places_visited[i:]
            seq = [PADDING_VALUE] * (SEQUENCE_LENGTH - len(remaining)) + remaining

        sequences.append(seq)

        # Limit sequences if max_sequences_per_user is specified
        if max_sequences_per_user and len(sequences) >= max_sequences_per_user:
            break

    return sequences


def generate_sequences_dataframe(users_ids: List[int], sequences: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    Generate a DataFrame of sequences and save to CSV more efficiently.

    Args:
        users_ids: List of user IDs
        sequences: DataFrame containing sequences indexed by user ID
        output_path: Path to save the resulting CSV file

    Returns:
        DataFrame containing generated sequences
    """
    # Preallocate data structure
    data = []

    for user_id in tqdm(users_ids, desc="Processing users"):
        try:
            if user_id not in sequences.index:
                continue

            user_sequences = sequences.loc[user_id].iloc[0]
            if not isinstance(user_sequences, list):
                continue

            # Process all trajectories at once
            for trajectory in user_sequences:
                data.append([user_id] + trajectory)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not process user ID {user_id}: {str(e)}")
            continue

    # Create DataFrame in one operation instead of row-by-row
    column_names = ['userid'] + list(range(1, SEQUENCE_LENGTH + 1))
    nextpoi_sequences = pd.DataFrame(data, columns=column_names)

    print(f'nextpoi_sequences shape: {nextpoi_sequences.shape}')

    # Save to CSV with proper error handling
    try:
        nextpoi_sequences.to_csv(output_path, index=False)
        print(f'Success: nextpoi_sequences saved at {output_path}\n')
    except IOError as e:
        print(f"Error saving to {output_path}: {str(e)}")

    return nextpoi_sequences


def processing_sequences_next(df_nextpoi_sequences: pd.DataFrame,
                              output_path: str,
                              embeddings_with_category: pd.DataFrame,
                              embeddings_without_category: pd.DataFrame,
                              save_step: int = 1000) -> pd.DataFrame:
    """
    Generate input data for next POI prediction model using batch processing.

    Args:
        df_nextpoi_sequences: DataFrame containing sequences indexed by user ID
        output_path: Path to save the resulting CSV file
        embeddings_with_category: DataFrame containing embeddings with category
        embeddings_without_category: DataFrame containing embeddings without category
        save_step: How often to save progress to disk
        EMBEDDING_SIZE: Size of the embeddings vector (default: 128)
        SEQUENCE_LENGTH: Length of the sequence (default: 10)

    Returns:
        DataFrame of generated input data
    """
    # Resume from existing file if it exists
    start_index = 0
    try:
        if os.path.exists(output_path):
            nextpoi_input = pd.read_csv(output_path)
            start_index = nextpoi_input.shape[0]
            print(f'Found existing file, resuming from index {start_index}\n')
        else:
            # Create empty DataFrame with proper columns
            col_count = (InputsConfig.EMBEDDING_DIM * InputsConfig.SLIDE_WINDOW)
            column_names = list(range(col_count)) + ['next_category', 'userid']
            nextpoi_input = pd.DataFrame(columns=column_names)
            # Pre-allocate memory for faster operations
            nextpoi_input = nextpoi_input.astype({col: np.float32 for col in range(col_count - 1)})
            nextpoi_input = nextpoi_input.astype({'userid': str})
    except Exception as e:
        print(f"Error reading existing file: {str(e)}. Starting from scratch.")
        col_count = (InputsConfig.EMBEDDING_DIM * InputsConfig.SLIDE_WINDOW)
        column_names = list(range(col_count)) + ['next_category', 'userid']
        nextpoi_input = pd.DataFrame(columns=column_names)
        # Pre-allocate memory with appropriate types
        nextpoi_input = nextpoi_input.astype({col: np.float32 for col in range(col_count - 1)})
        nextpoi_input = nextpoi_input.astype({'userid': str})
        start_index = 0

    # Create lookup dictionaries for faster access
    category_lookup = embeddings_with_category['category'].to_dict()

    # Convert embeddings to numpy array for faster access
    emb_lookup = {poi_id: row.values.astype(np.float32) for poi_id, row in embeddings_without_category.iterrows()}
    zero_emb = np.zeros(InputsConfig.EMBEDDING_DIM, dtype=np.float32)

    # Process in batches
    batch_size = min(save_step, len(df_nextpoi_sequences))
    total_batches = (len(df_nextpoi_sequences) - start_index + batch_size - 1) // batch_size

    # Preallocate memory for a batch of results
    results = []

    iterator = tqdm(enumerate(range(start_index, len(df_nextpoi_sequences), batch_size)), desc="Processing batches",
                    total=total_batches)

    # Cache for frequently accessed embeddings
    @lru_cache(maxsize=None)
    def get_embedding(poi_id):
        return emb_lookup.get(poi_id, zero_emb)

    for batch_idx, batch_start in iterator:
        batch_end = min(batch_start + batch_size, len(df_nextpoi_sequences))
        batch = df_nextpoi_sequences.iloc[batch_start:batch_end]

        # Clear results for this batch
        results = []

        for i, (userid, row) in enumerate(batch.iterrows()):
            try:
                # Get categories using the lookup dictionary
                categoria_target = category_lookup.get(row.iloc[InputsConfig.SLIDE_WINDOW], 'None')

                # Get the sequence without the target (more efficient than dropping)
                sequence_without_target = row.iloc[:InputsConfig.SLIDE_WINDOW].values

                # Use lookup dictionary instead of DataFrame access
                # Preallocate array for embeddings concatenation
                all_embeddings = np.empty((InputsConfig.SLIDE_WINDOW, InputsConfig.EMBEDDING_DIM), dtype=np.float32)

                # Fill the array with embeddings
                for j, poi in enumerate(sequence_without_target):
                    all_embeddings[j] = get_embedding(poi)

                # Flatten all embeddings at once
                poi_embedding = all_embeddings.flatten()

                # Combine embeddings, category and user ID
                # Using list comprehension for better performance
                sample = np.append(poi_embedding, [categoria_target] + [userid])
                results.append(sample)
            except Exception as e:
                print(f"Error processing row {batch_start + i} (user {userid}): {str(e)}")
                continue

        # Bulk add results to DataFrame
        if results:
            # Convert results to DataFrame and append
            batch_df = pd.DataFrame(results, columns=nextpoi_input.columns)
            nextpoi_input = pd.concat([nextpoi_input, batch_df], ignore_index=True)

        # Save checkpoint
        try:
            # Use efficient CSV writing
            nextpoi_input.to_csv(output_path, index=False, float_format='%.5f')
            iterator.set_postfix_str(f"Saved {len(nextpoi_input)} rows")

            # Explicitly call garbage collection after saving
            gc.collect()

        except IOError as e:
            print(f"Warning: Could not save progress: {str(e)}")

    print(f'Final save: {len(nextpoi_input)} rows to {output_path}')
    return nextpoi_input


def generate_next_input(df_embb, df_filter,
                        sequences_path: str,
                        next_input_path: str,
                        max_sequences_per_user: Optional[int] = None) -> pd.DataFrame:
    """
    Generate input data for next POI prediction model with improved performance.

    Args:
        embeddings: DataFrame containing POI embeddings
        paths: Either a PathConfig object or a list of file paths:
               [unused, checkins_path, sequences_path, output_path]
        max_sequences_per_user: Maximum number of sequences to generate per user

    Returns:
        DataFrame of generated input data
    """

    try:
        # Load check-ins data and convert datetime in one operation
        df_filter['datetime'] = pd.to_datetime(df_filter['datetime'])
    except Exception as e:
        print(f"Error loading check-ins data: {str(e)}")
        return pd.DataFrame()

    # Process users and generate sequences
    print("Generating sequences for each user...")
    user_ids = []
    visit_sequences = []

    # Sort once then group
    checkins_sorted = df_filter.sort_values(by=['userid', 'datetime'])

    for userid, group in tqdm(checkins_sorted.groupby('userid'), desc="Processing users"):
        places = group['placeid'].tolist()
        sequences = generate_sequences(places, max_sequences_per_user)

        # Skip users with insufficient data
        if sequences == 0:
            continue

        user_ids.append(userid)
        visit_sequences.append(sequences)

    # Create DataFrame of user IDs and their sequences
    checkins_sequence_by_user = pd.DataFrame({
        'userid': user_ids,
        'visit_sequence': visit_sequences
    })

    print(f'Users with valid sequences: {len(user_ids)}')

    # Set index for faster lookups
    sequences = checkins_sequence_by_user.set_index('userid')
    unique_users = sequences.index.unique()

    # Prepare embeddings - optimize by creating copies only when needed
    print("Preparing embeddings...")
    embeddings_with_category = df_embb.copy()
    embeddings_without_category = df_embb.drop(columns=['category'])

    # Add padding embedding
    padding_embedding = [0] * InputsConfig.EMBEDDING_DIM

    if PADDING_VALUE not in embeddings_with_category.index:
        embeddings_with_category.loc[PADDING_VALUE] = padding_embedding + [0]
        embeddings_with_category.loc[PADDING_VALUE, 'category'] = 'None'

        padding_df = pd.DataFrame([padding_embedding],
                                  columns=embeddings_without_category.columns,
                                  index=[PADDING_VALUE])
        embeddings_without_category = pd.concat([embeddings_without_category, padding_df])

    # Generate sequences CSV
    print("Generating sequences DataFrame...")
    next_sequences = generate_sequences_dataframe(unique_users, sequences, sequences_path)
    next_sequences.set_index('userid', inplace=True)

    # Generate input data directly without re-reading the CSV
    print("Generating next POI input data...")
    next_input = processing_sequences_next(
        next_sequences,
        next_input_path,
        embeddings_with_category,
        embeddings_without_category
    )

    print("Processing complete!")
    return next_input


def generate_category_input(df_embb: pd.DataFrame, category_path: str):
    df_embb.set_index('placeid', inplace=True)
    df_embb.to_csv(category_path)


from concurrent.futures import ProcessPoolExecutor


def process_state(state):
    df_embb = pd.read_csv(f'{OUTPUT_ROOT}/{state}/{state}-embeddings.csv')
    df_filter = pd.read_csv(f'{OUTPUT_ROOT}/{state}/{state}-filtrado.csv')
    output_path = f'{OUTPUT_ROOT}/{state}/pre-processing/'
    sequences_path = f'{output_path}poi-sequences.csv'
    next_input_path = f'{output_path}next-input.csv'
    category_input_path = f'{output_path}category-input.csv'

    os.makedirs(output_path, exist_ok=True)

    generate_category_input(df_embb, category_input_path)
    generate_next_input(df_embb, df_filter, sequences_path, next_input_path)


if __name__ == '__main__':
    # STATE_NAME = ["alabama","arizona","california", "florida", "georgia", "texas"]
    STATE_NAME = ["florida_new"]
    with ProcessPoolExecutor(max_workers=12) as executor:
        executor.map(process_state, STATE_NAME)
