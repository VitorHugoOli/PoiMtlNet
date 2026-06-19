"""
Input generation builders for MTL tasks.

This module contains the business logic for generating category and next-POI inputs
from embeddings. These are pure transformation functions used by pipeline orchestration.

Moved from pipelines/create_inputs.pipe.py to follow the pattern where pipes only
handle orchestration and modules contain business logic.
"""
from typing import Optional
import json
import subprocess
from datetime import datetime, timezone
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
    MIN_SEQUENCE_LENGTH,
)


def _git_sha() -> str:
    """Best-effort current git commit hash; 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _write_build_provenance(
    state: str,
    engine: EmbeddingEngine,
    task: str,
    *,
    min_sequence_length: int,
    stride: Optional[int],
    window_size: int,
) -> None:
    """Write an additive build-provenance sidecar next to the task's input parquet.

    Path: ``<engine_dir>/input/<task>_build_provenance.json``. This NEVER gates
    anything — it only records how the windowing-dependent inputs were built so a
    later reviewer can tell a stride-1/min_seq-10 board build apart from the frozen
    stride-9/min_seq-5 default build. ``stride=None`` means non-overlapping
    (step == window_size), matching the default code path.
    """
    try:
        # Anchor the sidecar to the same input dir the next parquet lives in.
        next_path = IoPaths.get_next(state, engine)
        sidecar = next_path.parent / f"{task}_build_provenance.json"
        payload = {
            "task": task,
            "engine": engine.value,
            "state": state,
            "min_sequence_length": min_sequence_length,
            # JSON-friendly: record the effective step explicitly too.
            "stride": stride,
            "effective_step": stride if stride is not None else window_size,
            "window_size": window_size,
            "git_sha": _git_sha(),
            "utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        # Provenance is best-effort and must never break a build.
        pass


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
    batch_size: int = DEFAULT_BATCH_SIZE,
    stride: Optional[int] = None,
    min_sequence_length: int = MIN_SEQUENCE_LENGTH,
) -> None:
    """
    Generate next-POI input from POI-level embeddings.

    Uses embeddings that are constant per POI (e.g., HGI, Space2Vec, DGI).

    Args:
        state: State name
        engine: Embedding engine
        batch_size: Batch size for processing sequences
        stride: Step between sequence starts. ``None`` (default) → step ==
            window_size (non-overlapping), byte-identical to the legacy path.
        min_sequence_length: Minimum user check-ins to emit any sequence.
            Default == MIN_SEQUENCE_LENGTH (5) → legacy behaviour preserved.
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
        lambda places: generate_sequences(
            places.tolist(),
            stride=stride,
            min_sequence_length=min_sequence_length,
        )
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

    # Additive build-provenance sidecar (never gates anything)
    _write_build_provenance(
        state, engine, "next",
        min_sequence_length=min_sequence_length,
        stride=stride,
        window_size=window_size,
    )


def generate_next_input_from_checkins(
    state: str,
    engine: EmbeddingEngine,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stride: int = None,
    min_sequence_length: int = MIN_SEQUENCE_LENGTH,
) -> None:
    """
    Generate next-POI input from check-in-level embeddings.

    Uses embeddings that vary per check-in (e.g., Time2Vec).
    Each check-in already has its contextual embedding.

    Args:
        state: State name
        engine: Embedding engine
        batch_size: Batch size for processing sequences (unused, kept for API compatibility)
        stride: Step between sequence starts. ``None`` (default) → step ==
            window_size (non-overlapping), byte-identical to the legacy path.
        min_sequence_length: Minimum user check-ins to emit any sequence.
            Default == MIN_SEQUENCE_LENGTH (5) → legacy behaviour preserved.
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

    # Process each user using shared function
    all_results = []
    all_sequences = []

    for userid, user_df in tqdm(embeddings_df.groupby('userid'), desc="Processing users"):
        user_df = user_df.reset_index(drop=True)

        results, sequences = convert_user_checkins_to_sequences(
            user_df, emb_cols, window_size, embedding_dim,
            stride=stride, min_sequence_length=min_sequence_length,
        )

        all_results.extend(results)
        all_sequences.extend(sequences)

    # Save intermediate sequences (for debugging/analysis)
    seq_cols = [f'poi_{i}' for i in range(window_size)] + ['target_poi', 'userid']
    sequences_df = pd.DataFrame(all_sequences, columns=seq_cols)
    # POI columns may mix str placeids with int padding (-1); unify to str for parquet
    poi_cols = [f'poi_{i}' for i in range(window_size)] + ['target_poi']
    for col in poi_cols:
        sequences_df[col] = sequences_df[col].astype(str)
    sequences_path = IoPaths.get_seq_next(state, engine)
    save_parquet(sequences_df, sequences_path)

    # Save output DataFrame
    save_next_input_dataframe(all_results, window_size, embedding_dim, state, engine)

    # Additive build-provenance sidecar (never gates anything)
    _write_build_provenance(
        state, engine, "next",
        min_sequence_length=min_sequence_length,
        stride=stride,
        window_size=window_size,
    )
