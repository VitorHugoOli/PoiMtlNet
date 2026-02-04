"""
MTL Input Generation Module

Modular, maintainable input generation for MTLnet training.

This module replaces the monolithic create_input.py with focused components:
- core.py: Pure logic functions (generate_sequences, create_*_lookup)
- loaders.py: Data loading with caching (EmbeddingLoader)
- builders.py: Input generation functions (generate_category_input, generate_next_input_*)
- fusion.py: Multi-embedding fusion (EmbeddingAligner, EmbeddingFuser, MultiEmbeddingInputGenerator)

Usage:
    # Core functions
    from etl.mtl_input import generate_sequences, create_embedding_lookup

    # Loaders
    from etl.mtl_input import EmbeddingLoader

    # Builders (business logic for input generation)
    from etl.mtl_input import generate_category_input, generate_next_input_from_poi

    # Fusion
    from etl.mtl_input import MultiEmbeddingInputGenerator
"""

from .core import (
    generate_sequences,
    create_embedding_lookup,
    create_category_lookup,
    get_zero_embedding,
    parse_and_sort_checkins,
    save_parquet,
    save_next_input_dataframe,
    convert_sequences_to_poi_embeddings,
    PADDING_VALUE,
    MIN_SEQUENCE_LENGTH,
    DEFAULT_BATCH_SIZE,
    MISSING_CATEGORY_VALUE,
)
from .loaders import EmbeddingLoader
from .builders import (
    generate_category_input,
    generate_next_input_from_poi,
    generate_next_input_from_checkins,
)
from .fusion import (
    EmbeddingAligner,
    EmbeddingFuser,
    MultiEmbeddingInputGenerator,
)

__all__ = [
    # Core functions
    'generate_sequences',
    'create_embedding_lookup',
    'create_category_lookup',
    'get_zero_embedding',
    'parse_and_sort_checkins',
    'save_parquet',
    'save_next_input_dataframe',
    'convert_sequences_to_poi_embeddings',
    # Constants
    'PADDING_VALUE',
    'MIN_SEQUENCE_LENGTH',
    'DEFAULT_BATCH_SIZE',
    'MISSING_CATEGORY_VALUE',
    # Loaders
    'EmbeddingLoader',
    # Builders
    'generate_category_input',
    'generate_next_input_from_poi',
    'generate_next_input_from_checkins',
    # Fusion
    'EmbeddingAligner',
    'EmbeddingFuser',
    'MultiEmbeddingInputGenerator',
]

__version__ = '2.0.0'
