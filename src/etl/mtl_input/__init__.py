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
from __future__ import annotations

import importlib
import sys

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

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # core
    'generate_sequences': ('etl.mtl_input.core', 'generate_sequences'),
    'create_embedding_lookup': ('etl.mtl_input.core', 'create_embedding_lookup'),
    'create_category_lookup': ('etl.mtl_input.core', 'create_category_lookup'),
    'get_zero_embedding': ('etl.mtl_input.core', 'get_zero_embedding'),
    'parse_and_sort_checkins': ('etl.mtl_input.core', 'parse_and_sort_checkins'),
    'save_parquet': ('etl.mtl_input.core', 'save_parquet'),
    'save_next_input_dataframe': ('etl.mtl_input.core', 'save_next_input_dataframe'),
    'convert_sequences_to_poi_embeddings': ('etl.mtl_input.core', 'convert_sequences_to_poi_embeddings'),
    'PADDING_VALUE': ('etl.mtl_input.core', 'PADDING_VALUE'),
    'MIN_SEQUENCE_LENGTH': ('etl.mtl_input.core', 'MIN_SEQUENCE_LENGTH'),
    'DEFAULT_BATCH_SIZE': ('etl.mtl_input.core', 'DEFAULT_BATCH_SIZE'),
    'MISSING_CATEGORY_VALUE': ('etl.mtl_input.core', 'MISSING_CATEGORY_VALUE'),
    # loaders
    'EmbeddingLoader': ('etl.mtl_input.loaders', 'EmbeddingLoader'),
    # builders
    'generate_category_input': ('etl.mtl_input.builders', 'generate_category_input'),
    'generate_next_input_from_poi': ('etl.mtl_input.builders', 'generate_next_input_from_poi'),
    'generate_next_input_from_checkins': ('etl.mtl_input.builders', 'generate_next_input_from_checkins'),
    # fusion
    'EmbeddingAligner': ('etl.mtl_input.fusion', 'EmbeddingAligner'),
    'EmbeddingFuser': ('etl.mtl_input.fusion', 'EmbeddingFuser'),
    'MultiEmbeddingInputGenerator': ('etl.mtl_input.fusion', 'MultiEmbeddingInputGenerator'),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        mod = importlib.import_module(module_name)
        value = getattr(mod, attr)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module 'etl.mtl_input' has no attribute {name!r}")
