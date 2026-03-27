"""Shim — canonical location is research/embeddings/ (Phase 5).

This package will be removed at the end of Phase 6.
Embedding trainers now live under research/embeddings/.
"""
import warnings as _warnings

_warnings.warn(
    "src/embeddings is deprecated; embedding trainers moved to research/embeddings/. "
    "This shim will be removed at the end of Phase 6.",
    DeprecationWarning,
    stacklevel=2,
)
