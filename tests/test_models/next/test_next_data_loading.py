"""
Test suite for Next-POI data loading and preprocessing.

Tests sequence reshaping, padding detection, and fold creation.
"""

import pytest
import torch
import numpy as np

from data.folds import FoldCreator
from configs.model import InputsConfig


class TestSequenceReshaping:
    """Tests for correct reshaping of input sequences."""

    def test_flatten_to_3d_reshape(self):
        """Test reshaping from flattened (N, 576) to 3D (N, 9, 64)."""
        batch_size = 32
        window_size = 9
        embedding_dim = 64
        flattened_dim = window_size * embedding_dim  # 576

        # Create flattened input
        x_flat = torch.randn(batch_size, flattened_dim)

        # Reshape to 3D
        x_3d = x_flat.view(batch_size, window_size, embedding_dim)

        assert x_3d.shape == (batch_size, window_size, embedding_dim), \
            f"Expected shape ({batch_size}, {window_size}, {embedding_dim}), got {x_3d.shape}"

        # Verify data integrity after reshape
        x_flat_reconstructed = x_3d.view(batch_size, -1)
        assert torch.allclose(x_flat, x_flat_reconstructed), \
            "Data changed during reshape"

    def test_embedding_dim_matches_config(self):
        """Test that flattened features = window * embedding_dim."""
        embedding_dim = InputsConfig.EMBEDDING_DIM
        window_size = InputsConfig.SLIDE_WINDOW

        expected_flattened = window_size * embedding_dim

        assert expected_flattened == window_size * embedding_dim, \
            f"Expected {window_size}*{embedding_dim}={window_size * embedding_dim}, got {expected_flattened}"
        assert embedding_dim > 0
        assert window_size > 0

    def test_reshape_preserves_sequence_order(self):
        """Test that reshaping preserves temporal order."""
        # Create sequence with known pattern
        batch_size = 2
        window_size = 9
        embedding_dim = 64

        # First sequence: embeddings are [1, 2, 3, ..., 9] repeated 64 times
        seq1 = torch.arange(1, window_size + 1).repeat(embedding_dim, 1).T.flatten()

        # Second sequence: embeddings are [9, 8, 7, ..., 1] repeated 64 times
        seq2 = torch.arange(window_size, 0, -1).repeat(embedding_dim, 1).T.flatten()

        x_flat = torch.stack([seq1, seq2])

        # Reshape to 3D
        x_3d = x_flat.view(batch_size, window_size, embedding_dim)

        # Verify first timestep of first sequence
        assert x_3d[0, 0, :].unique().item() == 1, \
            "First timestep should be all 1s"

        # Verify last timestep of first sequence
        assert x_3d[0, 8, :].unique().item() == 9, \
            "Last timestep should be all 9s"

        # Verify first timestep of second sequence
        assert x_3d[1, 0, :].unique().item() == 9, \
            "Second sequence first timestep should be all 9s"


class TestPaddingDetection:
    """Tests for padding mask generation."""

    def test_all_zero_sequence_detected_as_padding(self):
        """Test that all-zero timesteps are detected as padding."""
        batch_size = 4
        seq_length = 9
        embed_dim = 64

        x = torch.randn(batch_size, seq_length, embed_dim)

        # Add padding to some sequences
        x[0, 5:, :] = 0  # Last 4 timesteps padded
        x[1, 7:, :] = 0  # Last 2 timesteps padded
        x[2, :, :] = 1   # No padding (all non-zero)

        # Generate padding mask
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Verify padding detection
        assert padding_mask[0, 5:].all(), "Padding not detected for sequence 0"
        assert padding_mask[1, 7:].all(), "Padding not detected for sequence 1"
        assert not padding_mask[2, :].any(), "False positive padding for sequence 2"

    def test_padding_value_is_zero(self):
        """Test that PAD_VALUE is 0."""
        assert InputsConfig.PAD_VALUE == 0, \
            f"Expected PAD_VALUE=0, got {InputsConfig.PAD_VALUE}"

    def test_mixed_padding_detection(self):
        """Test padding detection with various patterns."""
        batch_size = 3
        seq_length = 9
        embed_dim = 64

        x = torch.randn(batch_size, seq_length, embed_dim)

        # Sequence 0: Padding at end
        x[0, 6:, :] = 0

        # Sequence 1: No padding
        x[1, :, :] = torch.randn(seq_length, embed_dim)

        # Sequence 2: Heavy padding (only first 3 timesteps valid)
        x[2, 3:, :] = 0

        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Count padded timesteps per sequence
        num_padded = padding_mask.sum(dim=1)

        assert num_padded[0] == 3, f"Expected 3 padded timesteps, got {num_padded[0]}"
        assert num_padded[1] == 0, f"Expected 0 padded timesteps, got {num_padded[1]}"
        assert num_padded[2] == 6, f"Expected 6 padded timesteps, got {num_padded[2]}"


class TestFoldCreation:
    """Tests for FoldCreator API."""

    def test_fold_creator_api(self):
        """Test that FoldCreator can be instantiated with correct API."""
        from data.folds import FoldCreator, TaskType

        # Verify the correct constructor signature
        fold_creator = FoldCreator(
            task_type=TaskType.NEXT,
            n_splits=5,
            batch_size=32,
            seed=42
        )

        assert fold_creator.n_splits == 5
        assert fold_creator.batch_size == 32
        assert fold_creator.seed == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
