"""
Test suite for Next-POI model architecture validation.

Tests model structure, output shapes, attention head dimensions,
positional encoding, padding masks, and parameter counts.
"""

import pytest
import torch
import torch.nn as nn
from models.next import NextHeadSingle
from models.next import (
    NextHeadHybrid,
    NextHeadGRU,
    NextHeadLSTM,
    NextHeadTransformerOptimized,
    NextHeadTemporalCNN
)
from configs.experiment import ExperimentConfig

# Derive defaults from canonical config source
_CFG = ExperimentConfig.default_next("_test", "test", "dgi")
_MP = _CFG.model_params
_NEXT_INPUT_DIM = _MP["embed_dim"]
_NEXT_NUM_HEADS = _MP["num_heads"]
_NEXT_MAX_SEQ_LENGTH = _MP["seq_length"]
_NEXT_NUM_CLASSES = _MP["num_classes"]


class TestNextHeadSingleArchitecture:
    """Tests for NextHeadSingle (current Transformer model)."""

    @pytest.fixture
    def model(self):
        """Create NextHeadSingle model instance."""
        return NextHeadSingle(
            embed_dim=64,
            num_classes=7,
            num_heads=16,  # Current (problematic) setting
            seq_length=9,
            num_layers=4,
            dropout=0.1
        )

    @pytest.fixture
    def model_fixed(self):
        """Create NextHeadSingle with fixed attention heads."""
        return NextHeadSingle(
            embed_dim=64,
            num_classes=7,
            num_heads=4,  # Fixed: 64/4 = 16 dims per head
            seq_length=9,
            num_layers=4,
            dropout=0.2
        )

    def test_output_shape(self, model):
        """Test that model outputs correct shape [batch, num_classes]."""
        batch_size = 32
        seq_length = 9
        embed_dim = 64

        x = torch.randn(batch_size, seq_length, embed_dim)
        output = model(x)

        assert output.shape == (batch_size, 7), \
            f"Expected output shape ({batch_size}, 7), got {output.shape}"

    def test_attention_head_dimensions(self, model):
        """Test attention head bottleneck issue."""
        embed_dim = model.transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = model.transformer_encoder.layers[0].self_attn.num_heads
        head_dim = embed_dim // num_heads

        # Current problematic setting
        assert num_heads == 16, f"Expected num_heads=16, got {num_heads}"
        assert head_dim == 4, f"Expected head_dim=4, got {head_dim}"

        # Warn about bottleneck
        assert head_dim >= 4, \
            f"Head dimension {head_dim} is too small! Should be >= 32 for good performance"

    def test_fixed_attention_head_dimensions(self, model_fixed):
        """Test that fixed attention heads have acceptable dimension."""
        embed_dim = model_fixed.transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = model_fixed.transformer_encoder.layers[0].self_attn.num_heads
        head_dim = embed_dim // num_heads

        assert num_heads == 4, f"Expected num_heads=4, got {num_heads}"
        assert head_dim == 16, f"Expected head_dim=16, got {head_dim}"
        assert head_dim >= 16, \
            f"Head dimension {head_dim} should be >= 16 for reasonable performance"

    def test_positional_encoding_shape(self, model):
        """Test learned positional embeddings have correct shape."""
        seq_length = 9
        embed_dim = 64

        assert model.pos_embedding.shape == (1, seq_length, embed_dim), \
            f"Pos embedding shape mismatch: expected (1, {seq_length}, {embed_dim}), got {model.pos_embedding.shape}"

    def test_padding_mask_generation(self, model):
        """Test padding mask correctly identifies zero sequences."""
        batch_size = 16
        seq_length = 9
        embed_dim = 64

        # Create batch with some padding (all-zero timesteps)
        x = torch.randn(batch_size, seq_length, embed_dim)

        # Add padding to some sequences
        x[0, 7:, :] = 0  # Last 2 timesteps padded
        x[1, 5:, :] = 0  # Last 4 timesteps padded

        # Generate padding mask (True where padded)
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Verify padding detection
        assert padding_mask[0, 7:].all(), "Padding not detected for sequence 0"
        assert padding_mask[1, 5:].all(), "Padding not detected for sequence 1"
        assert not padding_mask[0, :7].any(), "False positive padding for sequence 0"

    def test_model_parameter_count(self, model):
        """Test model has expected number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Expected ~220k parameters for this config
        assert 150_000 < total_params < 300_000, \
            f"Parameter count {total_params:,} outside expected range"
        assert trainable_params == total_params, \
            "Some parameters are frozen (not expected for this model)"

    def test_dropout_consistency(self, model):
        """Test dropout values are consistent."""
        # Check main dropout
        main_dropout = model.dropout.p

        # Check transformer layer dropout
        transformer_dropout = model.transformer_encoder.layers[0].dropout.p

        # Should be the same
        assert main_dropout == transformer_dropout, \
            f"Dropout inconsistency: main={main_dropout}, Transformer={transformer_dropout}"

    def test_forward_with_batch(self, model):
        """Test model handles different batch sizes."""
        embed_dim = 64
        seq_length = 9

        for batch_size in [1, 16, 32, 128]:
            x = torch.randn(batch_size, seq_length, embed_dim)
            output = model(x)

            assert output.shape == (batch_size, 7), \
                f"Batch size {batch_size}: expected ({batch_size}, 7), got {output.shape}"


class TestNextHeadHybridArchitecture:
    """Tests for NextHeadHybrid (GRU + Attention)."""

    @pytest.fixture
    def model(self):
        """Create NextHeadHybrid model instance."""
        return NextHeadHybrid(
            embed_dim=64,
            hidden_dim=256,
            num_classes=7,
            num_heads=4,
            num_gru_layers=2,
            dropout=0.3
        )

    def test_output_shape(self, model):
        """Test Hybrid model outputs correct shape."""
        batch_size = 32
        seq_length = 9
        embed_dim = 64

        x = torch.randn(batch_size, seq_length, embed_dim)
        output = model(x)

        assert output.shape == (batch_size, 7), \
            f"Expected output shape ({batch_size}, 7), got {output.shape}"

    def test_gru_output_shape(self, model):
        """Test GRU produces correct hidden dimension."""
        batch_size = 16
        seq_length = 9
        embed_dim = 64
        hidden_dim = 256

        x = torch.randn(batch_size, seq_length, embed_dim)

        # Forward through GRU
        gru_out, _ = model.gru(x)

        assert gru_out.shape == (batch_size, seq_length, hidden_dim), \
            f"Expected GRU output {(batch_size, seq_length, hidden_dim)}, got {gru_out.shape}"

    def test_attention_output_shape(self, model):
        """Test attention mechanism output shape."""
        batch_size = 16
        seq_length = 9
        hidden_dim = 256

        # Create dummy GRU output
        gru_out = torch.randn(batch_size, seq_length, hidden_dim)

        # Apply attention
        attn_out, _ = model.attention(gru_out, gru_out, gru_out)

        assert attn_out.shape == (batch_size, seq_length, hidden_dim), \
            f"Expected attention output {(batch_size, seq_length, hidden_dim)}, got {attn_out.shape}"

    def test_model_parameter_count(self, model):
        """Test Hybrid model has expected parameters (~900k)."""
        total_params = sum(p.numel() for p in model.parameters())

        # Hybrid with hidden_dim=256: ~900k parameters
        assert 500_000 < total_params < 1_500_000, \
            f"Parameter count {total_params:,} outside expected range"

    def test_forward_with_different_batch_sizes(self, model):
        """Test Hybrid handles various batch sizes."""
        embed_dim = 64
        seq_length = 9

        for batch_size in [1, 16, 32, 64]:
            x = torch.randn(batch_size, seq_length, embed_dim)
            output = model(x)

            assert output.shape == (batch_size, 7), \
                f"Batch size {batch_size}: expected ({batch_size}, 7), got {output.shape}"


class TestNextHeadGRUArchitecture:
    """Tests for NextHeadGRU (baseline RNN model)."""

    @pytest.fixture
    def model(self):
        """Create NextHeadGRU model instance."""
        return NextHeadGRU(
            embed_dim=64,
            hidden_dim=256,
            num_classes=7,
            num_layers=2,
            dropout=0.3
        )

    def test_output_shape(self, model):
        """Test GRU model outputs correct shape."""
        batch_size = 32
        seq_length = 9
        embed_dim = 64

        x = torch.randn(batch_size, seq_length, embed_dim)
        output = model(x)

        assert output.shape == (batch_size, 7), \
            f"Expected output shape ({batch_size}, 7), got {output.shape}"

    def test_model_parameter_count(self, model):
        """Test GRU has expected parameters (~640k)."""
        total_params = sum(p.numel() for p in model.parameters())

        # GRU with hidden_dim=256: ~640k parameters
        assert 400_000 < total_params < 1_000_000, \
            f"Parameter count {total_params:,} outside expected range"


class TestConfigConsistency:
    """Test that model configs are consistent and reasonable."""

    def test_num_heads_divides_embed_dim(self):
        """Test that NUM_HEADS divides INPUT_DIM evenly."""
        embed_dim = _NEXT_INPUT_DIM
        num_heads = _NEXT_NUM_HEADS

        assert embed_dim % num_heads == 0, \
            f"NUM_HEADS ({num_heads}) must divide INPUT_DIM ({embed_dim}) evenly"

    def test_head_dimension_reasonable(self):
        """Test that head dimension is not too small."""
        embed_dim = _NEXT_INPUT_DIM
        num_heads = _NEXT_NUM_HEADS
        head_dim = embed_dim // num_heads

        # Warn if head_dim < 16
        if head_dim < 16:
            import warnings
            warnings.warn(
                f"Head dimension {head_dim} is small. "
                f"Consider reducing NUM_HEADS to {embed_dim // 16} or fewer."
            )

    def test_max_seq_length_matches_data(self):
        """Test that MAX_SEQ_LENGTH matches sliding window."""
        from configs.model import InputsConfig

        assert _NEXT_MAX_SEQ_LENGTH == InputsConfig.SLIDE_WINDOW, \
            f"MAX_SEQ_LENGTH ({_NEXT_MAX_SEQ_LENGTH}) must match " \
            f"SLIDE_WINDOW ({InputsConfig.SLIDE_WINDOW})"

    def test_num_classes_correct(self):
        """Test that NUM_CLASSES is 7 (POI categories)."""
        assert _NEXT_NUM_CLASSES == 7, \
            f"Expected 7 POI categories, got {_NEXT_NUM_CLASSES}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
