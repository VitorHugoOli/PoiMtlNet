"""
Test suite for Next-POI configuration consistency.

Validates structural invariants and relationships between config values.
Does NOT assert specific default values -- those are tuned freely.
"""

import pytest

from configs.experiment import ExperimentConfig
from configs.model import InputsConfig
from models.heads.next import NextHeadSingle

# Derive all defaults from the canonical config source
_CFG = ExperimentConfig.default_next("_test", "test", "dgi")
_MP = _CFG.model_params

NEXT_INPUT_DIM = _MP["embed_dim"]
NEXT_NUM_HEADS = _MP["num_heads"]
NEXT_NUM_LAYERS = _MP["num_layers"]
NEXT_MAX_SEQ_LENGTH = _MP["seq_length"]
NEXT_NUM_CLASSES = _MP["num_classes"]
NEXT_DROPOUT = _MP["dropout"]

NEXT_LR = _CFG.learning_rate
NEXT_MAX_LR = _CFG.max_lr
NEXT_WEIGHT_DECAY = _CFG.weight_decay
NEXT_MAX_GRAD_NORM = _CFG.max_grad_norm

NEXT_BATCH_SIZE = _CFG.batch_size
NEXT_EPOCHS = _CFG.epochs
NEXT_K_FOLDS = _CFG.k_folds


class TestNumHeadsConfiguration:
    """Tests for attention head configuration."""

    def test_num_heads_divides_embed_dim(self):
        """num_heads must divide embed_dim evenly."""
        assert NEXT_INPUT_DIM % NEXT_NUM_HEADS == 0, \
            f"NUM_HEADS ({NEXT_NUM_HEADS}) must divide INPUT_DIM ({NEXT_INPUT_DIM}) evenly."

    def test_head_dimension_at_least_eight(self):
        """Each attention head should have at least 8 dimensions."""
        head_dim = NEXT_INPUT_DIM // NEXT_NUM_HEADS
        assert head_dim >= 8, \
            f"Head dimension {head_dim} is too small (< 8). " \
            f"Consider reducing NUM_HEADS."


class TestDropoutConfiguration:
    """Tests for dropout values."""

    def test_dropout_in_valid_range(self):
        """Dropout must be in [0, 1)."""
        assert 0 <= NEXT_DROPOUT < 1, \
            f"DROPOUT {NEXT_DROPOUT} outside valid range [0, 1)"

    def test_dropout_matches_model(self):
        """Config dropout must propagate correctly to model."""
        model = NextHeadSingle(
            embed_dim=NEXT_INPUT_DIM,
            num_classes=NEXT_NUM_CLASSES,
            num_heads=NEXT_NUM_HEADS,
            seq_length=NEXT_MAX_SEQ_LENGTH,
            num_layers=NEXT_NUM_LAYERS,
            dropout=NEXT_DROPOUT
        )
        assert model.dropout.p == NEXT_DROPOUT, \
            f"Model dropout ({model.dropout.p}) != config dropout ({NEXT_DROPOUT})"


class TestEmbeddingDimensionConfiguration:
    """Tests for embedding dimension consistency."""

    def test_embedding_dim_positive(self):
        """Embedding dimension must be positive."""
        assert InputsConfig.EMBEDDING_DIM > 0

    def test_fusion_dim_handling(self):
        """get_next_dim() should return a positive value if it exists."""
        try:
            next_dim = InputsConfig.get_next_dim()
            assert next_dim > 0
        except AttributeError:
            pytest.skip("InputsConfig.get_next_dim() not available")


class TestMaxSeqLengthConfiguration:
    """Tests for sequence length configuration."""

    def test_max_seq_length_matches_slide_window(self):
        """seq_length in model params must match InputsConfig.SLIDE_WINDOW."""
        assert NEXT_MAX_SEQ_LENGTH == InputsConfig.SLIDE_WINDOW, \
            f"seq_length ({NEXT_MAX_SEQ_LENGTH}) must match SLIDE_WINDOW ({InputsConfig.SLIDE_WINDOW})"


class TestLearningRateConfiguration:
    """Tests for learning rate settings."""

    def test_lr_positive(self):
        """Learning rate must be positive."""
        assert NEXT_LR > 0

    def test_max_lr_at_least_lr(self):
        """max_lr must be >= learning_rate for OneCycleLR."""
        assert NEXT_MAX_LR >= NEXT_LR, \
            f"MAX_LR ({NEXT_MAX_LR}) must be >= LR ({NEXT_LR})"

    def test_weight_decay_non_negative(self):
        """Weight decay must be non-negative."""
        assert NEXT_WEIGHT_DECAY >= 0


class TestTrainingConfiguration:
    """Tests for training parameters."""

    def test_batch_size_positive(self):
        """Batch size must be positive."""
        assert NEXT_BATCH_SIZE > 0

    def test_epochs_positive(self):
        """Epochs must be positive."""
        assert NEXT_EPOCHS > 0

    def test_k_folds_at_least_two(self):
        """Need at least 2 folds for cross-validation."""
        assert NEXT_K_FOLDS >= 2


class TestModelArchitectureConsistency:
    """Tests for consistency between config and model implementation."""

    def test_model_initializes_with_config(self):
        """Model must initialize successfully with config parameters."""
        model = NextHeadSingle(
            embed_dim=NEXT_INPUT_DIM,
            num_classes=NEXT_NUM_CLASSES,
            num_heads=NEXT_NUM_HEADS,
            seq_length=NEXT_MAX_SEQ_LENGTH,
            num_layers=NEXT_NUM_LAYERS,
            dropout=NEXT_DROPOUT
        )
        assert model is not None

    def test_num_layers_positive(self):
        """num_layers must be positive."""
        assert NEXT_NUM_LAYERS >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
