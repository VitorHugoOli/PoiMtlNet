"""Tests for model configuration."""

import pytest
from configs.model import InputsConfig


class TestInputsConfig:
    """Test suite for InputsConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        assert InputsConfig.EMBEDDING_DIM == 64
        assert InputsConfig.SLIDE_WINDOW == 9
        assert InputsConfig.PAD_VALUE == 0

    def test_embedding_dimensions(self):
        """Test embedding dimension configurations."""
        # With default FUSION_CONFIG=None, dimensions come from EMBEDDING_DIM
        original = InputsConfig.FUSION_CONFIG
        try:
            InputsConfig.FUSION_CONFIG = None
            assert InputsConfig.get_category_dim() == 64
            assert InputsConfig.get_next_dim() == 64
            assert InputsConfig.is_fusion_mode() is False
        finally:
            InputsConfig.FUSION_CONFIG = original

    def test_slide_window_size(self):
        """Test slide window size configuration."""
        assert InputsConfig.SLIDE_WINDOW == 9
        assert InputsConfig.SLIDE_WINDOW > 0
        assert InputsConfig.SLIDE_WINDOW < 20
