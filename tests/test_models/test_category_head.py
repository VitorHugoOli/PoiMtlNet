"""Tests for Category Head model."""

import pytest
import torch
import torch.nn as nn

from models.heads.category import CategoryHeadMTL, CategoryHeadTransformer


class TestCategoryHeadMTL:
    """Test suite for CategoryHeadMTL."""

    def test_initialization(self):
        """Test CategoryHeadMTL initialization."""
        head = CategoryHeadMTL(input_dim=64, hidden_dim=128, num_paths=3, num_classes=7)
        assert head.num_paths == 3
        assert len(head.paths) == 3
        assert isinstance(head.combiner, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass with embedding input."""
        head = CategoryHeadMTL(input_dim=64, hidden_dim=128, num_paths=3, num_classes=7)
        head.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (8, 7)
        assert torch.isfinite(out).all()

    def test_multi_path_ensemble(self):
        """Test multi-path ensemble architecture."""
        head = CategoryHeadMTL(input_dim=64, hidden_dim=128, num_paths=3, num_classes=7)
        head.eval()
        B = 4
        x = torch.randn(B, 64)
        # Each path should produce shape (B, hidden_dim)
        with torch.no_grad():
            for path in head.paths:
                path_out = path(x)
                assert path_out.shape == (B, 128), f"Expected (B, 128), got {path_out.shape}"
        # Combined via concat of 3 paths -> (B, hidden_dim * 3)
        assert len(head.paths) == 3

    def test_output_shape(self):
        """Test output shape matches number of categories."""
        head_eval = lambda nc: CategoryHeadMTL(input_dim=64, hidden_dim=128, num_paths=3, num_classes=nc)
        x = torch.randn(4, 64)
        for num_classes in [5, 7, 10]:
            head = head_eval(num_classes)
            head.eval()
            with torch.no_grad():
                out = head(x)
            assert out.shape == (4, num_classes), f"Expected (4, {num_classes}), got {out.shape}"


class TestCategoryHeadEnhanced:
    """Test suite for enhanced category head with transformer."""

    def test_transformer_integration(self):
        """Test transformer encoder integration."""
        head = CategoryHeadTransformer(
            input_dim=64,
            num_tokens=4,
            token_dim=16,
            num_layers=2,
            num_heads=8,
            num_classes=7,
        )
        head.eval()
        x = torch.randn(8, 64)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (8, 7), f"Expected (8, 7), got {out.shape}"
        assert torch.isfinite(out).all()
