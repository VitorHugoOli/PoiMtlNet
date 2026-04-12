"""Tests for FLOPs calculation utilities."""

import pytest
import torch
import torch.nn as nn

from utils.flops import calculate_model_flops
from utils.profiler import ModelProfiler


class TestCalculateModelFlops:
    """Test suite for FLOPs calculation."""

    def test_linear_layer_flops(self):
        """Test FLOPs calculation for Linear layer."""
        model = nn.Linear(64, 64)
        x = torch.randn(1, 64)
        result = calculate_model_flops(model, sample_input=x)
        assert result is not None
        assert isinstance(result, dict)

    def test_conv_layer_flops(self):
        """Test FLOPs calculation for Conv layer."""
        model = nn.Conv2d(3, 16, 3, padding=1)
        x = torch.randn(1, 3, 32, 32)
        result = calculate_model_flops(model, sample_input=x)
        assert result is not None
        assert isinstance(result, dict)

    def test_transformer_flops(self):
        """Test FLOPs calculation for Transformer."""
        model = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        x = torch.randn(2, 9, 64)
        result = calculate_model_flops(model, sample_input=x)
        assert result is not None
        assert isinstance(result, dict)

    def test_full_model_flops(self):
        """Test FLOPs calculation for complete MTLnet model."""
        from models.mtlnet import MTLnet
        model = MTLnet(
            feature_size=64,
            shared_layer_size=256,
            num_classes=7,
            num_heads=8,
            num_layers=4,
            seq_length=9,
            num_shared_layers=4,
        )
        cat_in = torch.randn(1, 64)
        next_in = torch.randn(1, 9, 64)
        sample = (cat_in, next_in)
        result = calculate_model_flops(model, sample_input=sample)
        assert result is not None
        assert isinstance(result, dict)


class TestModelProfiler:
    """Test suite for layer-wise model profiling."""

    def test_layer_wise_profiling(self):
        """Test layer-wise operation profiling."""
        model = nn.Linear(32, 16)
        x = torch.randn(1, 32)
        profiler = ModelProfiler(model, x)
        profiler.calculate_flops()
        assert profiler.results is not None

    def test_memory_profiling(self):
        """Test memory usage profiling."""
        model = nn.Linear(32, 16)
        x = torch.randn(1, 32)
        profiler = ModelProfiler(model, x)
        profiler.calculate_flops()
        assert isinstance(profiler.results, dict)
