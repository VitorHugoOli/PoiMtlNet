"""Tests for MTLnet multi-task learning model."""

import pytest
import torch
import torch.nn as nn

from models.mtlnet import MTLnet


def _make_model(seed: int = 0) -> MTLnet:
    """Build a small but real MTLnet for shape/equivalence tests."""
    torch.manual_seed(seed)
    return MTLnet(
        feature_size=64,
        shared_layer_size=256,
        num_classes=7,
        num_heads=8,
        num_layers=4,
        seq_length=9,
        num_shared_layers=4,
    )


class TestMTLPOIModel:
    """Test suite for MTL-POI model architecture."""

    def test_initialization(self):
        """Test model initialization with default parameters."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_forward_pass(self):
        """Test forward pass with sample input."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_task_specific_encoders(self):
        """Test task-specific encoder layers."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_shared_backbone(self):
        """Test shared backbone layers."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")


class TestMTLnetPerHeadForwards:
    """Pin the contract for cat_forward / next_forward.

    These methods exist so that scripts/evaluate.py and any future
    inference path can run a single head without needing to feed a
    dummy zero-tensor on the unused side. The contract is:

    1. Per-head methods accept the natural tensor shapes:
       - cat_forward: 2D (B, feature_size)
       - next_forward: 3D (B, seq_length, feature_size)

    2. Per-head outputs must be bit-exactly equal to the corresponding
       outputs of full forward() in eval mode (no dropout). If the
       shared subgraph ever stops being independent across the two
       heads, this test catches it.
    """

    def test_cat_forward_shape(self):
        model = _make_model()
        model.eval()
        B = 4
        cat_in = torch.randn(B, 64)
        out = model.cat_forward(cat_in)
        assert out.shape == (B, 7), f"unexpected cat_forward shape: {out.shape}"

    def test_next_forward_shape(self):
        model = _make_model()
        model.eval()
        B = 4
        next_in = torch.randn(B, 9, 64)
        out = model.next_forward(next_in)
        assert out.shape == (B, 7), f"unexpected next_forward shape: {out.shape}"

    def test_cat_forward_matches_full_forward(self):
        # cat_forward(x) must equal forward((x, anything))[0] in eval mode.
        # We pass a real next-shaped tensor to forward() and use a
        # bit-exact equality check (no atol) because the cat path is
        # supposed to be a literal subset of forward()'s computation —
        # any drift would be a refactor regression.
        model = _make_model(seed=42)
        model.eval()
        B = 8
        cat_in = torch.randn(B, 64)
        next_real = torch.randn(B, 9, 64)
        with torch.no_grad():
            full_cat, _ = model((cat_in, next_real))
            head_cat = model.cat_forward(cat_in)
        assert torch.equal(full_cat, head_cat), (
            "cat_forward output diverged from forward() — the per-head "
            "subgraph is no longer independent of the next-head path"
        )

    def test_next_forward_matches_full_forward(self):
        model = _make_model(seed=42)
        model.eval()
        B = 8
        cat_in = torch.randn(B, 64)
        next_real = torch.randn(B, 9, 64)
        with torch.no_grad():
            _, full_next = model((cat_in, next_real))
            head_next = model.next_forward(next_real)
        assert torch.equal(full_next, head_next), (
            "next_forward output diverged from forward() — the per-head "
            "subgraph is no longer independent of the category path"
        )

    def test_per_head_methods_independent_of_other_input(self):
        # Strong contract: cat_forward(x) must produce the same output
        # regardless of what next_input forward() would have received.
        # If this ever breaks, MTLnet has acquired a hidden cross-task
        # coupling and the dummy-zero refactor in scripts/evaluate.py
        # (now: per-head methods) is no longer safe.
        model = _make_model(seed=7)
        model.eval()
        B = 4
        cat_in = torch.randn(B, 64)
        next_a = torch.randn(B, 9, 64)
        next_b = torch.zeros(B, 9, 64)
        with torch.no_grad():
            full_cat_a, _ = model((cat_in, next_a))
            full_cat_b, _ = model((cat_in, next_b))
        assert torch.equal(full_cat_a, full_cat_b), (
            "MTLnet's category output depends on next input — the heads "
            "are no longer independent and per-head eval is unsafe"
        )

    def test_film_modulation(self):
        """Test FiLM (Feature-wise Linear Modulation) layers."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_residual_blocks(self):
        """Test residual block connections."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")


class TestParameterSeparation:
    """Test parameter separation for MTL optimizers."""

    def test_shared_parameters(self):
        """Test shared_parameters() method returns correct params."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_task_specific_parameters(self):
        """Test task_specific_parameters() method returns correct params."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")

    def test_no_parameter_overlap(self):
        """Test that shared and task-specific params don't overlap."""
        # TODO: Implement test
        pytest.skip("Not implemented yet")
