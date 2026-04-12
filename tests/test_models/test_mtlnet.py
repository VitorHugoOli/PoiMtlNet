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
        model = _make_model()
        assert hasattr(model, 'category_encoder')
        assert hasattr(model, 'next_encoder')
        assert hasattr(model, 'film')
        assert hasattr(model, 'shared_layers')
        assert hasattr(model, 'category_poi')
        assert hasattr(model, 'next_poi')
        assert model.num_classes == 7

    def test_forward_pass(self):
        """Test forward pass with sample input."""
        model = _make_model()
        model.eval()
        B = 4
        cat_in = torch.randn(B, 64)
        next_in = torch.randn(B, 9, 64)
        with torch.no_grad():
            cat_out, next_out = model((cat_in, next_in))
        assert cat_out.shape == (B, 7), f"Expected cat_out shape (4, 7), got {cat_out.shape}"
        assert next_out.shape == (B, 7), f"Expected next_out shape (4, 7), got {next_out.shape}"

    def test_task_specific_encoders(self):
        """Test task-specific encoder layers."""
        model = _make_model()
        model.eval()
        B = 4
        x = torch.randn(B, 64)
        with torch.no_grad():
            cat_enc_out = model.category_encoder(x)
            next_enc_out = model.next_encoder(x)
        # Both encoders map (B, feature_size) -> (B, shared_layer_size=256)
        assert cat_enc_out.shape == (B, 256), f"Expected (4, 256), got {cat_enc_out.shape}"
        assert next_enc_out.shape == (B, 256), f"Expected (4, 256), got {next_enc_out.shape}"

    def test_shared_backbone(self):
        """Test shared backbone layers."""
        model = _make_model()
        model.eval()
        B = 4
        x = torch.randn(B, 256)
        with torch.no_grad():
            out = model.shared_layers(x)
        # shared_layers preserves shape
        assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
        # Output should differ from input (transformation is non-trivial)
        assert not torch.equal(out, x)


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
        model = _make_model()
        model.eval()
        B = 4
        x = torch.randn(B, 256)
        with torch.no_grad():
            # Use task embedding weights directly for two different tasks
            emb_cat = model.task_embedding.weight[0].expand(B, -1)
            emb_next = model.task_embedding.weight[1].expand(B, -1)
            out_cat = model.film(x, emb_cat)
            out_next = model.film(x, emb_next)
        # FiLM outputs with different task embeddings should differ
        assert not torch.equal(out_cat, out_next), (
            "FiLM outputs should differ for different task embeddings"
        )

    def test_residual_blocks(self):
        """Test residual block connections."""
        model = _make_model()
        model.eval()
        B = 4
        x = torch.randn(B, 256)
        with torch.no_grad():
            out = model.shared_layers(x)
        # Shape is preserved
        assert out.shape == x.shape
        # Output differs from input (residual blocks apply transformations)
        assert not torch.equal(out, x)


class TestParameterSeparation:
    """Test parameter separation for MTL optimizers."""

    def test_shared_parameters(self):
        """Test shared_parameters() method returns correct params."""
        model = _make_model()
        shared_params = list(model.shared_parameters())
        assert len(shared_params) > 0
        shared_ids = {id(p) for p in shared_params}
        # Verify that parameters from shared_layers, task_embedding, and film are included
        for name, p in model.named_parameters():
            if any(key in name for key in ('shared_layers', 'task_embedding', 'film')):
                assert id(p) in shared_ids, f"Expected '{name}' in shared_parameters()"

    def test_task_specific_parameters(self):
        """Test task_specific_parameters() method returns correct params."""
        model = _make_model()
        task_params = list(model.task_specific_parameters())
        assert len(task_params) > 0
        task_ids = {id(p) for p in task_params}
        # Verify that parameters from task-specific submodules are included
        for name, p in model.named_parameters():
            if any(key in name for key in ('category_encoder', 'next_encoder', 'category_poi', 'next_poi')):
                assert id(p) in task_ids, f"Expected '{name}' in task_specific_parameters()"

    def test_no_parameter_overlap(self):
        """Test that shared and task-specific params don't overlap."""
        model = _make_model()
        shared_params = list(model.shared_parameters())
        task_params = list(model.task_specific_parameters())
        shared_ids = {id(p) for p in shared_params}
        task_ids = {id(p) for p in task_params}
        # No overlap
        overlap = shared_ids & task_ids
        assert len(overlap) == 0, f"Found {len(overlap)} parameter(s) in both shared and task-specific sets"
        # Union covers all parameters
        all_ids = {id(p) for p in model.parameters()}
        assert shared_ids | task_ids == all_ids, (
            "Union of shared and task-specific parameters does not cover all model parameters"
        )
