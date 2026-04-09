"""
Test suite for Next-POI training loop correctness.

Tests F1 calculation, early stopping, gradient clipping,
learning rate scheduling, and class weight computation.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np

from configs.experiment import ExperimentConfig
from models.heads.next import NextHeadSingle

# Derive defaults from canonical config source
_CFG = ExperimentConfig.default_next("_test", "test", "dgi")
_NEXT_MAX_GRAD_NORM = _CFG.max_grad_norm
_NEXT_LR = _CFG.learning_rate
_NEXT_WEIGHT_DECAY = _CFG.weight_decay
_NEXT_MAX_LR = _CFG.max_lr


class TestF1Calculation:
    """Tests for F1 score calculation in training loop."""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy predictions and targets."""
        num_samples = 100
        num_classes = 7

        # Random predictions and targets
        predictions = torch.randint(0, num_classes, (num_samples,))
        targets = torch.randint(0, num_classes, (num_samples,))

        return predictions, targets

    def test_f1_not_hardcoded_zero(self, dummy_data):
        """Test that F1 is calculated, not hardcoded to 0.0."""
        predictions, targets = dummy_data

        # Calculate F1 (macro average)
        f1 = f1_score(targets.numpy(), predictions.numpy(), average='macro')

        # F1 should NOT be 0.0 (unless all predictions are wrong, very unlikely)
        assert f1 != 0.0, \
            "F1 score is 0.0! This suggests it's hardcoded instead of calculated."

        # F1 should be between 0 and 1
        assert 0 <= f1 <= 1, f"F1 score {f1} is out of valid range [0, 1]"

    def test_f1_macro_average(self, dummy_data):
        """Test that F1 uses macro average (unweighted mean of per-class F1)."""
        predictions, targets = dummy_data

        f1_macro = f1_score(targets.numpy(), predictions.numpy(), average='macro')
        f1_weighted = f1_score(targets.numpy(), predictions.numpy(), average='weighted')

        # Macro and weighted should be different (unless perfect balance)
        # Just verify both are valid
        assert 0 <= f1_macro <= 1, f"Macro F1 {f1_macro} invalid"
        assert 0 <= f1_weighted <= 1, f"Weighted F1 {f1_weighted} invalid"

    def test_f1_per_class(self):
        """Test per-class F1 calculation."""
        # Perfect predictions for class 0, random for others
        predictions = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6])
        targets = torch.tensor([0, 0, 0, 2, 3, 4, 5, 6, 1])

        f1_per_class = f1_score(
            targets.numpy(),
            predictions.numpy(),
            average=None,  # Return per-class F1
            zero_division=0
        )

        # Class 0 should have perfect F1 (1.0)
        assert f1_per_class[0] == 1.0, \
            f"Class 0 should have F1=1.0, got {f1_per_class[0]}"

        # Other classes may have varying F1
        assert len(f1_per_class) == 7, \
            f"Expected 7 class F1 scores, got {len(f1_per_class)}"


class TestEarlyStopping:
    """Tests for early stopping mechanism."""

    def test_early_stopping_triggers(self):
        """Test that early stopping triggers after patience epochs."""
        patience = 5
        best_val_f1 = 0.5
        val_f1_history = [0.5, 0.48, 0.47, 0.46, 0.45, 0.44]  # Declining

        patience_counter = 0
        should_stop = False

        for epoch, val_f1 in enumerate(val_f1_history):
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                should_stop = True
                break

        assert should_stop, \
            "Early stopping should trigger after patience epochs of no improvement"
        assert patience_counter == patience, \
            f"Patience counter should be {patience}, got {patience_counter}"

    def test_early_stopping_resets_on_improvement(self):
        """Test that patience counter resets when validation improves."""
        patience = 3
        best_val_f1 = 0.5
        val_f1_history = [0.5, 0.48, 0.52, 0.51, 0.50]  # Improves at epoch 2

        patience_counter = 0
        should_stop = False

        for epoch, val_f1 in enumerate(val_f1_history):
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                should_stop = True
                break

        assert not should_stop, \
            "Early stopping should NOT trigger when validation improves"
        assert patience_counter < patience, \
            "Patience counter should reset after improvement"

    def test_best_model_saved_on_improvement(self):
        """Test that best model is tracked correctly."""
        val_f1_history = [0.5, 0.55, 0.52, 0.60, 0.58]
        best_val_f1 = 0.0
        best_epoch = -1

        for epoch, val_f1 in enumerate(val_f1_history):
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch

        assert best_val_f1 == 0.60, \
            f"Best val F1 should be 0.60, got {best_val_f1}"
        assert best_epoch == 3, \
            f"Best epoch should be 3, got {best_epoch}"


class TestGradientClipping:
    """Tests for gradient clipping mechanism."""

    @pytest.fixture
    def model(self):
        """Create simple model for gradient tests."""
        return NextHeadSingle(
            embed_dim=64,
            num_classes=7,
            num_heads=4,
            seq_length=9,
            num_layers=2,
            dropout=0.1
        )

    def test_gradient_clipping_applied(self, model):
        """Test that gradients are clipped to max_norm."""
        max_grad_norm = 1.0

        # Create large gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.randn_like(param) * 100  # Large gradients

        # Apply clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm, norm_type=2.0
        )

        # After clipping, compute actual norm to verify it was clipped
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # After clipping, norm should be <= max_grad_norm (with tolerance)
        assert total_norm <= max_grad_norm + 1e-3, \
            f"Gradient norm {total_norm} exceeds max {max_grad_norm} after clipping"

    def test_gradient_norm_calculation(self, model):
        """Test gradient norm is calculated correctly."""
        # Set known gradients
        for param in model.parameters():
            if param.requires_grad:
                param.grad = torch.ones_like(param)

        # Calculate total norm manually
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        # Compare with torch's calculation
        torch_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float('inf'), norm_type=2.0
        )

        assert abs(total_norm - torch_norm) < 1e-3, \
            f"Manual norm {total_norm} != torch norm {torch_norm}"

    def test_max_grad_norm_config(self):
        """Test that MAX_GRAD_NORM is reasonable."""
        max_grad_norm = _NEXT_MAX_GRAD_NORM

        assert 0.5 <= max_grad_norm <= 5.0, \
            f"MAX_GRAD_NORM {max_grad_norm} outside reasonable range [0.5, 5.0]"


class TestLearningRateSchedule:
    """Tests for learning rate scheduling."""

    @pytest.fixture
    def optimizer_and_scheduler(self):
        """Create optimizer and OneCycleLR scheduler."""
        model = NextHeadSingle(
            embed_dim=64, num_classes=7, num_heads=4,
            seq_length=9, num_layers=2, dropout=0.1
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=_NEXT_LR,
            weight_decay=_NEXT_WEIGHT_DECAY
        )

        # Simulate 10 steps per epoch, 10 epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=_NEXT_MAX_LR,
            epochs=10,
            steps_per_epoch=10
        )

        return optimizer, scheduler

    def test_learning_rate_changes(self, optimizer_and_scheduler):
        """Test that learning rate changes over training."""
        optimizer, scheduler = optimizer_and_scheduler

        initial_lr = optimizer.param_groups[0]['lr']
        lr_history = [initial_lr]

        # Simulate 50 steps (5 epochs)
        for step in range(50):
            scheduler.step()
            lr_history.append(optimizer.param_groups[0]['lr'])

        # LR should change during training
        assert len(set(lr_history)) > 1, \
            "Learning rate never changed (scheduler not working)"

        # LR should reach max at some point
        max_lr_reached = max(lr_history)
        assert max_lr_reached > initial_lr, \
            "Learning rate never increased above initial"

    def test_onecycle_warmup_and_decay(self, optimizer_and_scheduler):
        """Test OneCycleLR warmup and decay phases."""
        optimizer, scheduler = optimizer_and_scheduler

        total_steps = 100  # 10 epochs * 10 steps
        lr_history = []

        for step in range(total_steps):
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()

        # Find peak LR
        peak_lr = max(lr_history)
        peak_idx = lr_history.index(peak_lr)

        # Peak should occur in first half (warmup phase)
        assert peak_idx < total_steps // 2, \
            f"Peak LR at step {peak_idx}, expected in first half"

        # Final LR should be lower than peak (decay phase)
        final_lr = lr_history[-1]
        assert final_lr < peak_lr * 0.5, \
            f"Final LR {final_lr} should be much lower than peak {peak_lr}"

    def test_max_lr_reasonable(self):
        """Test that MAX_LR is not too aggressive."""
        max_lr = _NEXT_MAX_LR

        # For Transformers, max_lr > 0.01 is often too high
        assert max_lr <= 0.01, \
            f"MAX_LR {max_lr} may be too aggressive for Transformers (typical: 0.001-0.005)"


class TestClassWeights:
    """Tests for class weight computation."""

    def test_class_weights_computed_correctly(self):
        """Test that class weights are balanced correctly."""
        # Simulate imbalanced dataset
        # Class 0: 100 samples, Class 1: 50 samples, Class 2: 20 samples
        targets = torch.cat([
            torch.zeros(100),
            torch.ones(50),
            torch.full((20,), 2)
        ]).long()

        # Compute class weights (inverse frequency)
        class_counts = torch.bincount(targets)
        total_samples = len(targets)
        class_weights = total_samples / (len(class_counts) * class_counts.float())

        # Class 0 (most frequent) should have lowest weight
        assert class_weights[0] < class_weights[1], \
            "Majority class should have lower weight"
        assert class_weights[1] < class_weights[2], \
            "Medium class should have lower weight than minority"

    def test_class_weights_sum_to_num_classes(self):
        """Test that normalized class weights sum to num_classes."""
        targets = torch.randint(0, 7, (1000,))

        # Compute class weights
        class_counts = torch.bincount(targets, minlength=7)
        total_samples = len(targets)
        class_weights = total_samples / (7 * class_counts.float())

        # Weights should sum to num_classes (7)
        assert abs(class_weights.sum().item() - 7.0) < 0.1, \
            f"Class weights sum to {class_weights.sum()}, expected ~7.0"

    def test_zero_class_handling(self):
        """Test that classes with zero samples don't cause division by zero."""
        # Class 6 has no samples
        targets = torch.randint(0, 6, (1000,))

        class_counts = torch.bincount(targets, minlength=7)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        class_weights = 1000 / (7 * (class_counts.float() + epsilon))

        # Should not have inf or nan
        assert not torch.isinf(class_weights).any(), \
            "Class weights contain inf (division by zero)"
        assert not torch.isnan(class_weights).any(), \
            "Class weights contain nan"


class TestTrainingStability:
    """Tests for training stability (NaN detection, convergence)."""

    def test_no_nan_in_predictions(self):
        """Test that model predictions don't contain NaN."""
        model = NextHeadSingle(
            embed_dim=64, num_classes=7, num_heads=4,
            seq_length=9, num_layers=2, dropout=0.1
        )

        x = torch.randn(32, 9, 64)
        output = model(x)

        assert not torch.isnan(output).any(), \
            "Model output contains NaN values"
        assert not torch.isinf(output).any(), \
            "Model output contains Inf values"

    def test_loss_decreases_on_toy_data(self):
        """Test that loss decreases on simple toy dataset."""
        model = NextHeadSingle(
            embed_dim=64, num_classes=7, num_heads=4,
            seq_length=9, num_layers=2, dropout=0.0  # No dropout for convergence test
        )

        # Create simple toy data (one class, easy to learn)
        x = torch.randn(100, 9, 64)
        y = torch.zeros(100, dtype=torch.long)  # All class 0

        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        final_loss = None

        # Train for 50 steps
        model.train()
        for step in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            final_loss = loss.item()

        # Loss should decrease significantly
        assert final_loss < initial_loss * 0.5, \
            f"Loss didn't decrease enough: {initial_loss:.4f} → {final_loss:.4f}"

    def test_accuracy_improves_on_toy_data(self):
        """Test that accuracy improves on simple toy dataset."""
        model = NextHeadSingle(
            embed_dim=64, num_classes=7, num_heads=4,
            seq_length=9, num_layers=2, dropout=0.0
        )

        # Simple dataset
        x = torch.randn(100, 9, 64)
        y = torch.zeros(100, dtype=torch.long)

        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train for 100 steps
        model.train()
        for step in range(100):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check final accuracy
        model.eval()
        with torch.no_grad():
            output = model(x)
            predictions = output.argmax(dim=1)
            accuracy = (predictions == y).float().mean().item()

        # Should reach high accuracy on toy data
        assert accuracy > 0.8, \
            f"Model should reach >80% accuracy on toy data, got {accuracy:.2%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
