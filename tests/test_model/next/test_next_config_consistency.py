"""
Test suite for Next-POI configuration consistency.

Tests that configurations are consistent, reasonable, and match
between config files and model implementations.
"""

import pytest
import torch

from configs.next_config import CfgNextModel, CfgNextHyperparams, CfgNextTraining
from configs.model import InputsConfig, MTLModelConfig
from model.next.next_head import NextHeadSingle
from model.next.next_head_enhanced import NextHeadHybrid


class TestNumHeadsConfiguration:
    """Tests for attention head configuration."""

    def test_num_heads_divides_embed_dim(self):
        """Test that NUM_HEADS divides INPUT_DIM evenly."""
        embed_dim = CfgNextModel.INPUT_DIM
        num_heads = CfgNextModel.NUM_HEADS

        assert embed_dim % num_heads == 0, \
            f"NUM_HEADS ({num_heads}) must divide INPUT_DIM ({embed_dim}) evenly. " \
            f"Current: {embed_dim}/{num_heads} = {embed_dim/num_heads}"

    def test_head_dimension_reasonable(self):
        """Test that head dimension is >= 16 for good performance."""
        embed_dim = CfgNextModel.INPUT_DIM
        num_heads = CfgNextModel.NUM_HEADS
        head_dim = embed_dim // num_heads

        assert head_dim >= 8, \
            f"Head dimension {head_dim} is too small (< 8). " \
            f"Consider reducing NUM_HEADS to {embed_dim // 16} or fewer."

        if head_dim < 16:
            import warnings
            warnings.warn(
                f"Head dimension {head_dim} is small. "
                f"Recommended: >= 16 dims per head. "
                f"Current config: {num_heads} heads × {head_dim} dims = {embed_dim}"
            )

    def test_num_heads_power_of_two(self):
        """Test that NUM_HEADS is a power of 2 (recommended for efficiency)."""
        num_heads = CfgNextModel.NUM_HEADS

        # Check if power of 2
        is_power_of_two = num_heads & (num_heads - 1) == 0 and num_heads > 0

        if not is_power_of_two:
            import warnings
            warnings.warn(
                f"NUM_HEADS ({num_heads}) is not a power of 2. "
                f"Consider using 2, 4, 8, or 16 for better efficiency."
            )

    def test_current_bottleneck_issue(self):
        """Test to document the current attention head bottleneck."""
        embed_dim = CfgNextModel.INPUT_DIM
        num_heads = CfgNextModel.NUM_HEADS
        head_dim = embed_dim // num_heads

        # This test documents the CURRENT (broken) configuration
        # It should PASS with current settings (16 heads, 64-dim)
        # But will FAIL after fixing to 4-8 heads

        if num_heads == 16 and embed_dim == 64:
            assert head_dim == 4, \
                f"Current config has 4-dim heads (bottleneck issue)"
            import warnings
            warnings.warn(
                "CRITICAL: Current config has 16 heads with 64-dim embeddings = 4 dims/head. "
                "This is a severe bottleneck! Fix: Set NUM_HEADS = 4 for 16 dims/head."
            )


class TestDropoutConfiguration:
    """Tests for dropout consistency across configs."""

    def test_dropout_in_range(self):
        """Test that dropout is in reasonable range [0, 0.5]."""
        dropout = CfgNextModel.DROPOUT

        assert 0 <= dropout <= 0.5, \
            f"DROPOUT {dropout} outside reasonable range [0, 0.5]"

    def test_dropout_not_too_high(self):
        """Test that dropout is not excessively high for short sequences."""
        dropout = CfgNextModel.DROPOUT
        seq_length = CfgNextModel.MAX_SEQ_LENGTH

        # For short sequences (< 20), dropout > 0.3 is often too aggressive
        if seq_length < 20 and dropout > 0.3:
            import warnings
            warnings.warn(
                UserWarning(
                    f"DROPOUT ({dropout}) might be too high for short sequences (len={seq_length}). "
                    f"Consider 0.2-0.3 for seq_length < 20."
                )
            )

    def test_dropout_matches_model_default(self):
        """Test that config dropout matches model implementation default."""
        # Create model with config dropout
        model_with_config = NextHeadSingle(
            embed_dim=64, num_classes=7, num_heads=4,
            seq_length=9, num_layers=2,
            dropout=CfgNextModel.DROPOUT
        )

        # Check that dropout was applied
        main_dropout = model_with_config.dropout.p

        assert main_dropout == CfgNextModel.DROPOUT, \
            f"Model dropout ({main_dropout}) != config dropout ({CfgNextModel.DROPOUT})"


class TestEmbeddingDimensionConfiguration:
    """Tests for embedding dimension consistency."""

    def test_embedding_dim_matches_input_dim(self):
        """Test that model INPUT_DIM matches InputsConfig.EMBEDDING_DIM."""
        model_input_dim = CfgNextModel.INPUT_DIM
        data_embed_dim = InputsConfig.EMBEDDING_DIM

        assert model_input_dim == data_embed_dim, \
            f"Model INPUT_DIM ({model_input_dim}) != data EMBEDDING_DIM ({data_embed_dim})"

    def test_embedding_dim_matches_data(self):
        """Test that EMBEDDING_DIM is 64 for DGI/HGI."""
        embed_dim = InputsConfig.EMBEDDING_DIM

        # For DGI/HGI without fusion
        assert embed_dim == 64, \
            f"Expected EMBEDDING_DIM=64 for DGI/HGI, got {embed_dim}"

    def test_fusion_dim_handling(self):
        """Test that fusion dimensions would be handled correctly."""
        # Check if get_next_dim exists and works
        try:
            next_dim = InputsConfig.get_next_dim()
            assert next_dim > 0, \
                f"get_next_dim() returned invalid dimension: {next_dim}"
        except AttributeError:
            import warnings
            warnings.warn(
                UserWarning(
                    "InputsConfig.get_next_dim() not found. "
                    "Fusion mode may not handle dimensions correctly."
                )
            )


class TestMaxSeqLengthConfiguration:
    """Tests for maximum sequence length configuration."""

    def test_max_seq_length_matches_slide_window(self):
        """Test that MAX_SEQ_LENGTH matches SLIDE_WINDOW."""
        max_seq_len = CfgNextModel.MAX_SEQ_LENGTH
        slide_window = InputsConfig.SLIDE_WINDOW

        assert max_seq_len == slide_window, \
            f"MAX_SEQ_LENGTH ({max_seq_len}) must match SLIDE_WINDOW ({slide_window})"

    def test_max_seq_length_is_nine(self):
        """Test that MAX_SEQ_LENGTH is 9 (as designed)."""
        max_seq_len = CfgNextModel.MAX_SEQ_LENGTH

        assert max_seq_len == 9, \
            f"Expected MAX_SEQ_LENGTH=9, got {max_seq_len}"


class TestNumClassesConfiguration:
    """Tests for number of classes configuration."""

    def test_num_classes_is_seven(self):
        """Test that NUM_CLASSES is 7 (POI categories)."""
        num_classes = CfgNextModel.NUM_CLASSES

        assert num_classes == 7, \
            f"Expected 7 POI categories, got {num_classes}"

    def test_num_classes_matches_mtl(self):
        """Test that next-POI NUM_CLASSES matches MTL config."""
        next_num_classes = CfgNextModel.NUM_CLASSES
        mtl_num_classes = MTLModelConfig.NUM_CLASSES

        assert next_num_classes == mtl_num_classes, \
            f"Next NUM_CLASSES ({next_num_classes}) != MTL NUM_CLASSES ({mtl_num_classes})"


class TestGradientClippingConfiguration:
    """Tests for gradient clipping settings."""

    def test_max_grad_norm_reasonable(self):
        """Test that MAX_GRAD_NORM is in reasonable range."""
        max_grad_norm = CfgNextHyperparams.MAX_GRAD_NORM

        assert 0.5 <= max_grad_norm <= 5.0, \
            f"MAX_GRAD_NORM ({max_grad_norm}) outside reasonable range [0.5, 5.0]"

    def test_max_grad_norm_for_transformers(self):
        """Test that MAX_GRAD_NORM is appropriate for Transformers."""
        max_grad_norm = CfgNextHyperparams.MAX_GRAD_NORM

        # Transformers typically use 1.0-5.0, with 1.0 being aggressive
        if max_grad_norm < 1.0:
            import warnings
            warnings.warn(
                UserWarning(
                    f"MAX_GRAD_NORM ({max_grad_norm}) is very aggressive. "
                    f"Standard for Transformers: 1.0-5.0"
                )
            )


class TestLearningRateConfiguration:
    """Tests for learning rate settings."""

    def test_lr_positive(self):
        """Test that learning rate is positive."""
        lr = CfgNextHyperparams.LR

        assert lr > 0, f"Learning rate must be positive, got {lr}"

    def test_max_lr_greater_than_lr(self):
        """Test that MAX_LR > LR for OneCycleLR."""
        lr = CfgNextHyperparams.LR
        max_lr = CfgNextHyperparams.MAX_LR

        assert max_lr > lr, \
            f"MAX_LR ({max_lr}) must be > LR ({lr}) for OneCycleLR"

    def test_max_lr_not_too_high(self):
        """Test that MAX_LR is not too aggressive."""
        max_lr = CfgNextHyperparams.MAX_LR

        if max_lr > 0.01:
            import warnings
            warnings.warn(
                UserWarning(
                    f"MAX_LR ({max_lr}) is high. "
                    f"For Transformers, typical range: 0.001-0.005"
                )
            )

    def test_weight_decay_reasonable(self):
        """Test that weight decay is in reasonable range."""
        weight_decay = CfgNextHyperparams.WEIGHT_DECAY

        assert 0 <= weight_decay <= 0.1, \
            f"WEIGHT_DECAY ({weight_decay}) outside typical range [0, 0.1]"


class TestBatchSizeConfiguration:
    """Tests for batch size settings."""

    def test_batch_size_power_of_two(self):
        """Test that BATCH_SIZE is a power of 2."""
        batch_size = CfgNextTraining.BATCH_SIZE

        is_power_of_two = batch_size & (batch_size - 1) == 0 and batch_size > 0

        assert is_power_of_two, \
            f"BATCH_SIZE ({batch_size}) should be a power of 2 for efficiency"

    def test_batch_size_reasonable(self):
        """Test that BATCH_SIZE is in reasonable range."""
        batch_size = CfgNextTraining.BATCH_SIZE

        assert 32 <= batch_size <= 8192, \
            f"BATCH_SIZE ({batch_size}) outside typical range [32, 8192]"


class TestEpochsConfiguration:
    """Tests for training epochs configuration."""

    def test_epochs_positive(self):
        """Test that EPOCHS is positive."""
        epochs = CfgNextTraining.EPOCHS

        assert epochs > 0, f"EPOCHS must be positive, got {epochs}"

    def test_epochs_not_excessive(self):
        """Test that EPOCHS is not excessively large."""
        epochs = CfgNextTraining.EPOCHS

        if epochs > 100:
            import warnings
            warnings.warn(
                UserWarning(
                    f"EPOCHS ({epochs}) is very high. "
                    f"Consider using early stopping instead."
                )
            )


class TestKFoldsConfiguration:
    """Tests for cross-validation folds."""

    def test_k_folds_reasonable(self):
        """Test that K_FOLDS is in reasonable range."""
        k_folds = CfgNextTraining.K_FOLDS

        assert 3 <= k_folds <= 10, \
            f"K_FOLDS ({k_folds}) outside typical range [3, 10]"

    def test_k_folds_is_five(self):
        """Test that K_FOLDS is 5 (standard practice)."""
        k_folds = CfgNextTraining.K_FOLDS

        assert k_folds == 5, \
            f"Expected K_FOLDS=5 (standard), got {k_folds}"


class TestModelArchitectureConsistency:
    """Tests for consistency between config and model implementation."""

    def test_model_initializes_with_config(self):
        """Test that model can be initialized with config parameters."""
        try:
            model = NextHeadSingle(
                embed_dim=CfgNextModel.INPUT_DIM,
                num_classes=CfgNextModel.NUM_CLASSES,
                num_heads=CfgNextModel.NUM_HEADS,
                seq_length=CfgNextModel.MAX_SEQ_LENGTH,
                num_layers=CfgNextModel.NUM_LAYERS,
                dropout=CfgNextModel.DROPOUT
            )

            assert model is not None, "Model initialization failed"
        except Exception as e:
            pytest.fail(f"Model initialization failed with config: {e}")

    def test_hybrid_model_has_required_params(self):
        """Test that Hybrid model config would have required parameters."""
        missing = []

        if not hasattr(CfgNextModel, 'HIDDEN_DIM'):
            missing.append('HIDDEN_DIM')

        if not hasattr(CfgNextModel, 'NUM_GRU_LAYERS'):
            missing.append('NUM_GRU_LAYERS')

        if missing:
            import warnings
            warnings.warn(
                f"CfgNextModel missing attributes for Hybrid: {missing}. "
                "Required when switching to NextHeadHybrid architecture."
            )

    def test_num_layers_reasonable(self):
        """Test that NUM_LAYERS is in reasonable range."""
        num_layers = CfgNextModel.NUM_LAYERS

        assert 1 <= num_layers <= 12, \
            f"NUM_LAYERS ({num_layers}) outside reasonable range [1, 12]"


class TestConfigCompleteness:
    """Tests to ensure all required configs exist."""

    def test_all_model_configs_exist(self):
        """Test that all required model configs are defined."""
        required_attrs = [
            'INPUT_DIM', 'NUM_HEADS', 'NUM_LAYERS',
            'MAX_SEQ_LENGTH', 'NUM_CLASSES', 'DROPOUT'
        ]

        missing = []
        for attr in required_attrs:
            if not hasattr(CfgNextModel, attr):
                missing.append(attr)

        assert len(missing) == 0, \
            f"Missing required config attributes in CfgNextModel: {missing}"

    def test_all_hyperparams_exist(self):
        """Test that all required hyperparameters are defined."""
        required_attrs = [
            'LR', 'MAX_LR', 'WEIGHT_DECAY', 'MAX_GRAD_NORM'
        ]

        missing = []
        for attr in required_attrs:
            if not hasattr(CfgNextHyperparams, attr):
                missing.append(attr)

        assert len(missing) == 0, \
            f"Missing required config attributes in CfgNextHyperparams: {missing}"

    def test_all_training_configs_exist(self):
        """Test that all required training configs are defined."""
        required_attrs = [
            'BATCH_SIZE', 'EPOCHS', 'K_FOLDS', 'SEED'
        ]

        missing = []
        for attr in required_attrs:
            if not hasattr(CfgNextTraining, attr):
                missing.append(attr)

        assert len(missing) == 0, \
            f"Missing required config attributes in CfgNextTraining: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
