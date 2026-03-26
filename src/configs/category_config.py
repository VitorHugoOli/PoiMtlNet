"""DEPRECATED: Use ExperimentConfig from configs.experiment instead.

These classes are retained for backward compatibility. They will be removed
in Phase 5 (folder tree migration). New code should use ExperimentConfig.
"""

from configs.model import InputsConfig


class CfgCategoryHyperparams:
    LR = 0.0001
    MAX_LR = 0.01
    WEIGHT_DECAY = 5e-2
    MAX_GRAD_NORM = 1.0


class CfgCategoryModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    HIDDEN_DIMS = (128, 64, 32)
    NUM_CLASSES = 7
    DROPOUT = 0.1


class CfgCategoryTraining:
    BATCH_SIZE = 2**11  # 512
    EPOCHS = 2
    K_FOLDS = 5
    SEED = 42