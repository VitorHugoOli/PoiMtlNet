from configs.model import InputsConfig


class CfgCategoryHyperparams:
    LR = 0.0001
    MAX_LR = 0.01
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0
    DROPOUT = 0.5


class CfgCategoryModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    HIDDEN_DIMS = (512, 256, 128, 64, 32)
    NUM_CLASSES = 7
    DROPOUT = 0.3


class CfgCategoryTraining:
    BATCH_SIZE = 2**9  # 512
    EPOCHS = 50
    K_FOLDS = 3
    N_SPLITS = 5
    SEED = 42