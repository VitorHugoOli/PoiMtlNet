from configs.model import InputsConfig


class CfgCategoryHyperparams:
    LR = 0.0001
    MAX_LR = 0.1
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0


class CfgCategoryModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    HIDDEN_DIMS = (512, 256, 128, 64, 32)
    NUM_CLASSES = 7
    DROPOUT = 0.1


class CfgCategoryTraining:
    BATCH_SIZE = 2**9  # 512
    EPOCHS = 100
    K_FOLDS = 5
    SEED = 42