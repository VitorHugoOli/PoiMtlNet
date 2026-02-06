from configs.model import InputsConfig


class CfgNextHyperparams:
    LR = 1e-4
    MAX_LR = 1e-2
    WEIGHT_DECAY = 1e-2
    MAX_GRAD_NORM = 1.0


class CfgNextModel:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    NUM_HEADS = 4
    NUM_LAYERS = 8
    MAX_SEQ_LENGTH = 9
    NUM_CLASSES = 7
    DROPOUT = 0.1
    HIDDEN_DIM = 256
    NUM_GRU_LAYERS = 2


class CfgNextTraining:
    BATCH_SIZE = 2**9  #
    EPOCHS = 100
    K_FOLDS = 5
    SEED = 42
    EARLY_STOPPING_PATIENCE = -1