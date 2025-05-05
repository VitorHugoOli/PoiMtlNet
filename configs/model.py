class InputsConfig:
    EMBEDDING_DIM = 64
    SLIDE_WINDOW = 9

class MTLModelConfig:
    NUM_CLASSES = 7
    BATCH_SIZE = 2**9 # 512
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    K_FOLDS = 5

class CategoryModelConfig:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    HIDDEN_DIMS = (512, 256, 128, 64, 32)
    NUM_CLASSES = 7
    DROPOUT = 0.5
    BATCH_SIZE = 2**9 # 512
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    K_FOLDS = 3

    N_SPLITS = 5
    SEED = 42

    LR = 0.0001
    MAX_LR = 0.01
    WEIGHT_DECAY = 1e-4
    MAX_GRAD_NORM = 1.0




class ModelParameters:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    SHARED_LAYER_SIZE = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    SEQ_LENGTH = 9
    NUM_SHARED_LAYERS = 4