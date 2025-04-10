class ModelConfig:
    NUM_CLASSES = 7
    BATCH_SIZE = 2**9 # 512
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    K_FOLDS = 5

class ModelParameters:
    INPUT_DIM = 100
    SHARED_LAYER_SIZE = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    SEQ_LENGTH = 9
    NUM_SHARED_LAYERS = 4