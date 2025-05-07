class InputsConfig:
    EMBEDDING_DIM = 64 # Alvaro
    # EMBEDDING_DIM = 50*2+7 #HMRM
    SLIDE_WINDOW = 9
    PAD_VALUE = 0

class MTLModelConfig:
    NUM_CLASSES = 7
    BATCH_SIZE = 2**9 # 512
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    K_FOLDS = 5

class ModelParameters:
    INPUT_DIM = InputsConfig.EMBEDDING_DIM
    SHARED_LAYER_SIZE = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    SEQ_LENGTH = 9
    NUM_SHARED_LAYERS = 4