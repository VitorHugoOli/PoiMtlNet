class InputsConfig:
    # EMBEDDING_DIM = 128 # DGI
    EMBEDDING_DIM = 64 # DGI
    # EMBEDDING_DIM = 128 # DGI
    # EMBEDDING_DIM = 256 # HGI
    # EMBEDDING_DIM = 50*2+7 #HMRM
    SLIDE_WINDOW = 9
    PAD_VALUE = 0
    TIMEOUT_TEST = None
    # NEXT_TARGET = 32.2
    # CATEGORY_TARGET = 47.0
    NEXT_TARGET = None
    CATEGORY_TARGET = None

    # Multi-embedding fusion configuration (None = single-embedding mode)
    FUSION_CONFIG = None  # Type: Optional[FusionConfig]

    @classmethod
    def get_category_dim(cls) -> int:
        """Get category task input dimension (single or fused)."""
        if cls.FUSION_CONFIG is None:
            return cls.EMBEDDING_DIM
        return cls.FUSION_CONFIG.get_category_dim()

    @classmethod
    def get_next_dim(cls) -> int:
        """Get next-POI task input dimension (single or fused)."""
        if cls.FUSION_CONFIG is None:
            return cls.EMBEDDING_DIM
        return cls.FUSION_CONFIG.get_next_dim()

    @classmethod
    def is_fusion_mode(cls) -> bool:
        """Check if using multi-embedding fusion."""
        return cls.FUSION_CONFIG is not None

class MTLModelConfig:
    NUM_CLASSES = 7
    BATCH_SIZE = 2**11 # 2048
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    K_FOLDS = 5

class ModelParameters:
    # Dynamic input dimensions (supports both single and fused embeddings)
    CATEGORY_INPUT_DIM = InputsConfig.get_category_dim()
    NEXT_INPUT_DIM = InputsConfig.get_next_dim()

    # Legacy support (deprecated - use CATEGORY_INPUT_DIM or NEXT_INPUT_DIM)
    INPUT_DIM = InputsConfig.EMBEDDING_DIM

    SHARED_LAYER_SIZE = 256
    NUM_HEADS = 8
    NUM_LAYERS = 4
    SEQ_LENGTH = 9
    NUM_SHARED_LAYERS = 4

    ENCODER_OUTPUT_SIZE = 256
    EXPERT_HIDDEN_SIZE = 256
    EXPERT_OUTPUT_SIZE = 256
    NUM_EXPERTS = 9
