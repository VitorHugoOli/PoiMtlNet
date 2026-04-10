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

