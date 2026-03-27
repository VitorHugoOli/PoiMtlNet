"""Model registry — select models by name from config.

Usage:
    from models.registry import register_model, create_model

    @register_model("my_model")
    class MyModel(nn.Module): ...

    model = create_model("my_model", input_dim=64, num_classes=7)

Created in Phase 4a.
"""

from torch import nn

_MODEL_REGISTRY: dict[str, type[nn.Module]] = {}
_REGISTERED = False


def register_model(name: str):
    """Decorator to register a model class under a canonical name."""
    def decorator(cls):
        if name in _MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' already registered by {_MODEL_REGISTRY[name].__name__}; "
                f"cannot re-register with {cls.__name__}"
            )
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_registered():
    """Lazily import all model modules so @register_model decorators execute."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True
    import models.heads.category  # noqa: F401
    import models.heads.next  # noqa: F401
    import model.mtlnet.mtl_poi  # noqa: F401


def create_model(name: str, **kwargs) -> nn.Module:
    """Instantiate a registered model by name."""
    _ensure_registered()
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. "
            f"Available: {sorted(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return sorted list of registered model names."""
    _ensure_registered()
    return sorted(_MODEL_REGISTRY.keys())
