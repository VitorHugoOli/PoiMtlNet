"""Loss registry — select loss functions by name from config.

Usage:
    from losses.registry import create_loss

    loss = create_loss("nash_mtl", n_tasks=2, device=device)
    loss = create_loss("focal", alpha=weights, gamma=2.0)

Registered losses:
    nash_mtl   — NashMTL (Nash equilibrium gradient balancing)
    focal      — FocalLoss (class-imbalance-aware)
    pcgrad     — PCGrad (projected conflict gradients)
    gradnorm   — GradNormLoss (gradient magnitude balancing)
    naive      — NaiveLoss (dynamic alpha/beta weighted sum)

Created in Phase 4a.
"""

_LOSS_REGISTRY: dict[str, type] = {}
_REGISTERED = False


def register_loss(name: str):
    """Decorator to register a loss class under a canonical name."""
    def decorator(cls):
        if name in _LOSS_REGISTRY:
            raise ValueError(
                f"Loss '{name}' already registered by {_LOSS_REGISTRY[name].__name__}; "
                f"cannot re-register with {cls.__name__}"
            )
        _LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_registered():
    """Lazily import all loss modules so @register_loss decorators execute."""
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    # Import each criterion module to trigger @register_loss decorators
    from criterion.nash_mtl import NashMTL  # noqa: F401
    from criterion.FocalLoss import FocalLoss  # noqa: F401
    from criterion.pcgrad import PCGrad  # noqa: F401
    from criterion.gradnorm import GradNormLoss  # noqa: F401
    from criterion.NaiveLoss import NaiveLoss  # noqa: F401

    # Register them (since the original files don't have decorators)
    _LOSS_REGISTRY.setdefault("nash_mtl", NashMTL)
    _LOSS_REGISTRY.setdefault("focal", FocalLoss)
    _LOSS_REGISTRY.setdefault("pcgrad", PCGrad)
    _LOSS_REGISTRY.setdefault("gradnorm", GradNormLoss)
    _LOSS_REGISTRY.setdefault("naive", NaiveLoss)


def create_loss(name: str, **kwargs):
    """Instantiate a registered loss by name."""
    _ensure_registered()
    if name not in _LOSS_REGISTRY:
        raise KeyError(
            f"Loss '{name}' not registered. "
            f"Available: {sorted(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[name](**kwargs)


def list_losses() -> list[str]:
    """Return sorted list of registered loss names."""
    _ensure_registered()
    return sorted(_LOSS_REGISTRY.keys())
