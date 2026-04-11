"""Loss registry — select loss functions by name from config.

Usage:
    from losses.registry import create_loss

    loss = create_loss("nash_mtl", n_tasks=2, device=device)
    loss = create_loss("focal", alpha=weights, gamma=2.0)

Registered losses:
    nash_mtl   — NashMTL (Nash equilibrium gradient balancing)
    equal_weight — Equal scalarization for MTL diagnostics
    static_weight — Fixed next/category scalarization
    uncertainty_weighting — Homoscedastic uncertainty weighting
    uw_so      — Soft Optimal Uncertainty weighting
    random_weight — Random loss weighting with Dirichlet samples
    rlw        — Alias for random_weight
    famo       — Fast adaptive multitask optimization-style weighting
    fairgrad   — Fair resource allocation from gradient interactions
    bayesagg_mtl — Bayesian uncertainty-inspired gradient aggregation
    go4align   — Risk-guided task alignment via interaction-aware weighting
    excess_mtl — Robust excess-risk-based dynamic weighting
    stch       — Smooth Tchebycheff scalarization
    db_mtl     — Dual-balancing style log-loss gradient weighting
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

    # Import each loss module to trigger registration
    from losses.nash_mtl import NashMTL  # noqa: F401
    from losses.equal_weight.loss import EqualWeightLoss  # noqa: F401
    from losses.static_weight.loss import StaticWeightLoss  # noqa: F401
    from losses.uncertainty_weighting.loss import UncertaintyWeightingLoss  # noqa: F401
    from losses.uw_so.loss import SoftOptimalUncertaintyWeightingLoss  # noqa: F401
    from losses.random_weight.loss import RandomWeightLoss  # noqa: F401
    from losses.famo.loss import FAMOLoss  # noqa: F401
    from losses.fairgrad.loss import FairGradLoss  # noqa: F401
    from losses.bayesagg_mtl.loss import BayesAggMTLLoss  # noqa: F401
    from losses.go4align.loss import GO4AlignLoss  # noqa: F401
    from losses.excess_mtl.loss import ExcessMTLLoss  # noqa: F401
    from losses.stch.loss import STCHLoss  # noqa: F401
    from losses.db_mtl.loss import DBMTLLoss  # noqa: F401
    from losses.focal import FocalLoss  # noqa: F401
    from losses.pcgrad import PCGrad  # noqa: F401
    from losses.gradnorm import GradNormLoss  # noqa: F401
    from losses.naive import NaiveLoss  # noqa: F401

    # Register them (since the original files don't have decorators)
    _LOSS_REGISTRY.setdefault("nash_mtl", NashMTL)
    _LOSS_REGISTRY.setdefault("equal_weight", EqualWeightLoss)
    _LOSS_REGISTRY.setdefault("static_weight", StaticWeightLoss)
    _LOSS_REGISTRY.setdefault("uncertainty_weighting", UncertaintyWeightingLoss)
    _LOSS_REGISTRY.setdefault("uw_so", SoftOptimalUncertaintyWeightingLoss)
    _LOSS_REGISTRY.setdefault("random_weight", RandomWeightLoss)
    _LOSS_REGISTRY.setdefault("rlw", RandomWeightLoss)
    _LOSS_REGISTRY.setdefault("famo", FAMOLoss)
    _LOSS_REGISTRY.setdefault("fairgrad", FairGradLoss)
    _LOSS_REGISTRY.setdefault("bayesagg_mtl", BayesAggMTLLoss)
    _LOSS_REGISTRY.setdefault("bayesagg", BayesAggMTLLoss)
    _LOSS_REGISTRY.setdefault("go4align", GO4AlignLoss)
    _LOSS_REGISTRY.setdefault("excess_mtl", ExcessMTLLoss)
    _LOSS_REGISTRY.setdefault("excessmtl", ExcessMTLLoss)
    _LOSS_REGISTRY.setdefault("stch", STCHLoss)
    _LOSS_REGISTRY.setdefault("db_mtl", DBMTLLoss)
    _LOSS_REGISTRY.setdefault("focal", FocalLoss)
    _LOSS_REGISTRY.setdefault("pcgrad", PCGrad)
    _LOSS_REGISTRY.setdefault("gradnorm", GradNormLoss)
    _LOSS_REGISTRY.setdefault("naive", NaiveLoss)
    _REGISTERED = True


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
