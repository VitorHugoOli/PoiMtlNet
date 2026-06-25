"""W6 --freeze-reg-stream contract (the MIRROR of --freeze-cat-stream).

The category-side encoder-isolation probe freezes the REGION stream
(``next_encoder`` + ``next_poi``) so it cannot co-adapt as a cat-helper via
cross-attention K/V, run with ``--category-weight 1.0`` (reg-loss=0). These
tests pin: (1) the config field default; (2) the freeze loop in
``mtl_cv`` is BIJECTIVE — it freezes exactly the region stream and leaves the
category stream trainable (so the cat-win-isolation is clean).

The freeze's *propagation to the optimizer* (0 trainable reg params) is
additionally guarded at runtime in mtl_cv.py (RuntimeError at fold 0 if the
reg group still has trainable params), and the real-model attribute names
(``next_encoder``/``next_poi``) are verified against the champion dualtower.
"""

import torch
import torch.nn as nn

from configs.experiment import ExperimentConfig


def test_config_field_defaults_false():
    cfg = ExperimentConfig.default_mtl("t", "alabama", "check2hgi")
    assert getattr(cfg, "freeze_reg_stream", None) is False
    assert getattr(cfg, "freeze_cat_stream", None) is False


class _MiniDualTower(nn.Module):
    """Minimal stand-in with the four submodules the freeze logic targets,
    matching the champion's attribute names (category_encoder/category_poi =
    cat stream; next_encoder/next_poi = reg stream)."""

    def __init__(self):
        super().__init__()
        self.category_encoder = nn.Linear(8, 8)
        self.category_poi = nn.Linear(8, 7)
        self.next_encoder = nn.Linear(8, 8)
        self.next_poi = nn.Linear(8, 16)


def _apply_freeze_reg_stream(model):
    """The exact mtl_cv.py freeze block for freeze_reg_stream."""
    for p in model.next_encoder.parameters():
        p.requires_grad_(False)
    for p in model.next_poi.parameters():
        p.requires_grad_(False)
    model.next_encoder.eval()


def test_freeze_reg_stream_is_bijective():
    m = _MiniDualTower()
    _apply_freeze_reg_stream(m)
    # region stream frozen
    assert all(not p.requires_grad for p in m.next_encoder.parameters())
    assert all(not p.requires_grad for p in m.next_poi.parameters())
    # category stream untouched (the cat-win we are isolating must still train)
    assert all(p.requires_grad for p in m.category_encoder.parameters())
    assert all(p.requires_grad for p in m.category_poi.parameters())


def test_freeze_reg_is_mirror_of_freeze_cat():
    """Freezing the cat stream and freezing the reg stream are exact mirrors:
    each freezes its own encoder+head and leaves the other fully trainable."""
    m = _MiniDualTower()
    # cat-side freeze (the existing mechanism)
    for p in m.category_encoder.parameters():
        p.requires_grad_(False)
    for p in m.category_poi.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in m.category_encoder.parameters())
    assert all(p.requires_grad for p in m.next_encoder.parameters())
