"""Ensure research variant folders exist for tracked ablation/model keys."""

from pathlib import Path

from ablation.candidates import CANDIDATES
from models.registry import list_models


def _assert_variant_files(root: Path, domain: str, variant: str, code_file: str) -> None:
    base = root / "research" / domain / variant
    assert (base / "metadata.yaml").exists(), f"Missing metadata for {domain}/{variant}"
    assert (base / "README.md").exists(), f"Missing README for {domain}/{variant}"
    assert (base / code_file).exists(), f"Missing code file for {domain}/{variant}"


def test_candidate_losses_have_research_variant_folders():
    root = Path(__file__).resolve().parents[2]
    loss_keys = sorted({candidate.mtl_loss for candidate in CANDIDATES})

    for loss_key in loss_keys:
        _assert_variant_files(root, "losses", loss_key, "loss.py")


def test_candidate_mtl_models_have_research_variant_folders():
    root = Path(__file__).resolve().parents[2]
    model_keys = sorted({candidate.model_name for candidate in CANDIDATES})

    for model_key in model_keys:
        _assert_variant_files(root, "mtl", model_key, "model.py")


def test_registered_task_heads_have_research_variant_folders():
    root = Path(__file__).resolve().parents[2]
    registered = set(list_models())
    next_keys = sorted(name for name in registered if name.startswith("next_"))
    category_keys = sorted(name for name in registered if name.startswith("category_"))

    for key in next_keys:
        _assert_variant_files(root, "next", key, "head.py")
    for key in category_keys:
        _assert_variant_files(root, "category", key, "head.py")
