"""ExperimentConfig and RunManifest for unified experiment configuration.

ExperimentConfig is the canonical input: it defines what to run.
RunManifest is write-only output: it freezes provenance after a run.

Created in Phase 3 of the refactoring plan.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ExperimentConfig:
    """Canonical experiment configuration.

    One instance per experiment. Different task types (MTL, category, next)
    get different instances with different defaults via factory classmethods.

    Serialization: save()/load() use JSON. Enums stored as .value strings.
    """

    # --- Identification ---
    name: str
    state: str
    embedding_engine: str  # EmbeddingEngine.value string

    # --- Model ---
    model_name: str = "mtlnet"
    model_params: dict = field(default_factory=dict)

    # --- Training ---
    task_type: str = "mtl"  # "mtl", "category", "next"
    epochs: int = 50
    batch_size: int = 2048
    learning_rate: float = 1e-4
    max_lr: float = 1e-3
    weight_decay: float = 0.05
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    optimizer_eps: float = 1e-8

    # torch.compile: disabled by default.  On CUDA it uses the inductor
    # backend; MPS compatibility needs separate testing first.
    use_torch_compile: bool = False

    # --- Loss ---
    task_loss: str = "cross_entropy"
    mtl_loss: str = "nash_mtl"
    mtl_loss_params: dict = field(default_factory=dict)
    use_class_weights: bool = True

    # --- Cross-validation and split protocol ---
    k_folds: int = 5
    seed: int = 42
    split_relaxation: bool = False
    min_category_val_fraction: float = 0.05
    min_next_val_fraction: float = 0.05
    min_class_count: int = 5
    min_class_fraction: float = 0.03

    # --- Early stopping ---
    timeout: Optional[float] = None
    target_cutoff: Optional[float] = None
    early_stopping_patience: int = -1

    # --- Schema evolution ---
    schema_version: int = 1

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {self.k_folds}")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported schema_version: {self.schema_version}")

    def save(self, path) -> Path:
        """Serialize to JSON file.

        Args:
            path: File path (str or Path). Parent directory must exist.

        Returns:
            The Path written to.
        """
        path = Path(path)
        data = asdict(self)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> ExperimentConfig:
        """Deserialize from JSON file.

        Validates schema_version before constructing.

        Args:
            path: File path (str or Path).

        Returns:
            ExperimentConfig instance.

        Raises:
            ValueError: If schema_version doesn't match.
            FileNotFoundError: If path doesn't exist.
        """
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        sv = data.get("schema_version", None)
        if sv != 1:
            raise ValueError(
                f"Cannot load ExperimentConfig with schema_version={sv} "
                f"(supported: 1)"
            )
        return cls(**data)

    # ------------------------------------------------------------------
    # Factory classmethods — produce defaults matching current code exactly
    # ------------------------------------------------------------------

    @classmethod
    def default_mtl(
        cls,
        name: str,
        state: str,
        embedding_engine: str,
        **overrides,
    ) -> ExperimentConfig:
        """MTL defaults for multi-task training."""
        defaults = dict(
            name=name,
            state=state,
            embedding_engine=embedding_engine,
            model_name="mtlnet",
            model_params={
                "feature_size": 64,
                "shared_layer_size": 256,
                "num_classes": 7,
                "num_heads": 8,
                "num_layers": 4,
                "seq_length": 9,
                "num_shared_layers": 4,
            },
            task_type="mtl",
            epochs=50,
            batch_size=2**12,
            learning_rate=1e-4,
            max_lr=1e-3,
            weight_decay=0.05,
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            optimizer_eps=1e-8,
            task_loss="cross_entropy",
            mtl_loss="nash_mtl",
            mtl_loss_params={
                # max_norm=1.0 is the upstream Nash-MTL paper default. The
                # previous 2.2 was tuned against the broken-NashMTL ([1,1])
                # regime; under real Nash weighting the dgi+alabama 2-fold
                # sweep showed 1.0 wins on cat F1/Acc and matches 2.2 on
                # next-F1 within fold noise, with the tightest per-fold
                # spread on next-F1 (range 0.001 vs 0.013 for 2.2).
                "max_norm": 1.0,
                "update_weights_every": 4,
                "optim_niter": 30,
            },
            use_class_weights=True,
            k_folds=5,
            seed=42,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def default_category(
        cls,
        name: str,
        state: str,
        embedding_engine: str,
        **overrides,
    ) -> ExperimentConfig:
        """Category defaults for single-task training."""
        defaults = dict(
            name=name,
            state=state,
            embedding_engine=embedding_engine,
            model_name="category_ensemble",
            model_params={
                "input_dim": 64,
                "hidden_dim": 64,
                "num_classes": 7,
                "dropout": 0.1,
            },
            task_type="category",
            epochs=2,
            batch_size=2048,
            learning_rate=1e-4,
            max_lr=1e-2,
            weight_decay=0.05,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            optimizer_eps=1e-8,
            task_loss="cross_entropy",
            mtl_loss="",
            mtl_loss_params={},
            use_class_weights=False,
            k_folds=5,
            seed=42,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def default_next(
        cls,
        name: str,
        state: str,
        embedding_engine: str,
        **overrides,
    ) -> ExperimentConfig:
        """Next-POI defaults for single-task training."""
        defaults = dict(
            name=name,
            state=state,
            embedding_engine=embedding_engine,
            model_name="next_single",
            model_params={
                "embed_dim": 64,
                "num_classes": 7,
                "num_heads": 4,
                "seq_length": 9,
                "num_layers": 4,
                "dropout": 0.1,
            },
            task_type="next",
            epochs=100,
            batch_size=2**13,
            learning_rate=1e-4,
            max_lr=1e-2,
            weight_decay=0.01,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            optimizer_eps=1e-8,
            task_loss="cross_entropy",
            mtl_loss="",
            mtl_loss_params={},
            use_class_weights=True,
            k_folds=5,
            seed=42,
            early_stopping_patience=-1,
        )
        defaults.update(overrides)
        return cls(**defaults)


# ---------------------------------------------------------------------------
# RunManifest — write-only provenance record
# ---------------------------------------------------------------------------

@dataclass
class DatasetSignature:
    """Immutable fingerprint of a dataset file."""
    path: str
    sha256: str
    size_bytes: int
    mtime: str  # ISO 8601

    @staticmethod
    def from_path(p: Path) -> DatasetSignature:
        """Compute signature from a file on disk."""
        p = Path(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        stat = p.stat()
        return DatasetSignature(
            path=str(p.resolve().as_posix()),
            sha256=h.hexdigest(),
            size_bytes=stat.st_size,
            mtime=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        )


def _get_git_commit() -> str:
    """Return current git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class RunManifest:
    """Write-only provenance record. Serialized as manifest.json.

    Never drives training — only captures what happened.
    """
    config: ExperimentConfig
    git_commit: str
    seeds: dict
    pytorch_version: str
    device: str
    deterministic_flags: dict
    timestamp: str  # ISO 8601
    dataset_signatures: dict  # str -> DatasetSignature
    split_signature: Optional[DatasetSignature] = None
    feasibility_report_signature: Optional[DatasetSignature] = None
    schema_version: int = 1

    def write(self, output_dir: Path) -> Path:
        """Serialize as manifest.json in output_dir.

        Returns:
            Path to the written manifest file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "manifest.json"

        data = {
            "config": asdict(self.config),
            "git_commit": self.git_commit,
            "seeds": self.seeds,
            "pytorch_version": self.pytorch_version,
            "device": self.device,
            "deterministic_flags": self.deterministic_flags,
            "timestamp": self.timestamp,
            "dataset_signatures": {
                k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                for k, v in self.dataset_signatures.items()
            },
            "split_signature": (
                asdict(self.split_signature)
                if self.split_signature is not None
                else None
            ),
            "feasibility_report_signature": (
                asdict(self.feasibility_report_signature)
                if self.feasibility_report_signature is not None
                else None
            ),
            "schema_version": self.schema_version,
        }
        path.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )
        return path

    @classmethod
    def from_current_env(
        cls,
        config: ExperimentConfig,
        dataset_paths: Optional[dict[str, Path]] = None,
        split_path: Optional[Path] = None,
        feasibility_path: Optional[Path] = None,
    ) -> RunManifest:
        """Capture current environment into a manifest.

        Args:
            config: The experiment config used for this run.
            dataset_paths: Mapping of dataset name to file path.
            split_path: Path to the split manifest artifact.
            feasibility_path: Path to the feasibility report artifact.
        """
        import torch

        dataset_sigs = {}
        if dataset_paths:
            for name, p in dataset_paths.items():
                p = Path(p)
                if p.exists():
                    dataset_sigs[name] = DatasetSignature.from_path(p)

        split_sig = None
        if split_path is not None and Path(split_path).exists():
            split_sig = DatasetSignature.from_path(split_path)

        feasibility_sig = None
        if feasibility_path is not None and Path(feasibility_path).exists():
            feasibility_sig = DatasetSignature.from_path(feasibility_path)

        return cls(
            config=config,
            git_commit=_get_git_commit(),
            seeds={
                "torch_manual_seed": config.seed,
                "numpy_seed": config.seed,
                "python_seed": config.seed,
            },
            pytorch_version=torch.__version__,
            device=str(torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )),
            deterministic_flags={
                "torch.backends.cudnn.deterministic": (
                    torch.backends.cudnn.deterministic
                    if hasattr(torch.backends, "cudnn")
                    else False
                ),
                "torch.backends.cudnn.benchmark": (
                    torch.backends.cudnn.benchmark
                    if hasattr(torch.backends, "cudnn")
                    else False
                ),
            },
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            dataset_signatures=dataset_sigs,
            split_signature=split_sig,
            feasibility_report_signature=feasibility_sig,
        )
