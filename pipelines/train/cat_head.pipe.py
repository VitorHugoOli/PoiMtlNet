"""DEPRECATED: Use scripts/train.py directly instead.

Usage:
    python scripts/train.py --task category --state california --engine poi2hgi

This wrapper will be removed in a future release.
"""
import warnings
warnings.warn(
    "pipelines/train/cat_head.pipe.py is deprecated. Use scripts/train.py directly.",
    DeprecationWarning,
    stacklevel=1,
)
import sys
import subprocess
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent.parent)
_train = str(Path(_root) / "scripts" / "train.py")


def _parse_args():
    import argparse
    sys.path.insert(0, str(Path(_root) / "src"))
    from configs.paths import EmbeddingEngine

    parser = argparse.ArgumentParser(description="Category training pipeline (thin wrapper)")
    parser.add_argument(
        "--state", type=str, nargs="+",
        default=["california"],
        help="State(s) to train on",
    )
    parser.add_argument(
        "--engine", type=str, nargs="+",
        default=["poi2hgi"],
        choices=[e.value for e in EmbeddingEngine],
        help="Embedding engine(s) to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rc = 0
    for state in args.state:
        for engine in args.engine:
            cmd = [sys.executable, _train, "--state", state, "--engine", engine, "--task", "category"]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                rc = result.returncode
    sys.exit(rc)
