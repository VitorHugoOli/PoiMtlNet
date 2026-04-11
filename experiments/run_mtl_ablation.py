"""Backward-compatible shim for staged MTL ablation runner.

Canonical entrypoint lives at ``scripts/run_mtl_ablation.py``.
"""

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from scripts.run_mtl_ablation import main


if __name__ == "__main__":
    main()
