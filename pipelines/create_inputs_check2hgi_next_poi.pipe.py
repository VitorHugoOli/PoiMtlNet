"""Create next-POI input for the check2HGI MTL track.

Usage::

    python pipelines/create_inputs_check2hgi_next_poi.pipe.py --state alabama
    python pipelines/create_inputs_check2hgi_next_poi.pipe.py --state florida

Prerequisite: ``pipelines/embedding/check2hgi.pipe.py`` must have run
for the state. This script consumes the ``checkin_graph.pt`` +
``sequences_next.parquet`` + ``next.parquet`` triple it produces.

Output:
  - ``output/check2hgi/<state>/input/next_poi.parquet``
  - ``output/check2hgi/<state>/input/_next_poi_stats.json``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_src = str(_root / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from configs.paths import EmbeddingEngine, IoPaths
from data.inputs.next_poi import build_next_poi_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_next_poi_input(state: str) -> int:
    out_path = IoPaths.get_next_poi(state, EmbeddingEngine.CHECK2HGI)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[%s] Building next_poi frame…", state)
    df, n_pois = build_next_poi_frame(state)

    counts = df["poi_idx"].value_counts()
    top10 = counts.head(10).to_dict()
    majority_fraction = float(counts.iloc[0]) / float(len(df))

    logger.info(
        "[%s] rows=%d, poi_cardinality=%d, majority_fraction=%.6f, top10=%s",
        state,
        len(df),
        n_pois,
        majority_fraction,
        top10,
    )

    df.to_parquet(out_path)
    logger.info("[%s] Saved: %s (n_pois=%d)", state, out_path, n_pois)

    stats = {
        "state": state,
        "n_rows": int(len(df)),
        "n_pois": int(n_pois),
        "majority_fraction": majority_fraction,
        "top10_poi_idx_frequencies": {int(k): int(v) for k, v in top10.items()},
    }
    stats_path = out_path.parent / "_next_poi_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("[%s] Stats written: %s", state, stats_path)
    return n_pois


def main() -> None:
    parser = argparse.ArgumentParser(description="Build check2HGI next_poi inputs")
    parser.add_argument(
        "--state",
        type=str,
        action="append",
        default=None,
        help="State name (repeat for multiple). Defaults to ['alabama','florida'].",
    )
    args = parser.parse_args()
    states = args.state if args.state else ["alabama", "florida"]

    for state in states:
        try:
            n_pois = generate_next_poi_input(state)
            logger.info("[%s] OK (n_pois=%d)", state, n_pois)
        except Exception as exc:
            logger.error("[%s] FAILED: %s", state, exc, exc_info=True)
            raise


if __name__ == "__main__":
    main()
