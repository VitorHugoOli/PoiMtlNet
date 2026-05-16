"""T1.3 helper — regenerate check2hgi embedding for a single state with custom α.

Args:
    --state {alabama,arizona}
    --alpha-c2p FLOAT
    --alpha-p2r FLOAT
    --alpha-r2c FLOAT

Output dir is the canonical `output/check2hgi/{state}/` (overwriting prior).
After this, the caller must rebuild next_region.parquet + per-fold log_T.
"""

import argparse
import sys
from argparse import Namespace
from copy import copy
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "research"))

from configs.paths import Resources, EmbeddingEngine
from configs.model import InputsConfig
from embeddings.check2hgi.check2hgi import create_embedding
from data.inputs.builders import generate_next_input_from_checkins

STATE_TO_SHP = {
    "alabama": ("Alabama", Resources.TL_AL),
    "arizona": ("Arizona", Resources.TL_AZ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, choices=list(STATE_TO_SHP))
    ap.add_argument("--alpha-c2p", type=float, required=True)
    ap.add_argument("--alpha-p2r", type=float, required=True)
    ap.add_argument("--alpha-r2c", type=float, required=True)
    ap.add_argument("--epoch", type=int, default=500)
    args = ap.parse_args()

    name_camel, shapefile = STATE_TO_SHP[args.state]
    cfg = Namespace(
        dim=InputsConfig.EMBEDDING_DIM,
        num_layers=2,
        attention_head=4,
        alpha_c2p=args.alpha_c2p,
        alpha_p2r=args.alpha_p2r,
        alpha_r2c=args.alpha_r2c,
        lr=0.001,
        gamma=1.0,
        max_norm=0.9,
        epoch=args.epoch,
        mini_batch_threshold=5_000_000,
        batch_size=2**13,
        num_neighbors=10,
        device='cuda',
        shapefile=shapefile,
        force_preprocess=False,  # graphs already preprocessed; only retrain
        edge_type='user_sequence',
        temporal_decay=3600.0,
        use_compile=True,
        use_amp=False,
    )
    print(f"[regen-α] state={args.state} α=({args.alpha_c2p},{args.alpha_p2r},{args.alpha_r2c}) epoch={args.epoch}", flush=True)
    create_embedding(state=name_camel, args=cfg)
    print(f"[inputs] state={args.state} regenerating next-POI inputs", flush=True)
    generate_next_input_from_checkins(name_camel, EmbeddingEngine.CHECK2HGI)
    print(f"[done] state={args.state}", flush=True)


if __name__ == "__main__":
    main()
