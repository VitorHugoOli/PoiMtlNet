# Phase P-1 — Vanilla check2HGI embedding generation

**Gates:** none (entry phase).
**Exit gate:** `embeddings.parquet`, `poi_embeddings.parquet`, `region_embeddings.parquet` exist for both FL and AL, and training loss has flattened (not just early epochs).

## Purpose

Produce the engine artefact every downstream phase depends on. Vanilla — no temporal/spatial enrichment, no hard negatives (those are deferred per CH11).

## Runbook

```bash
# From the worktree root
DATA_ROOT=/Volumes/Vitor's\ SSD/ingred/data \
OUTPUT_DIR=/Volumes/Vitor's\ SSD/ingred/output \
"/Volumes/Vitor's SSD/ingred/.venv/bin/python" pipelines/embedding/check2hgi.pipe.py
```

States configured via the `STATES` dict at the top of the pipe script. FL + AL should be uncommented; other states commented out.

## Hyperparameters (from `pipelines/embedding/check2hgi.pipe.py`)

- `dim=64`, `num_layers=2`, `attention_head=4`
- `alpha_c2p=0.4`, `alpha_p2r=0.3`, `alpha_r2c=0.3`
- `lr=0.001`, `gamma=1.0`, `max_norm=0.9`
- `epoch=500`, `batch_size=8192`
- `edge_type='user_sequence'`, `temporal_decay=3600.0`
- Device: MPS on Apple Silicon (autodetected).

## Expected dataset stats (confirmed from interrupted run)

- **Alabama:** 113,846 check-ins; 11,848 POIs; 1,109 regions; 219,976 edges.
- **Florida:** to be confirmed once the pipeline finishes Florida.

## Data-integrity checks (before moving to P0)

1. `pd.read_parquet("output/check2hgi/{state}/embeddings.parquet")` has `len == n_checkins`.
2. `poi_embeddings.parquet` has `len == n_pois`; `region_embeddings.parquet` has `len == n_regions`.
3. Training loss curve monotone-ish (allow some noise), final loss < initial loss by ≥ 20%.
4. No NaN in any embedding column (quick `.isna().any().any()` check).

## Failure modes

- **Corrupt graph artefact** (from killed mid-write): delete `output/check2hgi/{state}/temp/` and re-run. Preprocessing is deterministic.
- **Loss plateau at > 1.35** (initial was 1.40): may indicate insufficient epochs. Extend to 1000 or reduce LR mid-training.
- **OOM on MPS:** fall back to `device='cpu'` in the pipe config — slower but reliable.

## Artifacts

On success, writes under `output/check2hgi/{state}/`:

- `embeddings.parquet` — per check-in; `[n_checkins, dim]` + metadata (userid, placeid, datetime).
- `poi_embeddings.parquet` — per POI; `[n_pois, dim]` + placeid.
- `region_embeddings.parquet` — per region; `[n_regions, dim]` + region_idx.
- `temp/checkin_graph.pt` — full preprocessing artefact (placeid → region_idx map lives here).
- `temp/sequences_next.parquet` — pre-built next-POI sequences (consumed by P0 for label derivation).

## Claims touched

None directly. P-1 is infrastructure.
