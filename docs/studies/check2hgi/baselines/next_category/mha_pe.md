# MHA+PE (Zeng-2019)

## Source
- **Paper:** Zeng, He, Tang, Wen. *A Next Location Predicting Approach Based on a Recurrent Neural Network and Self-attention.* IDEAL 2019, pp. 309–322. [doi:10.1007/978-3-030-30146-0_21](https://doi.org/10.1007/978-3-030-30146-0_21).
- **Reference Python impl (we ported from):** `/Users/vitor/Desktop/mestrado/nest_poi/tcc_pedro/src/model/next_pytorch_new.py::NEXT` + `src/utils/{sequences,to_int,model}.py`.
- **TS original (cross-checked):** `/Users/vitor/Desktop/mestrado/POI_Models/fork/poi_detection/`.
- **Architecture (paper §3 / reference):** Sinusoidal positional encoding added to a category-id embedding (7-d) and a hour-of-week embedding (3-d). Concatenated input fed through a single GRU(hidden=22), dropped out, then through a single-layer multi-head self-attention. The attention output is concatenated with the original spatial+temporal embeddings (with PE) per timestep, flattened across the 8-step window, and projected by a single Linear → 7-class logits.

## Why this is a baseline (not our model)
External published-method reference for the next-category task. Combines a recurrence (GRU) + self-attention head — a different inductive bias from POI-RGNN's RNN+GNN combo. We use it to:

1. Establish a **transformer-style sequence baseline** for next-category, complementing POI-RGNN's RNN+graph approach.
2. Test how much value a self-attention layer over a temporal sequence adds vs the count-based Markov-K-cat floor at the same context length (paper uses 8 steps, our Markov-K table goes up to k=9).

## What's faithful, what's adapted

### Faithful to paper / reference
- Sinusoidal positional encoding applied additively to category(7-d) and hour(3-d) embeddings (PE module identical to reference).
- GRU(in=10, hidden=22) over the concatenated [category+PE, hour+PE] sequence.
- Single-layer multi-head self-attention over the dropped-out GRU output.
- Output: concat[MHA_out, cat_emb_with_PE, hour_emb_with_PE] per timestep → flatten → Linear → n_categories.
- Hour token = `hour + 24*(weekday>=5)` (0..47).
- Skip consecutive-duplicate POIs.
- 7-class category labels (Shopping, Community, Food, Entertainment, Travel, Outdoors, Nightlife).
- Adam(lr=7e-4, betas=(0.8, 0.9)) — paper hyperparameters for Gowalla 'next' model.
- Batch 400, 11 epochs, early stop patience 3 — paper §5.4.
- Step size = 8 (paper config).
- Dropout 0.5 on GRU output and on the pre-head flattened representation.

### Adapted because our task / data differ
- **Drop user embedding.** Reference uses 2-d learned per-user embedding concatenated to GRU output before MHA, evaluated under warm-user splits (each user's sequences split independently into train/val). We evaluate under cold-user `StratifiedGroupKFold(5, seed=42)` for table-comparability with the rest of our baselines, which makes a learned user embedding random for held-out users → noise. Dropped to avoid pretending we have signal we don't (same reasoning as Faithful STAN).
- **MHA `embed_dim=22` and `num_heads=2`.** Reference has `embed_dim=24` (22 GRU + 2 user) with 4 heads (head_dim=6). Without user embedding, embed_dim=22; we reduce `num_heads` to 2 so `head_dim=11` divides cleanly. Documented divergence; minor.
- **Output LOGITS, not softmax probabilities.** Reference `forward` ends with `F.softmax(...)` — correct for Keras `CategoricalCrossentropy(from_logits=False)` but a double-softmax bug under PyTorch `nn.CrossEntropyLoss`. We output raw logits + use `nn.CrossEntropyLoss`. Confirmed correctness adaptation.
- **Window strategy.** Non-overlapping windows of size 8 + 1-step target. Reference uses overlapping prefix-expansion (`for i in range(step_size, len(events)-1): X.append(events[i-step_size:i])`). We match our in-house pipeline so cross-method comparisons stay apples-to-apples. Note: this halves the row count vs reference for the same trajectory length.
- **Loss.** `CrossEntropyLoss` over 7-class logits, no class weights — for parity with other baselines.
- **No PAD index.** Window only emitted when `n >= WINDOW_SIZE+1`, so all 8 positions are filled. Embedding tables are `Embedding(7, 7)` and `Embedding(48, 3)` exactly as reference.
- **Cross-validation protocol.** `StratifiedGroupKFold(5, seed=42)` cold-user vs reference per-user `KFold(5)` warm-user.

## Variants we run

| Variant | Inputs | Output | Where |
|---|---|---|---|
| `faithful` | raw category id + hour-of-week (8-step window) | linear → 7-class logits | `research/baselines/mha_pe/` |

## Reproduction commands

```bash
PY=/Users/vitor/Desktop/mestrado/ingred/.venv/bin/python
ENV='PYTHONPATH=src DATA_ROOT=/Users/vitor/Desktop/mestrado/ingred/data
     OUTPUT_DIR=/Users/vitor/Desktop/mestrado/ingred/output
     PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 PYTORCH_ENABLE_MPS_FALLBACK=1'

# ETL — one pass per state
caffeinate -i env $ENV "$PY" -m research.baselines.mha_pe.etl --state alabama

# Train — 5f x 11ep (paper config)
caffeinate -i env $ENV "$PY" -m research.baselines.mha_pe.train \
    --state alabama --folds 5 --epochs 11 \
    --tag FAITHFUL_MHAPE_al_5f11ep
```

Wall time: AL 5f x 11ep ~35 s, AZ ~70 s, FL ~6.5 min on M4 Pro MPS.

## Source JSONs

| Variant | State | JSON |
|---|---|---|
| `faithful` | AL | `docs/studies/check2hgi/results/baselines/faithful_mha_pe_alabama_5f_11ep_FAITHFUL_MHAPE_al_5f11ep.json` |
| `faithful` | AZ | `docs/studies/check2hgi/results/baselines/faithful_mha_pe_arizona_5f_11ep_FAITHFUL_MHAPE_az_5f11ep.json` |
| `faithful` | FL | `docs/studies/check2hgi/results/baselines/faithful_mha_pe_florida_5f_11ep_FAITHFUL_MHAPE_fl_5f11ep.json` |

## Cross-references

- Aggregated metrics by state: `results/{alabama,arizona,florida}.json`.
- Cross-baseline comparison: `comparison.md`.
- Reference implementation: `/Users/vitor/Desktop/mestrado/nest_poi/tcc_pedro/`.
