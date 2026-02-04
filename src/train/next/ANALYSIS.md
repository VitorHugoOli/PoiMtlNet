# Next-Head Training Analysis

This note analyzes the current Next-POI training (single-task “next” head) and gives targeted changes to improve convergence and macro-F1.

Context: Alabama + DGI, run: `results/dgi/alabama/next_lr1.0e-04_bs1024_ep100_20260204_1148`.

## Summary
- Macro F1 ≈ 26.85% across folds; val accuracy plateaus near 25–27%.
- Minority classes show low precision, some recall spikes (e.g., Travel/Nightlife), suggesting loss/imbalance calibration issues.
- Training loss decreases steadily; val loss flattens, then drifts → schedule/regularization likely mismatched.

## Current Setup (key points)
- Model: Transformer encoder (4 layers, 8 heads, dropout=0.1), positional encoding + attention pooling.
  - Ref: `src/model/next/next_head.py:49` (class `NextHeadSingle`).
- Inputs: short sequences (`L=9`) with DGI embeddings (`D=64`).
  - Ref: `src/configs/model.py:1`, `src/configs/next_config.py:15`.
- Loss: weighted CrossEntropy with per-class weights from train distro.
  - Ref: `src/train/next/cross_validation.py:43`.
- Optimizer/Schedule: AdamW (lr=1e-4, weight_decay=1e-2) + OneCycleLR (max_lr=1e-2).
  - Ref: `src/train/next/cross_validation.py:50`, `:57`.
- DataLoader/Imbalance: weighted sampling OFF in folds.
  - Ref: `pipelines/train/next_head.pipe.py:72` (use_weighted_sampling=False).
- Batch size: 1024; Epochs: 100; K-folds: 5.
  - Ref: `src/configs/next_config.py:22`.

## Likely Bottlenecks
1) OneCycle is too aggressive for this setup (max_lr=100× base). With L=9 and small D, the schedule can over/under-shoot; val curves flatten early.
2) Transformer depth/head count vs. sequence length. With 8 heads at D=64 (head_dim=8) and 4 layers, model may be over-parameterized for the available signal; dropout=0.1 is modest.
3) Imbalance handling trades precision for recall. Weighted CE alone can push the model to over-call some minority classes.
4) Large batch (1024) + AdamW + aggressive schedule reduces gradient noise, hindering minority learning.

## Recommendations (high impact first)
1) Stabilize optimization
- OneCycle tuning: set `LR=3e-4`, `MAX_LR=1e-3`, use `pct_start=0.1`, `div_factor=10`, `final_div_factor=100`.
- Or switch to CosineAnnealingLR with constant `lr=5e-4`.
- Reduce weight decay to `1e-4`.

2) Improve imbalance handling
- Try Focal Loss (hard-example focus): `FocalLoss(alpha=class_weights, gamma=1.5–2.0)`.
- Alternative: turn ON weighted sampling in folds and use unweighted CE. Do not combine sampler + class-weighted CE (over-correction).

3) Right-size the architecture for L=9
- If keeping Transformer: 2 layers, 4 heads, dropout=0.2–0.3.
- Or switch to GRU/LSTM baselines (often better for short sequences): `NextHeadGRU` or `NextHeadLSTM`.
- Consider `NextHeadHybrid` (GRU + attention) for robustness and interpretability.

4) Regularization and calibration
- Increase dropout to 0.2–0.3.
- If using CE, try label smoothing (e.g., 0.05).

5) Batch size and context
- Try `BATCH_SIZE=256–512`.
- If feasible, increase `SLIDE_WINDOW` to 12–15 and regenerate inputs to add context.

## Concrete Changes (by file)
- `src/configs/next_config.py` (hyperparams)
  - Set: `LR=3e-4`, `MAX_LR=1e-3`, `WEIGHT_DECAY=1e-4`.
  - Model: `NUM_LAYERS=2`, `NUM_HEADS=4`, `DROPOUT=0.2`.
  - Training: `BATCH_SIZE=512`.
- `src/train/next/cross_validation.py`
  - Loss option A (Focal):
    ```python
    from criterion.FocalLoss import FocalLoss  # top of file
    ...
    criterion = FocalLoss(alpha=alpha, gamma=1.5, reduction='mean')
    ```
  - Or CE with smoothing:
    ```python
    criterion = nn.CrossEntropyLoss(weight=alpha, label_smoothing=0.05)
    ```
  - OneCycle tuning:
    ```python
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=CfgNextHyperparams.MAX_LR,
        epochs=CfgNextTraining.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        div_factor=10,
        final_div_factor=100,
    )
    ```
  - Model swap (try GRU):
    ```python
    from model.next.next_head_enhanced import NextHeadGRU
    model = NextHeadGRU(
        embed_dim=CfgNextModel.INPUT_DIM,
        hidden_dim=256,
        num_classes=CfgNextModel.NUM_CLASSES,
        num_layers=2,
        dropout=0.3,
    ).to(DEVICE)
    ```
- `pipelines/train/next_head.pipe.py`
  - Enable weighted sampling (if using unweighted CE): set `use_weighted_sampling=True` at fold creation.

## Small Experiment Grid
- A (Transformer small): CE + smoothing 0.05; OneCycle tuned (3e-4→1e-3); WD=1e-4; heads=4; layers=2; dropout=0.2; bs=512.
- B (Transformer + Focal): Focal(gamma=1.5, alpha=weights); Cosine(lr=5e-4); WD=1e-4; heads=4; layers=2; dropout=0.3; bs=512.
- C (GRU): Weighted sampler ON + unweighted CE; Cosine(lr=5e-4); WD=1e-4; GRU(2×256, dp=0.3); bs=512.
- D (Hybrid): Focal(gamma=2, alpha=weights); OneCycle tuned; Hybrid(hid=256, heads=2–4, dp=0.3); bs=512.

## Notes from Results
- Class-wise trade-offs indicate recall-heavy bias on some minorities (e.g., Travel/Nightlife) with low precision; better calibration expected from Focal or balanced sampling.
- Flattening val metrics after ~10–20 epochs suggests schedule/regularization mismatch; gentle schedulers and slightly more dropout typically help.

## Next Steps
1) Apply hyperparam changes (A), re-run Alabama/DGI, compare macro-F1 vs. current baseline.
2) If improvement < +3–5 F1, try GRU (C) and/or Focal (B/D).
3) If compute allows, extend window to 12–15 for another pass.

If you want, I can wire a clean config switch (model type, criterion type, scheduler type, sampler toggle) to make these experiments one-line toggles.

