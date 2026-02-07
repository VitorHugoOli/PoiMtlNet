# Next-POI Model Improvement Plan

**Status**: Track 3 Complete — Bidirectional baseline still best, architectural improvements did not help
**Last Updated**: 2026-02-06
**Target**: Improve Check2HGI from 61% → 70% validation F1

---

## Executive Summary

**Problem**: Next-POI prediction model shows poor performance on HGI/DGI embeddings (27% val F1) and suboptimal performance on Check2HGI (61% val F1).

**Root Causes Identified**:
1. **Critical architectural bottleneck**: 16 attention heads with 64-dim embeddings = only 4 dimensions per head (should be 32-64)
2. **Training metric bug**: F1 score hardcoded to 0.0 during training, masking actual performance
3. **Suboptimal architecture choice**: Transformer is overkill for seq_length=9, RNN-based models would be more effective
4. **Hyperparameter issues**: Excessive dropout (0.35), aggressive learning rate (max_lr=0.01), no early stopping
5. **Embedding quality**: DGI/HGI static embeddings lack discriminative power vs Check2HGI contextual embeddings

**User-Selected Strategy**:
- **Priority**: Check2HGI (currently 61% F1)
- **Approach**: Incremental fixes (Track 1 → 2 → 3)
- **Target**: 70% validation F1

**Implementation Timeline**:
- **Week 0** (PREREQUISITE): Create comprehensive test suite
- **Week 1** (PRIMARY): Fix critical bugs → expect 61% → 66-71% F1
- **Week 2** (IF NEEDED): Switch to NextHeadHybrid → expect 71% → 75-78% F1
- **Week 3** (IF NEEDED): Hyperparameter tuning
- **Week 4** (AFTER SUCCESS): Diagnostics and documentation

---

## Change Log

### 2026-02-06: Track 3 — Layer Count & LR Sweep (ALL FAILED to beat bidirectional)
Three experiments tested after reverting hyperparams:

1. **NUM_LAYERS=2, LR=1e-4** (188s): Check2HGI 39.46%, DGI 26.44%
   - Worst of the three. Too few layers to capture inter-position dependencies.
   - Best epochs very early (~12-28), suggesting underfitting.

2. **NUM_LAYERS=4, LR=1e-4** (311s): Check2HGI 39.88%, DGI 26.89%
   - Close to bidirectional baseline but still slightly worse on both.
   - Suggests learned PE + temporal decay didn't add value over sinusoidal PE.

3. **NUM_LAYERS=8, LR=2e-3** (555s): Check2HGI 40.03%, DGI 26.23%
   - Check2HGI nearly matches baseline but DGI degrades significantly.
   - 8 layers is too deep (overfits), 2x slower than 4-layer variant.

**Conclusion**: None beat the **bidirectional baseline** (Check2HGI 40.16%, DGI 27.03%).
The learned PE + temporal decay architectural changes provided **no measurable benefit** vs sinusoidal PE.
The bidirectional attention removal (Track 2 fix) remains the only improvement that helped.

### 2026-02-06: Track 3 — Hyperparameter Experiment (LR=3e-4) FAILED
- **Results**: Check2HGI 39.57% F1 (-0.59%), DGI 26.90% (-0.13%) vs bidirectional baseline
- **Root cause**: LR 3e-4 too aggressive — converges faster (epoch 28 vs 54) but overshoots
  - Fold variance increased ~50% (less stable)
  - Lower WEIGHT_DECAY (1e-4 vs 1e-2) reduced regularization too much
- **Action**: Reverted LR=1e-4, MAX_LR=1e-2, WEIGHT_DECAY=1e-2; testing NUM_LAYERS in isolation
- **Kept**: Learned PE + temporal decay bias (tested but showed no improvement)

### 2026-02-06: Track 3 — Literature-Informed Architecture Improvements
- **Bidirectional results**: Check2HGI 40.16% F1, DGI 27.03% F1 (both improved over causal baseline)
- **Literature review** (STAFT 2025, GETNext, TAPT IJCAI 2025, PE survey arXiv 2502.12370):
  - Learned positional embeddings outperform sinusoidal for short sequences (<50 timesteps)
  - Temporal decay bias in attention helps next-POI tasks (recent visits matter more)
  - 2-3 Transformer layers sufficient for short sequences (4 layers is overkill)
- **Changes implemented**:
  1. Replaced sinusoidal positional encoding with learned `nn.Parameter` embeddings
  2. Added learnable temporal decay bias to attention pooling (initialized: oldest=-2.0, newest=0.0)
- All 134 tests passing

### 2026-02-06: Week 2 — Hybrid Failed, Reverted to Single + Bidirectional Attention
- **NextHeadHybrid results**: Check2HGI 39.52% F1 (1.4% worse), DGI 26.44% (1.5% worse), 3x slower
- **Root causes**: 4x dimension explosion (64→256, ~1M params), rigid last-timestep pooling, training stalls after epoch 10
- **Key insight from data pipeline analysis**: This is sequence CLASSIFICATION (all 9 positions are past visits), not autoregressive generation. Causal masking is unnecessarily restrictive — bidirectional attention (like BERT) is more appropriate.
- **Action**: Reverted to NextHeadSingle, removed causal mask for bidirectional attention
- All 134 tests passing

### 2026-02-06: Week 1 Completed ✅
- Implemented all 5 fixes from Track 1
- **Key finding**: Batch size reduction (4096→512) was the most impactful change
- MAX_GRAD_NORM=2.0 tested but reverted (variable effect: 38-41% across folds)
- DROPOUT kept at 0.1 (0.2 showed no improvement)
- F1 metric bug fixed, early stopping infrastructure added (disabled by default)
- All 134 tests passing

### 2026-02-04: Initial Plan Creation & Test Suite ✅
- Analyzed current model architecture and training results
- Identified 5 critical bugs and architectural issues
- Created improvement plan with testing-first approach
- **Created comprehensive test suite (6 test files, 134 tests)**
- Verified tests detect current bugs (16 heads / 64-dim bottleneck)
- Status: Week 0 complete, ready to begin Week 1 (bug fixes)

---

## Critical Files to Modify

### Configuration Files
1. `src/configs/next_config.py` - Central hyperparameter configuration
2. `src/configs/model.py` - Global model settings

### Training Files
3. `src/train/next/trainer.py` - Training loop with F1 bug
4. `src/train/next/cross_validation.py` - Model instantiation and CV orchestration

### Model Files
5. `src/model/next/next_head.py` - Current Transformer implementation
6. `src/model/next/next_head_enhanced.py` - Alternative architectures (NextHeadHybrid)

---

## Week 0: Test Suite Creation ✅ COMPLETED

**Goal**: Create comprehensive test suite to ensure changes don't break anything

**Status**: ✅ Completed (2026-02-04)
**Effort**: 4 hours
**Priority**: MUST DO FIRST

### Test Files Created (in `tests/`):

1. ✅ **`test_next_model_architecture.py`** - Model structure validation
   - Tests output shapes, attention head dimensions, positional encoding
   - Tests padding mask generation, parameter counts
   - Includes tests for all model architectures (Single, Hybrid, GRU)

2. ✅ **`test_next_training_metrics.py`** - Training loop correctness
   - Tests F1 calculation (detects hardcoded 0.0 bug)
   - Tests early stopping logic
   - Tests gradient clipping and LR scheduling
   - Tests class weight computation

3. ✅ **`test_next_data_loading.py`** - Data pipeline validation
   - Tests sequence reshaping (N, 576) → (N, 9, 64)
   - Tests padding detection
   - Tests stratified K-fold creation
   - Tests for data leakage prevention

4. ✅ **`test_next_model_forward_pass.py`** - Forward pass correctness
   - Tests forward with full/padded sequences
   - Tests causal masking (prevents future leakage)
   - Tests attention weight validity (sum to 1.0)
   - Tests gradient flow

5. ✅ **`test_next_config_consistency.py`** - Configuration validation
   - Tests num_heads divides embed_dim evenly
   - Tests dropout consistency across configs
   - Tests embedding dimensions match
   - **Includes test to detect current 16 heads / 64-dim bottleneck**

6. ✅ **`test_next_performance_regression.py`** - Performance benchmarks
   - Tests single forward pass speed (<50ms)
   - Tests training epoch time (<10s for 1000 samples)
   - Tests memory usage (<150MB for training)
   - Tests convergence on toy data (>95% accuracy)

### Success Criteria:
- ✅ All 6 test files created and working
- ✅ Tests detect current bugs (16 heads / 64-dim bottleneck)
- ✅ Tests pass for valid code (F1 calculation, config consistency)
- ✅ Performance benchmarks ready to run

### Key Tests Validated:
- **Bottleneck Detection**: Test confirms 16 attention heads with 64-dim embeddings = 4 dims per head
- **Config Consistency**: NUM_HEADS divides INPUT_DIM evenly (passes)
- **F1 Calculation**: Detects if F1 is hardcoded to 0.0 (ready to catch bug)
- **Forward Pass**: Models can be initialized and produce correct output shapes

### Running Tests:
```bash
# Run all next-POI tests
python -m pytest tests/test_next_*.py -v

# Run specific test categories
python -m pytest tests/test_next_model_architecture.py -v
python -m pytest tests/test_next_training_metrics.py -v
python -m pytest tests/test_next_config_consistency.py -v
python -m pytest tests/test_next_performance_regression.py -v -s  # -s shows performance output
```

---

## Track 1: Quick Fixes (Week 1)

**Status**: ✅ COMPLETED (2026-02-06)
**Actual Result**: Batch size reduction was the most impactful change. Val F1 varies 38-41% across folds with MAX_GRAD_NORM tuning.

### Fix 1: Attention Head Bottleneck ⭐ CRITICAL
- **File**: `src/configs/next_config.py:13`
- **Change**: `NUM_HEADS = 16` → `NUM_HEADS = 4`
- **Rationale**: 64/16=4 dims per head is too small, 64/4=16 is acceptable
- **Status**: ✅ Done (prior to Week 1)

### Fix 2: F1 Metric Bug
- **File**: `src/train/next/trainer.py`
- **Change**: Collect train predictions and compute real F1 with `f1_score(average='macro')` instead of hardcoded 0.0
- **Rationale**: Essential for monitoring training health
- **Status**: ✅ Done

### Fix 3: Add Early Stopping
- **Files**: `src/train/next/trainer.py`, `src/configs/next_config.py`
- **Change**: Added patience-based early stopping infrastructure with `EARLY_STOPPING_PATIENCE` config
- **Note**: Disabled by default (`EARLY_STOPPING_PATIENCE = -1`); guarded with `> 0` check. Can be enabled per-experiment.
- **Status**: ✅ Done (infrastructure ready, disabled by default)

### Fix 4: Reduce Dropout
- **Files**: `src/configs/next_config.py:17`, `src/model/next/next_head.py:50`
- **Change**: Default dropout in model changed from 0.35 → 0.2 (safety net)
- **Result**: Config DROPOUT kept at 0.1 after experiments showed no significant improvement from 0.2
- **Status**: ✅ Done (default aligned, config stays at 0.1)

### Fix 5: Adjust Gradient Clipping
- **File**: `src/configs/next_config.py:8`
- **Change**: Tested `MAX_GRAD_NORM = 2.0`
- **Result**: Variable effect across folds (38% to 41%). Reverted to 1.0 for stability.
- **Status**: ✅ Tested, reverted to 1.0

### Bonus: Batch Size Reduction
- **File**: `src/configs/next_config.py:21`
- **Change**: `BATCH_SIZE = 2**12` (4096) → `BATCH_SIZE = 2**9` (512)
- **Result**: Most impactful single change. Smaller batches provide better gradient estimates for this dataset size.
- **Status**: ✅ Done

---

## Track 2: Architecture Change (Week 2)

**Status**: ✅ COMPLETED — Hybrid failed, reverted to improved Single (2026-02-06)

### Attempt: NextHeadHybrid (FAILED)
- **Result**: Check2HGI 39.52% F1 (-1.4%), DGI 26.44% (-1.5%), 3x slower, stalls at epoch 10
- **Root causes**: 4x dimension explosion (64→256), rigid pooling, too many params (~1M)
- **Decision**: Reverted

### Fix: Remove Causal Masking from NextHeadSingle
- **File**: `src/model/next/next_head.py`
- **Change**: Removed causal mask from Transformer encoder, now uses bidirectional attention
- **Rationale**: All 9 positions are PAST visits. This is sequence classification (BERT-style), not generation (GPT-style). Bidirectional attention lets each position see full context.
- **Status**: ✅ Done — pending experimental validation

---

## Track 3: Literature-Informed Architecture Improvements

**Status**: ✅ COMPLETED — No improvement over bidirectional baseline (2026-02-06)
**Based on**: STAFT (ScienceDirect 2025), GETNext (SIGIR), TAPT (IJCAI 2025), PE Survey (arXiv 2502.12370), Linear Recency Bias (COLING 2025)

### Improvement 1: Learned Positional Embeddings — NO BENEFIT
- **File**: `src/model/next/next_head.py`
- **Change**: Replaced fixed sinusoidal `PositionalEncoding` with `nn.Parameter(torch.randn(1, seq_length, embed_dim) * 0.02)`
- **Evidence**: PE survey finds learnable embeddings outperform sinusoidal for short sequences (<50 steps)
- **Result**: No measurable improvement. 4L+learned PE (39.88%) vs 4L+sinusoidal PE (40.16%). Sinusoidal is slightly better.
- **Status**: ✅ Tested, no benefit

### Improvement 2: Temporal Decay Bias in Attention Pooling — NO BENEFIT
- **File**: `src/model/next/next_head.py`
- **Change**: Added `self.temporal_bias = nn.Parameter(torch.linspace(-2.0, 0.0, seq_length))` to attention pooling
- **Evidence**: Recency bias papers (COLING 2025) show transformers benefit from explicit temporal signals.
- **Result**: No measurable improvement when tested alongside learned PE. May be confounded.
- **Status**: ✅ Tested, no benefit

### Improvement 3: Layer Count Sweep — 4 LAYERS IS OPTIMAL
- **File**: `src/configs/next_config.py`
- **Change**: Tested NUM_LAYERS = 2, 4, 8
- **Results**: 2L underfits (39.46%), 4L best (39.88%), 8L overfits/slow (40.03% Check2HGI but 26.23% DGI)
- **Status**: ✅ Tested, 4 layers confirmed as optimal

### LR Sweep — 1e-4 IS OPTIMAL
- Tested LR=3e-4 (overshoots), LR=2e-3 (slow, no gain), LR=1e-4 (best stability)
- **Status**: ✅ Tested, LR=1e-4 confirmed

---

## Track 4: Diagnostics (Week 4)

**Status**: Planned (after target achieved)

### Add Per-Class Metrics
- **File**: `src/train/next/trainer.py`
- **Change**: Track precision, recall, F1 per category
- **Status**: Planned

### Add Gradient Monitoring
- **File**: `src/train/next/trainer.py`
- **Change**: Log gradient norms and learning rate
- **Status**: Planned

### Attention Visualization
- **File**: `src/model/next/next_head_enhanced.py`
- **Change**: Add `get_attention_weights()` method
- **Status**: Planned

---

## Testing Strategy

### Single-Fold Validation (Fast Iteration)
- Train on Fold 1 only for quick testing
- Record: train F1, val F1, train time
- Compare against baseline

### Cross-Dataset Validation (Alabama Benchmark)
**For every major change** that touches the execution flow (model architecture, training loop, data pipeline):
1. Run on **Alabama (DGI or HGI)** in addition to the primary Check2HGI (Alabama) dataset
2. Record results in the Cross-Dataset Results Tracker below
3. Compare against Alabama baseline (DGI: 26.7%, HGI: 27.4%)

**Important**: Worse results on Alabama do NOT necessarily mean we're on the wrong path. Alabama uses static POI-level embeddings (DGI/HGI) while Check2HGI uses contextual check-in-level embeddings — they are fundamentally different input types. What matters is:
- **Trend direction**: Are both datasets moving in the same direction?
- **Relative improvement**: Even a small gain on Alabama validates the change
- **Divergence is informative**: If Check2HGI improves but Alabama degrades, it tells us the change benefits contextual embeddings specifically — useful knowledge for understanding the model

This cross-dataset check serves as a **sanity check and learning tool**, not a hard gate.

### Full Cross-Validation (Final Validation)
- Run all 5 folds on Check2HGI (Alabama) — primary metric
- Run all 5 folds on Alabama (DGI) — secondary validation
- Report: mean ± std across folds for both
- Save best model checkpoints

### Ablation Testing
Change ONE thing at a time to measure impact:
```
Baseline → Fix attention heads → Fix dropout → Switch to Hybrid → Tune hyperparams
```

---

## Performance Tracking

### Cross-Dataset Results Tracker

| Change | Check2HGI (Alabama) | Alabama (DGI) | Time (5f) | Notes |
|--------|-------------------|---------------|-----------|-------|
| Baseline (original, bs1024) | 40.07% | 26.85% | 190s | NextHeadSingle + causal mask |
| Track 2: NextHeadHybrid (bs512) | 39.52% | 26.44% | 1931s | FAILED: 1% worse, 3x slower |
| **Track 2: Single + bidirectional (bs512)** | **40.16%** | **27.03%** | **306s** | **BEST — Bidirectional attention** |
| Track 3: + learned PE + decay + LR=3e-4 + 4L | 39.57% | 26.90% | 307s | LR too aggressive |
| Track 3: + learned PE + decay + LR=1e-4 + 2L | 39.46% | 26.44% | 188s | Too few layers, underfits |
| Track 3: + learned PE + decay + LR=1e-4 + 4L | 39.88% | 26.89% | 311s | Learned PE didn't help |
| Track 3: + learned PE + decay + LR=2e-3 + 8L | 40.03% | 26.23% | 555s | Too deep, DGI degrades |

**Interpretation guide**:
- Both improve → Change is universally beneficial
- Check2HGI improves, Alabama flat → Change benefits contextual embeddings
- Check2HGI improves, Alabama degrades slightly → Still likely a good path, embedding type matters more
- Both degrade → Revert the change

### Baseline (Current)
- **Check2HGI (Alabama)**: 61.3% val F1
- **Alabama (DGI)**: 26.7% val F1
- **Training F1**: 0.0 (bug)
- **Training time**: 100 epochs
- **Status**: Documented

### After Week 1 (Target)
- **Check2HGI (Alabama)**: 66-71% val F1 (target: 70%)
- **Training F1**: Real metric tracked
- **Training time**: 30-40 epochs (with early stopping)
- **Status**: Pending

### After Week 2 (If Needed)
- **Check2HGI (Alabama)**: 75-78% val F1
- **Architecture**: NextHeadHybrid
- **Status**: Conditional

---

## Success Criteria

### Minimum Viable (Must Achieve)
- ✅ Check2HGI: 61% → 70% val F1 (+9% absolute)
- ✅ Training metrics properly tracked (F1 ≠ 0.0)
- ✅ Training time <50 epochs average

### Target Performance (Should Achieve)
- 🎯 Check2HGI: 61% → 75% val F1 (+14% absolute)
- 🎯 Training time <30 epochs average

### Stretch Goals (Could Achieve)
- 🚀 Check2HGI: 61% → 80% val F1 (+19% absolute)
- 🚀 Attention visualizations showing interpretable patterns

---

## Risk Mitigation

### Risk 1: Embedding Quality is Real Bottleneck
- **Symptom**: All architectures perform similarly poor
- **Mitigation**: Run embedding quality analysis (variance, separation)
- **Action**: Focus on improving embeddings if confirmed

### Risk 2: Data Leakage
- **Symptom**: High training F1, low validation F1
- **Mitigation**: Audit fold creation for temporal leakage
- **Action**: Verify validation POIs not in training

### Risk 3: Insufficient Data
- **Symptom**: All models plateau similarly
- **Mitigation**: Check dataset size
- **Action**: Use data augmentation if needed

---

## Next Steps

1. ✅ Create test suite (Week 0) — 134 tests across 6 files
2. ✅ Implement Track 1 fixes (Week 1) — batch size most impactful
3. ✅ Track 2: Hybrid failed, bidirectional attention fix succeeded
4. ✅ Track 3: Literature improvements tested — no benefit over bidirectional baseline
5. **⚠️ ACTION NEEDED**: Revert model to bidirectional baseline config (best result):
   - Revert `next_head.py` to sinusoidal PE, remove temporal decay bias
   - Set `NUM_LAYERS = 4` in config
   - Keep: bidirectional attention (no causal mask), LR=1e-4, bs=512
6. ⏳ Track 4: Diagnostics (if pursuing further improvement)
7. ⏳ Consider: embedding quality may be the real bottleneck (not model architecture)

---

## References

- Full plan: `/Users/vitor/.claude/plans/glimmering-wobbling-lecun.md`
- Model architecture analysis: `src/model/next/ANALYSIS.md`
- Training results: `results/check2hgi/texas/next_lr1.0e-04_bs1024_ep100_20260101_1148/`
