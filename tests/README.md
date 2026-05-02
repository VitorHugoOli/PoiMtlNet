# MTLnet Test Suite

Comprehensive test suite for the MTLnet project, covering ETL, models, loss functions, utilities, and configurations.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific module tests
pytest tests/etl/ -v          # ETL tests only
pytest tests/model/ -v        # Model tests only
pytest tests/criterion/ -v    # Loss function tests only

# View coverage report
open htmlcov/index.html  # macOS
```

## Test Structure

```
tests/
├── etl/                    # ETL and input generation tests
│   ├── test_mtl_input_core.py
│   ├── test_mtl_input_builders.py
│   ├── test_mtl_input_fusion.py
│   └── test_mtl_input_checkin_conversion.py
├── criterion/              # Loss function tests
│   ├── test_nash_mtl.py
│   ├── test_focal_loss.py
│   ├── test_pcgrad.py
│   └── test_gradnorm.py
├── model/                  # Model architecture tests
│   ├── test_mtlnet.py
│   ├── test_category_head.py
│   └── test_next_head.py
├── common/                 # Utility tests
│   ├── test_ml_history.py
│   ├── test_calc_flops.py
│   └── test_training_progress.py
├── configs/                # Configuration tests
│   ├── test_model_config.py
│   └── test_paths.py
└── embeddings/             # Embedding utility tests
    └── test_embedding_utils.py
```

## Test Status

### ✅ Implemented (ETL Module)

| File | Tests | Coverage | Description |
|------|-------|----------|-------------|
| `etl/test_mtl_input_core.py` | 32 | 96% | Sequence generation, lookups, I/O |
| `etl/test_mtl_input_builders.py` | 7 | 52% | Input generation builders |
| `etl/test_mtl_input_fusion.py` | 13 | 37% | Multi-embedding fusion |
| `etl/test_mtl_input_checkin_conversion.py` | TBD | TBD | Check-in conversions |

### 📝 Stub Files Created (To Be Implemented)

| Module | Test Files | Priority |
|--------|------------|----------|
| **criterion/** | nash_mtl, focal_loss, pcgrad, gradnorm | High |
| **model/** | mtlnet, category_head, next_head | High |
| **common/** | ml_history, calc_flops, training_progress | Medium |
| **configs/** | model_config, paths | Medium |
| **embeddings/** | embedding_utils | Low |

## Running Specific Tests

```bash
# Run all tests in a module
pytest tests/etl/ -v
pytest tests/model/ -v
pytest tests/criterion/ -v

# Single test file
pytest tests/etl/test_mtl_input_core.py -v

# Single test class
pytest tests/etl/test_mtl_input_core.py::TestGenerateSequences -v

# Single test method
pytest tests/etl/test_mtl_input_core.py::TestGenerateSequences::test_empty_list_returns_empty -v

# Tests matching pattern
pytest tests/ -k "lookup" -v
pytest tests/ -k "focal" -v     # Run all focal loss tests
pytest tests/ -k "mtlnet" -v    # Run all MTLnet tests
```

## Coverage Options

```bash
# Full project coverage
pytest tests/ --cov=src --cov-report=term-missing

# Module-specific coverage
pytest tests/etl/ --cov=src/etl/mtl_input --cov-report=term-missing
pytest tests/model/ --cov=src/model --cov-report=term-missing
pytest tests/criterion/ --cov=src/criterion --cov-report=term-missing

# HTML report (detailed, interactive)
pytest tests/ --cov=src --cov-report=html

# XML report (for CI/CD)
pytest tests/ --cov=src --cov-report=xml
```

## Debugging Failed Tests

```bash
# Show full traceback
pytest tests/ -v --tb=long

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Show print statements
pytest tests/ -s
```

## Test Results Summary

### Current Status
- **Implemented Tests**: 52+ (ETL module)
- **Stub Tests**: 50+ (other modules)
- **Passed**: 50/52
- **Skipped**: 2
- **Overall Coverage**: 58% (ETL only)
- **Core Logic Coverage**: 96% ✓

See [TEST_SUMMARY.md](TEST_SUMMARY.md) for detailed results.

## What's Tested

### ✅ ETL Module (Implemented)

**Core Logic** (`etl/test_mtl_input_core.py`):
- ✓ Sequence generation with sliding windows
- ✓ Non-overlapping sequences
- ✓ Padding for short sequences
- ✓ POI → embedding lookups
- ✓ POI → category mappings
- ✓ Batch processing
- ✓ File I/O operations

**Builders** (`etl/test_mtl_input_builders.py`):
- ✓ Category input generation
- ✓ Next-POI with POI-level embeddings
- ✓ Next-POI with check-in embeddings
- ✓ Batch size configuration
- ✓ Path management via IoPaths

**Fusion** (`etl/test_mtl_input_fusion.py`):
- ✓ POI-level embedding alignment
- ✓ Multi-source embedding fusion
- ✓ Missing value handling
- ✓ Column renaming
- ✓ Concatenation order preservation

### 📝 To Be Implemented

**Criterion Module** (High Priority):
- NashMTL gradient balancing
- FocalLoss for imbalanced classes
- PCGrad gradient projection
- GradNorm magnitude balancing

**Model Module** (High Priority):
- MTLnet architecture (encoders, shared layers, FiLM)
- CategoryHeadMTL (multi-path ensemble)
- NextHeadMTL (transformer, attention pooling)
- Parameter separation for MTL

**Common Module** (Medium Priority):
- MLHistory experiment tracking
- FLOPs calculation and profiling
- Training progress utilities

**Configs Module** (Medium Priority):
- Configuration validation
- Path management and routing

**Embeddings Module** (Low Priority):
- Embedding loading and validation
- Dimension checks
- Alignment utilities

## CI/CD Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/tests.yml
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: coverage.xml
```

## Implementation Priorities

When implementing the stub tests, follow this priority order:

1. **High Priority - Core Functionality**:
   - `criterion/test_nash_mtl.py` - Primary MTL loss
   - `criterion/test_focal_loss.py` - Class imbalance handling
   - `model/test_mtlnet.py` - Main model architecture
   - `model/test_category_head.py` - Category prediction head
   - `model/test_next_head.py` - Next-POI prediction head

2. **Medium Priority - Supporting Infrastructure**:
   - `common/test_ml_history.py` - Experiment tracking
   - `common/test_calc_flops.py` - Performance profiling
   - `configs/test_paths.py` - Path management

3. **Low Priority - Optional Components**:
   - `criterion/test_pcgrad.py` - Alternative MTL method
   - `criterion/test_gradnorm.py` - Alternative MTL method
   - `embeddings/test_embedding_utils.py` - Utilities

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Aim for ≥90% coverage on new code
3. Run full test suite before committing
4. Update test documentation

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:

```bash
# Make sure you're in the project root
cd <REPO_ROOT>

# Verify pytest.ini exists
ls pytest.ini

# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"
```

### Fixture Not Found

Make sure `conftest.py` is in the `tests/` directory and properly formatted.

### Coverage Not Working

```bash
# Reinstall pytest-cov
pip uninstall pytest-cov
pip install pytest-cov

# Run with explicit source
pytest tests/ --cov=src/etl/mtl_input --cov-report=term
```

## Related Documentation

- [REFACTORING_SUMMARY.md](../REFACTORING_SUMMARY.md) - Refactoring details
- [FUSION_GUIDE.md](../FUSION_GUIDE.md) - Multi-embedding fusion
- [CLAUDE.md](../CLAUDE.md) - Project overview
