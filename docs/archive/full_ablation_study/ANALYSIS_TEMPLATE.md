# Stage {N} Results — {Date}

## Configuration

- Engine: fusion (128-dim)
- State: alabama
- Folds: {folds}
- Epochs: {epochs}
- Seed: 42

## Results

| Rank | Candidate | Joint Score | Next F1 | Cat F1 | Time |
|------|-----------|-------------|---------|--------|------|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |

## Observations

- Which architecture performed best?
- Which optimizer performed best?
- Was there a strong interaction between architecture and optimizer?
- How does fusion compare to HGI/DGI results from prior ablation?

## Decision

- Candidates promoted to next stage: {list}
- Rationale: {why}

## Artifacts

- Summary CSV: `results/ablations/{label}/summary.csv`
- Manifest: `results/ablations/{label}/manifest.json`
- Logs: `results/ablations/{label}/logs/`
