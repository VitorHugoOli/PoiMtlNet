# docs/infra/colab/study_runner.md — Study-driven Colab runner

`scripts/study/colab_runner.py` parallelises study tests across Colab sessions. Assign `test_id`s to Colab while the primary machine handles others; each test produces a portable tarball under Drive that `/study import` can consume.

## When to use

- You have an enrolled multi-test sweep (e.g., a phase-grid in `state.json`) and want to run subsets in parallel on Colab T4 sessions while M2/M4 Pro handles others.
- You want determinism: each Colab run produces a tarball with the full run dir, importable on the primary machine for analysis.

## When NOT to use

- One-off training runs — use [`notebooks/colab_check2hgi_mtl.ipynb`](../../notebooks/colab_check2hgi_mtl.ipynb) directly.
- Embedding generation — use the embedding notebooks (`HGI.ipynb`, etc.).

## Setup (one cell, first time per session)

```python
!python /content/PoiMtlNet/scripts/study/colab_runner.py bootstrap \
    --drive-root /content/drive/MyDrive/mestrado/PoiMtlNet
```

## List planned tests in a phase

```python
!python /content/PoiMtlNet/scripts/study/colab_runner.py list --phase P1
```

## Run a single test

```python
!python /content/PoiMtlNet/scripts/study/colab_runner.py run \
    --phase P1 --test-id P1_AL_smoke
```

The tarball path prints at the end. Sync Drive, then on the primary machine:

```bash
/study import --run-dir <unpacked> --phase P1 --test-id P1_AL_smoke
```

## Companion notebook

[`notebooks/colab_study_runner.ipynb`](../../notebooks/colab_study_runner.ipynb) — wraps the above commands in cells. Use it for the multi-test parallel workflow.

## Drive layout

The runner expects:

```
<DRIVE_ROOT>/PoiMtlNet/
├── data/                 # input data (mirrored from local)
├── output/               # embedding artefacts (mirrored from local)
├── study/<phase>/        # enrolled tests' state + run-dir tarballs
└── ...
```

See [`README.md`](README.md) §Drive layout for the canonical structure.

## State file conventions

The runner consumes `state.json` from the active study folder:
- Primary check2hgi study: `docs/state.json` (if present) or follow active-study pointer.
- Active follow-up studies: `docs/studies/<study>/state.json`.
- Archived fusion study: `docs/archive/fusion-study/state.json` (read-only — historical).

The default in `scripts/study/_state.py` is `docs/archive/fusion-study/` (legacy). Override with `STUDY_DIR=...` env var when targeting a different study.
