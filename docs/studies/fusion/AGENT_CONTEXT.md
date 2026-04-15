# Fusion Study — Agent Context

Read this before any scientific work on the fusion-study branch. It is the
study-specific briefing extracted from the repo-wide `CLAUDE.md`.

---

## Current status

The project is in a **clean-slate ablation phase** (2026-04-13+): the prior
dataset had a labeling bug that invalidated all earlier comparisons with
published baselines. Embeddings were regenerated. **All prior results are
suspect.**

Phase P0 is complete (CBIC sanity verified). P1 starts next.

---

## Study navigation

| File | Purpose |
|------|---------|
| `docs/studies/fusion/README.md` | Entry point |
| `docs/studies/fusion/QUICK_REFERENCE.md` | One-page overview |
| `docs/studies/fusion/MASTER_PLAN.md` | 6-phase strategy (P0 prep → P1 arch×optim → P2 heads+MTL → P3 cross-embedding → P4 hparams → P5 mechanism) |
| `docs/studies/fusion/CLAIMS_AND_HYPOTHESES.md` | Authoritative claim catalog (C01..C30 + N01–N04) |
| `docs/studies/fusion/COORDINATOR.md` | Orchestrator agent spec |
| `docs/studies/fusion/phases/` | Per-phase execution plans |
| `docs/studies/fusion/state.json` | Runtime state (current phase + test statuses) |
| `docs/studies/fusion/KNOWLEDGE_SNAPSHOT.md` | Current project snapshot |
| `docs/studies/fusion/HANDOFF.md` | Session handoff notes |

Before doing any scientific work, read `QUICK_REFERENCE.md` and
`KNOWLEDGE_SNAPSHOT.md`.

---

## Skills

```
/coordinator P0    # analyze a phase, check gate, recommend next action
/worker P1         # run next planned test in P1
/study status      # show current phase + test statuses
```

---

## Key environmental facts

- **`requirements.txt` pins `scikit-learn==1.8.0`** — do not downgrade
  (1.8 fixed a `StratifiedGroupKFold(shuffle=True)` bug)
- **Torch 2.11.0** — regression test floor recalibrated to 0.88
- **MPS runs:** before long training, set
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- **Colab parallelism:** use `scripts/study/colab_runner.py` via
  `notebooks/colab_study_runner.ipynb`

---

## Issue tracker (open as of 2026-04-15)

| ID | Kind | Summary |
|----|------|---------|
| `az_fl_dgi_stale` | blocker_for_P3 | AZ/FL DGI parquets are pre-Phase-2 (missing `placeid`); folds fell back to StratifiedKFold |
| `fl_fusion_scale` | watch | Florida/fusion half-L2 ratio = 40.99× (expected 5–30×) |
