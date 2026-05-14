# Project Context & References

Reference documentation for the MTLnet multi-task POI prediction study. Organized by topic for use in the research paper.

## Contents

| File | Topic |
|------|-------|
| [TASKS.md](TASKS.md) | The two prediction tasks: POI category classification and next-region prediction |
| [DATASETS.md](DATASETS.md) | Per-state datasets (Gowalla US-state splits): users, check-ins, regions |
| [DATA_SPLITS.md](DATA_SPLITS.md) | Cross-validation fold protocol (StratifiedGroupKFold, MTL fold pairing, per-fold log_T leak prevention) |
| [METRICS.md](METRICS.md) | Evaluation metrics (F1, Acc@10, MRR, Δm) + paired-Wilcoxon significance testing (n=20 paper-grade vs n=5 screening) + F51 canonical extraction |
| [EMBEDDINGS.md](EMBEDDINGS.md) | All embedding engines: HGI, Check2HGI, Sphere2Vec, Time2Vec, DGI, and others |
| [FUSION.md](FUSION.md) | Multi-embedding fusion: design, motivation, task-specific combinations |
| [MTL_ARCHITECTURES.md](MTL_ARCHITECTURES.md) | MTL backbone architectures: MTLnet, CGC, MMoE, DSelectK, PLE |
| [MTL_OPTIMIZERS.md](MTL_OPTIMIZERS.md) | All MTL loss/gradient balancing methods with sources |
| [TASK_HEADS.md](TASK_HEADS.md) | Category and next-region task heads |
| [check2hgi_overview.tex](check2hgi_overview.tex) | LaTeX/TikZ figure asset documenting the Check2HGI engine architecture (POI/region/city hierarchy + check-in pipeline + losses) |
