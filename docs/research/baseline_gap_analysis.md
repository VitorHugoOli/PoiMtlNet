# Baseline Gap Analysis

> Compiled 2026-06-12. Inventory from repo inspection (`docs/baselines/`, `docs/PAPER_BASELINES_STRATEGY.md`, `research/embeddings/`, `src/models/`, `src/losses/`); gap assessment against [`literature_review.md`](literature_review.md).

---

## 1. What the repository already has

### 1.1 External baselines (published methods)

| Baseline | Task | Fidelity | Status / numbers | Where |
|---|---|---|---|---|
| **STAN** (Luo et al., WWW 2021) | next-region | Faithful reimplementation from raw trajectories | FL Acc@10 65.36±0.69 (5-fold); AL/AZ/FL/GA done; CA/TX infeasible at scale | `src/models/next/next_stan/`, `docs/baselines/next_region/comparison.md` |
| **ReHDM** (Li et al., IJCAI 2025) | next-region | Faithful (paper protocol: chronological 80/10/10, 24h sessions, 5 seeds) | AL 66.06±0.98, AZ 54.65±0.77, FL 65.68±0.26; CA/TX deferred (compute) | `docs/baselines/next_region/comparison.md` |
| **MHA+PE** (Zeng et al., 2019) | next-category | Faithful | FL macro-F1 32.06±0.23 | `docs/baselines/next_category/comparison.md` |
| **POI-RGNN** (Capanema et al., 2021/2022) | next-category | Faithful, at user-disjoint folds (more conservative than paper) | FL 34.49±1.14, CA 31.78, TX 33.03 | `docs/baselines/POI_RGNN_AUDIT.md` |
| **PGC** (Capanema et al., 2022) | POI category *labeling* (not sequential) | Faithful | FL 40.79 | `docs/baselines/BASELINE.md` |
| **HAVANA** | POI category labeling | **Reported numbers only, no reimplementation** | FL 62.9 (paper-reported) | `docs/baselines/BASELINE.md` |
| Markov-1-region / Markov-K-cat / majority | floors | exact | Markov-1 binds at FL (~65% Acc@10, ≥85% coverage) | `docs/PAPER_BASELINES_STRATEGY.md` |
| GETNext | — | **Deliberately scoped out** — `next_stan_flow` borrows its α·log_T prior pattern but is not a reproduction; cited as inspiration only | — | `docs/PAPER_BASELINES_STRATEGY.md §Why not GETNext` |

### 1.2 Embedding substrates (internal comparison axis)

DGI, HGI (256-d, RightBank codebase), POI2HGI, Check2HGI, Time2Vec, Space2Vec, Sphere2Vec, HMRM, FUSION (`research/embeddings/*`, `src/data/inputs/fusion.py`). HGI is the substrate comparator for the headline claims (CH16/CH15), with matched-head probes (`next_gru` cat, `next_stan_flow` reg) controlling for the head.

### 1.3 MTL machinery (internal comparison axis)

13+ architectures (`src/models/mtl/*`: shared-FiLM, cross-attn/MulT, cross-stitch, MMoE, PLE-lite/CGC-lite, DSelect-k, dual-tower variants) and ~19 loss/weighting methods (`src/losses/registry.py`: NashMTL, PCGrad, GradNorm, CAGrad, Aligned-MTL, DWA, uncertainty weighting, FAMO, FairGrad, …). All adaptive-optimizer arms are documented null (`docs/studies/archive/mtl_improvement/FINAL_SYNTHESIS.md`).

### 1.4 Honest accounting of existing weaknesses

- **PLE is non-canonical** (per-task-input stacked CGC, missing the inter-level shared-gate chain) — if the paper claims a PLE comparison, this must be footnoted or fixed.
- **ReHDM-STL** (ReHDM fed frozen substrates) underperforms catastrophically (AL 26.22 vs 66.06 faithful) for architecture-bound reasons; correctly demoted to a footnote — do not present it as evidence against ReHDM.
- **HAVANA is numbers-only**; on a labeling (not sequential) task; weak as a comparison anchor.
- **ReHDM coverage is AL/AZ/FL only**; STAN coverage excludes CA/TX. The two strongest external baselines are missing precisely at the two largest states.
- **HGI-vs-Check2HGI tuning parity is not fully documented** — HGI had historical per-state tuning; an explicit tuning-budget statement is needed for the fairness claim.

---

## 2. Missing baselines, in priority order

### Tier 1 — necessary for the paper's central claims to survive review

1. **CTLE** (Lin et al., AAAI 2021, [code](https://github.com/Logan-Lin/CTLE)) — *the* contextual check-in embedding precedent. Run it as a substrate under the repo's matched heads (`next_gru` cat / `next_stan_flow` reg), same folds. Without it, "per-visit contextual embeddings carry category" cannot be attributed to the hierarchical-infomax design vs *any* contextualization. This is the single most important missing baseline.
2. **Feature-concat control**: HGI (and/or POI2Vec) embedding ⊕ raw per-visit features (category one-hot + hour/dow sin/cos) → same heads. Check2HGI's node features include exactly these signals; this control answers whether the graph/infomax machinery adds anything beyond feature injection. Cheap (no new embedding training) and decisive — its absence is the biggest internal-logic hole in CH16.
3. **Skip-gram / POI2Vec as standalone substrates** — the canonical baseline set of every location-embedding paper (CTLE's own suite: one-hot, skip-gram, POI2Vec, Geo-Teaser, TALE, Hier). POI2Vec exists in-repo as an HGI input; it must also appear as a substrate column.

### Tier 2 — strongly expected by reviewers of the MTL story

4. **HMT-GRN-style multi-task baseline** (SIGIR 2022, [code](https://github.com/poi-rec/HMT-GRN)) — the canonical region-multi-task precedent. Either adapt it to the category+region pairing or run its shared-LSTM + per-task-head design on the repo's data. Right now the MTL story has *no external MTL baseline at all* — every MTL comparison is internal.
5. **Cascaded category→region baseline** (CSLSL/CatDM pattern): predict category, condition region on it. Tests whether the cascade — the dominant published alternative to parallel MTL — beats the cross-attn design.
6. **Sequential STL baselines beyond STAN**: DeepMove and/or Flashback adapted to region targets. STAN alone is a thin sequential-baseline set by 2026 standards; Flashback is designed for sparse traces (the AL/AZ regime).

### Tier 3 — nice-to-have / robustness

7. **TALE / Geo-Teaser / CACSR** as substrates (completes the embedding-paper baseline canon).
8. **LBSN2Vec** (hypergraph check-in embedding, WWW 2019) — relevant given the hypergraph trend (STHGCN, ReHDM).
9. **An LLM zero-shot reference row** (LLM-Mob / AgentMove style) — increasingly expected in 2025/26 submissions, even if only to show the supervised models beat zero-shot prompting on these tasks.
10. **True PLE** (fix the inter-level gate chain) if any PLE claim stays in the paper.

---

## 3. Which comparisons a credible paper needs (minimum credible table)

For the **substrate claim** (C1 in [`project_positioning.md §4`](project_positioning.md)):

| Substrate column | Status |
|---|---|
| Check2HGI | ✅ |
| HGI | ✅ |
| HGI ⊕ raw visit features | ❌ Tier-1 |
| CTLE | ❌ Tier-1 |
| POI2Vec / skip-gram | ❌ Tier-1 (partially in repo) |
| DGI, Time2Vec, Space2Vec (aux) | ✅ |

For the **MTL claim** (C2/C3):

| Row | Status |
|---|---|
| STL ceilings per task (matched heads) | ✅ |
| Champion G (single joint model) | ✅ at AL/AZ/GE/FL; ❌ CA/TX (closing_data) |
| Adaptive-optimizer arms (null result) | ✅ |
| External MTL baseline (HMT-GRN-style) | ❌ Tier-2 |
| Cascade baseline (category→region) | ❌ Tier-2 |
| Markov floors | ✅ |
| External end-to-end (STAN, ReHDM) | ✅ partial state coverage |

A reviewer can currently say: *"all comparisons that matter are between the authors' own configurations."* Tier 1 + item 4 close that.
