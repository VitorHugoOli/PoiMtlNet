# External-Baseline Triage (B1–B5) — DRAFT for P1b/P2 sign-off

> **STATUS: DRAFT, NOT FINAL.** Triage for the `baseline_gap` P1b decision (which external baselines the new
> paper's board carries). The kept set becomes rows/columns in
> [`RUN_MATRIX.md`](../closing_data/RUN_MATRIX.md) §2 and is signed off by the user at the **P2 freeze**. Grounded in
> `docs/research/baseline_gap_analysis.md` + `literature_review.md` + `project_positioning.md`,
> `docs/baselines/{BASELINE.md,README.md}`, [`PAPER_BASELINES_STRATEGY.md`](../../PAPER_BASELINES_STRATEGY.md),
> the `EmbeddingEngine` enum (`src/configs/paths.py`), `research/embeddings/`. Each candidate paper web-verified.
> Terminology held throughout: **next-category** (macro-F1) vs **next-region** (Acc@10), distinct from
> next-POI (the project deliberately has NO next-POI head). External literature baselines = SOTA-equivalent.

## Triage table

| ID | Baseline | Targets | Integration | Recommendation | Role in matrix | Tier | Comparability |
|---|---|---|---|---|---|---|---|
| **B1** | CTLE (Lin et al., AAAI 2021) | embedding/substrate | substrate-column | **INCLUDE** | The head-on novelty competitor — the only OTHER per-visit contextual location embedding in the comparison; answers "why is a hierarchical-infomax check-in substrate better than CTLE's contextual-transformer embeddings?" One cat column + one reg column. | **Tier-1** (paper-blocking) | **SC** — inherits frozen base via matched-head pipeline (same v14 slot REPLACED by CTLE 64-d vectors; same folds/seeds/windowing/labels). Drop-in, no sequence-building. **One obligation:** CTLE pre-trains on **train-portion-only per fold** (its MLM objective) → no transductive advantage; coordinate the fairness statement with pre_freeze_gates/A4. |
| **B2a** | POI2Vec (Feng et al., AAAI 2017) | embedding/substrate | substrate-column | **INCLUDE** | Static-embedding substrate floor against which BOTH contextual substrates (Check2HGI, CTLE) are measured. One cat + one reg column. | **Tier-1** | **SC**. **NET-NEW caveat:** in-repo POI2Vec is an **fclass-level HGI input feature** (`research/embeddings/hgi/poi2vec.py`), NOT a standalone per-POI substrate column. B2a funds emitting a fresh **per-POI 64-d** column. |
| **B2b** | skip-gram / word2vec over check-in seqs | embedding/substrate | substrate-column | **INCLUDE** | Completes the canonical location-embedding baseline suite (CTLE's own one-hot/skip-gram/POI2Vec/TALE canon). One cat + one reg column. | **Tier-1** | **SC**. Train **train-portion-only per fold** (parity with CTLE); 64-d. Cheap (minutes/state). |
| **B3** | HMT-GRN-style MTL (Lim et al., SIGIR 2022 / TORS 2023) | multi-task | end-to-end | **INCLUDE** | The **sole external MTL row** — calibrates "is the cross-attn dual-tower champion better than the canonical published shared-LSTM MTL design?" Today the MTL table has ZERO external MTL rows. | **Tier-2** (paper-blocking for the MTL story) | **E2E** — builds its own sequences, MUST mirror windowing/stride + user-disjoint folds + seeds + train-only priors + label spaces. **Documented deviations** ("HMT-GRN-STYLE", not faithful): beam-search/selectivity DROPPED (its regions are geohash auxiliaries to a next-POI MAIN task; here category+region are headline, no next-POI head), geohash→TIGER tract, native per-user 80/20 split → our fold regime. Blocked on windowing + freeze. |
| **B4** | Cascade category→region (CSLSL / CatDM pattern) | multi-task | either | **CONDITIONAL** (recommend SC cascade) | The dominant PUBLISHED alternative to parallel MTL — predict category, condition region on predicted category. One row testing "does the cascade beat the parallel dual-tower?" | **Tier-2** (SC variant) / Tier-3 (faithful E2E) | **EITHER.** Recommended **SC cascade** over the frozen substrate reusing `next_gru`+`next_stan_flow`/`next_lstm` (wire predicted-category into the region head) → inherits windowing/folds/labels, isolates "cascade vs parallel" cleanly. Faithful CSLSL/CatDM reimpl = **E2E** (costlier, blocked on windowing). No next-POI head. |
| **B5** | Flashback (IJCAI 2020) / DeepMove (WWW 2018) | next-region | end-to-end | **CONDITIONAL** (recommend Flashback-only) | Additional sequential next-region STL reference beyond STAN + ReHDM. Flashback targets the SPARSE-TRACE regime (AL/AZ) → the most defensible single add. | **Tier-2** (lowest marginal value — reg axis already best-covered) | **E2E** — swap POI head → region head (Acc@10), mirroring the repo's existing STAN region adaptation; keep Flashback's spatiotemporal hidden-state weighting / DeepMove's historical attention faithful; only the target space changes. Blocked on windowing + freeze. Recommend **Flashback ONLY**; DEFER DeepMove to Tier-3. |

## Summary / verdict

- **INCLUDE B1 (CTLE) and B2 (POI2Vec + skip-gram)** as **substrate columns** — both drop-in 64-d
  embeddings that inherit the frozen v14 base through the matched-head pipeline (`next_gru` cat /
  `next_stan_flow` reg), comparability guaranteed by construction, both **Tier-1 paper-blocking**. Without
  B1, the headline substrate claim ("per-visit contextual embeddings carry next-category") cannot be
  attributed to the hierarchical-infomax design vs *any* contextualization — a reviewer can dissolve the
  contribution in one line. B2 completes the canonical static-substrate floor.
- **INCLUDE B3 (HMT-GRN-style)** as the **sole external MTL row** — end-to-end, must mirror
  windowing/folds/labels, **documented as an adaptation** (regions-as-tools dropped; "HMT-GRN-STYLE"),
  mirroring the existing STAN-Flow/GETNext naming-conflict precedent in PAPER_BASELINES_STRATEGY.md.
- **CONDITIONAL B4 (cascade)** — recommend the **cheap SC cascade** reusing existing heads (clean
  "cascade vs parallel" isolation) over a costly faithful CSLSL/CatDM reimpl (Tier-3 fallback).
- **CONDITIONAL B5 (Flashback/DeepMove)** — lowest marginal value (STAN + ReHDM already anchor the
  next-region axis multi-state); recommend **Flashback-only** (sparse-trace fit for AL/AZ) or defer.
- **KEY COMPARABILITY SPLIT:** SC baselines (B1, B2a, B2b, recommended-B4) inherit the base automatically;
  **E2E baselines (B3, B5, faithful-B4)** build their own sequences and are **BLOCKED on the overlapping-
  windows ADOPT/KEEP decision + the P2 freeze**, with ONE budgeted re-run if overlap is adopted.
- **B2 net-new caveat** (load-bearing): in-repo POI2Vec is fclass-level (an HGI input), NOT a standalone
  POI-level column — B2 is genuinely net-new, a separate RUN_MATRIX decision, not "already present."
- **B1 transductive coupling:** CTLE pre-trains train-portion-only per its protocol while the Check2HGI
  substrate currently trains transductively — coordinate with pre_freeze_gates/A4 to report the asymmetry
  as a measured fairness gap (A4 RESOLVED ON NULL: downstream inflation ≈0 both axes — so the asymmetry is
  small, but the fairness statement must still be coherent).
- **Second-dataset scope:** the full B1–B5 suite is OUT of scope for the Massive-STEPS (NYC/Istanbul)
  validation (champion G + STL ceilings + Markov floor only) unless the user explicitly pulls one in.
- All five candidates target embedding / MTL / next-region; **none targets next-POI**. Decision is
  cheap and pre-freeze; runs trail the P2 freeze and fold into P3 (M1/M3).
