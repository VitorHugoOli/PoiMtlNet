# docs/research/ — Critical Research Assessment (2026-06-12)

Independent positioning-and-protocol audit of the project (Check2HGI substrate + MTLnet category/region MTL) against the 2016–2026 literature. Produced by a repo-evidence sweep (paper draft, claims registry, concerns/issues logs, code) plus a targeted web survey (ACM/IEEE/Springer/arXiv/OpenReview).

| Doc | What it answers |
|---|---|
| [`literature_review.md`](literature_review.md) | Who else does next-POI / next-category / next-region / location embeddings / mobility MTL, with what protocols |
| [`project_positioning.md`](project_positioning.md) | What is novel, what is incremental, what is already known; the strongest defensible framing |
| [`baseline_gap_analysis.md`](baseline_gap_analysis.md) | Baselines present vs missing; the minimum credible comparison table |
| [`evaluation_protocol_review.md`](evaluation_protocol_review.md) | Splits/metrics/leakage vs field norms; open risks (transductive substrate, selection-on-reporting-fold, comparability) |
| [`experiment_roadmap.md`](experiment_roadmap.md) | Prioritized must-have / should-have / nice-to-have experiments, mapped onto `docs/studies/closing_data/` |
| [`future_work.md`](future_work.md) | Defensible future directions (next-POI extension, inductive substrate, alignment, external validation) + guidance for the two committed changes (overlapping windows §7, second dataset §8) |
| [`mtl_frontier.md`](mtl_frontier.md) | The 2023–2026 MTL frontier (post-optimizer consensus, asymmetric architectures, output-level coupling, merging, Pareto profiling) + the ranked R1–R9 program for producing a *real* MTL gain in this study |
| [`community_insights.md`](community_insights.md) | The community-worthy insights mined from `docs/studies/` — 15 ranked findings (positive, negative, methodological) with evidence strength, plus publication vehicles |
| [`references.md`](references.md) | All references with links |

**Headline verdict** (updated 2026-06-12 after user decisions): the work supports a defensible *empirical* contribution — task-asymmetric substrate effect + MTL confound autopsy + orthogonal-gradient regime finding — and, within LBSN specifically, a defensible **domain-frontier claim**: the first rigorous post-2022-style MTL regime study in POI prediction (`mtl_frontier.md §3`). The v11 BRACIS draft is being dropped and rewritten ground-up on v14 + champion G with closing_data as the engine. Binding gaps before submission: CTLE baseline, HGI⊕raw-features control, transductivity bound, and the temporal-split/second-dataset bridge (Massive-STEPS NYC recommended). Details and priorities in the roadmap.
