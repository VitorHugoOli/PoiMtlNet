# docs/studies/archive/ — closed check2hgi follow-up studies

Fully-closed studies layered on the primary check2hgi study, moved here 2026-06-14 to declutter the
active set in [`docs/studies/`](../). **Archiving is organizational, not a deprecation:** these findings
are authoritative and are still heavily cross-referenced by the active pre-freeze family, `closing_data`,
the paper drafts, and the canonical docs (`CLAUDE.md`, `docs/results/CANONICAL_VERSIONS.md`, …). Inbound
links were rewritten from `studies/<study>` → `studies/archive/<study>` at move time.

> This is distinct from the repo-level [`docs/archive/`](../../archive/), which holds *non-study* material
> (old reorg plans, the earlier **fusion-study**, pre-B3 framing snapshots).

| Study | Closed | Read first | Headline outcome |
|---|---|---|---|
| [`mtl_improvement/`](mtl_improvement/) | 2026-06-12 | [`FINAL_SYNTHESIS.md`](mtl_improvement/FINAL_SYNTHESIS.md) | The C25 class-weighting confound *was* the "MTL sacrifices region" gap. Champion **G (= canon v16, the `train.py --task mtl` default)** matches the STL reg ceiling (Δ −0.09…−0.31) and beats the STL cat ceiling (+2.6…+4.1) at 4 states × 4 seeds. Mechanism = gradient orthogonality (cos≈0, tested intrinsic); no balancer/optimizer helps; cat gain is architecture-dominated. CA/TX + the aligned-pairing gate handed to `closing_data`. |
| [`embedding_eval/`](embedding_eval/) | 2026-06-02 | [`FINAL_SYNTHESIS.md`](embedding_eval/FINAL_SYNTHESIS.md) | 4-level leak-aware substrate ladder; champion **v14 = `check2hgi_design_k_resln_mae_l0_1`**; design_k re-validated at FL (overturned the AL/AZ-only K≡J discard); leak-free reg +0.9–1.1 pp over canonical; **NO MTL benefit** → the joint regime is the wall. |
| [`substrate-protocol-cleanup/`](substrate-protocol-cleanup/) | 2026-05-29 | [`CLOSURE.md`](substrate-protocol-cleanup/CLOSURE.md) | log_T-KD PROMOTED (v12 default); all substrate designs NULL in MTL at AL/AZ/FL (regime-limited — even HGI ≈ canonical in MTL); ResLN encoder = STL-best. |
| [`mtl-protocol-fix/`](mtl-protocol-fix/) | 2026-05-24 | [`AGENT_PROMPT.md`](mtl-protocol-fix/AGENT_PROMPT.md) | F1-selector fix recovers ~95% substrate capacity at deploy (+5.6 pp FL); P4 isolates the residual gap as **architectural** (not cat-vs-reg, not long-tail, not substrate). |
| [`canonical_improvement/`](canonical_improvement/) | 2026-05-19 | [`AGENT_PROMPT.md`](canonical_improvement/AGENT_PROMPT.md) | 18-experiment Tier 1-6 slate; substrate axis exhausted at a ±0.8 pp ceiling; ResLN cat micro-gain (STL-only); GATv2 temporal-edge leak diagnosed. |
| [`hgi_category_injection/`](hgi_category_injection/) | 2026-05-04 | [`STATUS.md`](hgi_category_injection/STATUS.md) | 6 category-injection variants into HGI's POI2Vec all inert (AZ falsified). **Do NOT treat as active without an explicit re-open commit.** |
| [`fusion/`](fusion/) | — | — | Leftover `results/` snapshot only. The fusion *study* proper is archived at [`docs/archive/fusion-study/`](../../archive/fusion-study/). |

Generalizable insights mined across these studies: [`docs/research/community_insights.md`](../../research/community_insights.md).
