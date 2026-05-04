# Check2HGI Study — Entry Point

**Status (2026-05-02 v11, post-v9/v10/v11 closures):** all paper-headline numbers committed; article is at `articles/[BRACIS]_Beyond_Cross_Task/` for BRACIS 2026 submission. The story is **substrate task-asymmetry first, classic MTL tradeoff second**.

> ⚠ **Single source of truth.** All paper numerical claims must round-trip to **`results/RESULTS_TABLE.md §0` (v11)**. Numbers anywhere else in this folder that contradict v11 are either stale (mark, fix, or archive) or audit-historical (clearly framed as such). See `CHANGELOG.md` for the dated rationale.

---

## What the paper says (one-paragraph summary)

Check-in-level Check2HGI provides a **task-asymmetric substrate** for joint POI prediction. Per-visit context lifts single-task next-category macro-F1 by **+15.5 pp at AL, +14.5 at AZ, +29.0 at FL, +28.8 at CA, +28.3 at TX** under matched-head STL (paired Wilcoxon p = 0.0312 at every state, head-invariant at the **AL/AZ ablation scale** and matched-head replicated at **FL/CA/TX**); on next-region, per-place embeddings (HGI) tie or marginally exceed Check2HGI under matched-head ceilings (TOST tied at CA/TX). With Check2HGI fixed, joint MTL over a cross-attention backbone adds a small cat lift at four of five states (AZ +1.20 p < 1e-4 / FL +1.40 p = 2e-06 / CA +1.68 p = 2e-06 / TX +1.89 p = 2e-06, all n = 20) while AL is small-significantly negative (Δ = −0.78 pp, p = 0.036, n = 20 multi-seed) and pays a sign-consistent **7 to 17 pp cost** on next-region Acc@10 at every state — the textbook MTL tradeoff. A pooled-vs-canonical counterfactual at Alabama attributes ~72 % of the cat substrate gap to per-visit context. **A methodological side-finding** (cross-attention `task_weight = 0` ablations co-adapt the silenced encoder via attention K/V; encoder-frozen isolation is required) generalises beyond our study.

---

## Where to start (read in order)

1. **`CHANGELOG.md`** ⭐ — chronological timeline of findings + lessons learned. The single timeline source-of-truth.
2. **`results/RESULTS_TABLE.md §0`** ⭐ — the **canonical numerical source** for paper tables (v11). All other numbers in this folder reference this.
3. **`articles/[BRACIS]_Beyond_Cross_Task/`** — the **article-side working folder** for the BRACIS 2026 submission. Read `AGENT.md` first if you are writing prose, then the section beats in `PAPER_DRAFT.md`.
4. **`AGENT_CONTEXT.md`** — long-form study briefing (read once, refer back as needed).
5. **`NORTH_STAR.md`** — committed champion config (B9 = MTL recipe; H3-alt small-state recipe).
6. **`CLAIMS_AND_HYPOTHESES.md`** (with whitelist banner) — claim catalogue. **Whitelisted as paper-facing safe:** CH16 (substrate cat) / CH18-cat (substrate cat under MTL) / CH15 reframing (substrate reg) / CH19 (per-visit mechanism, AL only) / CH22 (Δm leak-free). Other entries contain superseded leak-era content; do not cite as paper canon without cross-checking RESULTS_TABLE v11.
7. **`FINAL_SURVEY.md`** — substrate-axis 5-state matrix (cat + reg panels).
8. **`MTL_ARCHITECTURE_JOURNEY.md`** — supplementary material narrative (F-trail through B3 → F21c → F45 → F48-H3-alt → F49 → paper closure). **Do not narrate the F-trail in main paper text** (per article-side AGENT.md).
9. **`CONCERNS.md`** — acknowledged risks + resolutions audit log.
10. **`PAPER_BASELINES_STRATEGY.md`** — which baselines appear in which paper table; what is deliberately scoped out.

---

## Navigation

```
docs/studies/check2hgi/
├── README.md                              ← you are here
├── CHANGELOG.md                           ← timeline of findings + lessons (source of truth for chronology)
├── AGENT_CONTEXT.md                       ← long-form study briefing
├── NORTH_STAR.md                          ← committed champion config
├── CLAIMS_AND_HYPOTHESES.md               ← claim catalogue (with whitelist banner)
├── FINAL_SURVEY.md                        ← substrate-axis 5-state matrix
├── CONCERNS.md                            ← acknowledged-risks audit log
├── MTL_ARCHITECTURE_JOURNEY.md            ← supplementary material narrative (F-trail)
├── PAPER_BASELINES_STRATEGY.md            ← which baselines in which paper table
├── results/
│   ├── RESULTS_TABLE.md §0  ⭐            ← canonical numerical source (v11)
│   ├── paired_tests/                      ← Wilcoxon JSONs
│   ├── P0/, P1/, P1_5b/, ...              ← raw JSON artefacts by phase
│   └── phase1_perfold/, probe/            ← Phase-1 substrate-comparison data
├── research/                              ← per-experiment findings + analysis
│   ├── GAP_FILL_WILCOXON.json             ← v8/v9 Wilcoxon (cat-Δ + recipe gap fill)
│   ├── ARCH_DELTA_WILCOXON.json           ← v10 CA/TX §0.1 arch-Δ n=20
│   ├── FL_CAT_DELTA_WILCOXON.json         ← v11 FL §0.1 arch-Δ n=20
│   ├── PAPER_CLOSURE_WILCOXON.json
│   ├── PAPER_CLOSURE_RECIPE_WILCOXON.json
│   ├── F49_LAMBDA0_DECOMPOSITION_GAP.md   ← cross-attn methodology contribution
│   ├── F50_DELTA_M_FINDINGS_LEAKFREE.md
│   ├── F50_T1_RESULTS_SYNTHESIS.md        ← drop-in MTL ablation (FAMO, Aligned-MTL, HSM)
│   ├── F51_MULTI_SEED_FINDINGS.md
│   ├── SUBSTRATE_COMPARISON_FINDINGS.md   ← Phase-1 substrate-comparison verdict
│   └── ...                                ← per-F-number findings
├── baselines/                             ← faithful baseline ports + audits
│   ├── README.md
│   ├── next_category/{poi_rgnn,mha_pe,comparison}.md + results/<state>.json
│   └── next_region/{stan,rehdm,comparison}.md + results/<state>.json
├── paper/                                 ← paper-prep section drafts
│   ├── methods.md, results.md, limitations.md, appendix_methodology.md
├── review/                                ← dated critical reviews
├── issues/, scope/, launch_plans/         ← audit / planning sub-dirs
└── archive/
    ├── post_paper_closure_2026-05-01/     ← stale paper-closure docs (2026-05-01 cleanup)
    │   └── README.md                       ← what's archived and why
    ├── 2026-04-20_status_reports/
    ├── pre_b3_framing/
    ├── research_pre_b3/
    ├── research_pre_b5/
    ├── phases_original/
    └── v1_wip_mixed_scope/
```

**Article-side (BRACIS submission, separate working folder):** `articles/[BRACIS]_Beyond_Cross_Task/` with `AGENT.md`, `PAPER_DRAFT.md`, `PAPER_STRUCTURE.md`, `STATISTICAL_AUDIT.md`, `TABLES_FIGURES.md`, `samplepaper.tex`, `references.bib`, `AUDIT_LOG.md`. The article folder is the source-of-truth for paper writing; this study folder is the source-of-truth for the science.

---

## Task pair

| Slot | Task | Classes | Primary metric |
|---|---|---|---|
| **task_a** | `next_category` | 7 | macro-F1 |
| **task_b** | `next_region` | 1,109 (AL) / 1,547 (AZ) / 4,702 (FL) / 8,501 (CA) / 6,553 (TX) | Acc@10, MRR |

**Preset:** `CHECK2HGI_NEXT_REGION`.

---

## Baselines (see `PAPER_BASELINES_STRATEGY.md` for detail)

- **next-cat:** POI-RGNN (Capanema 2022, faithful), MHA+PE (Zeng 2019, faithful), Markov-1-POI / Majority (simple floors), STL Check2HGI cat (matched-head ceiling), STL HGI cat (substrate ablation CH16).
- **next-reg:** Markov-1-region (simple floor), STL STAN (Luo 2021, faithful — AL/AZ/FL only), STL `next_stan_flow` (matched-head reg ceiling), ReHDM (Li 2025, faithful — AL/AZ/FL only; CA/TX deferred for compute).

---

## Maintenance rules

1. **One canonical source per number.** `results/RESULTS_TABLE.md §0` for paper tables; `CHANGELOG.md` for chronology.
2. **Append to CHANGELOG, don't edit history.** New dated row at the top; never edit historic rows.
3. **Stale artefacts go to `archive/`.** When a tracker / handoff / prompt's work has landed, move the artefact to `archive/post_paper_closure_*/` (or a new dated subfolder) and log the move in CHANGELOG.
4. **Workflow language stays in working notes, not paper-prep docs.** No "in flight", "ETA", "must check before commit" in the article-side files (per article AGENT.md §5).
5. **Article-side and study-side mirror, not duplicate.** When RESULTS_TABLE updates, update the article-side files in the same commit. The article folder is the paper deliverable; this folder is the science record.
