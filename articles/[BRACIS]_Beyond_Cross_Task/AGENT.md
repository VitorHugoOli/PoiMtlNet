# AGENT.md — BRACIS 2026 Paper Working Guide

> **Read order for any agent picking this up:** (1) this file, (2) `PAPER_STRUCTURE.md`, (3) `PAPER_DRAFT.md`, (4) the open decisions in §0 of `PAPER_DRAFT.md`, (5) the linked study artefacts in `docs/studies/check2hgi/`. Do not draft prose until §0 of `PAPER_DRAFT.md` is locked by the user.

This file is operational, not encyclopedic. It exists so a sub-agent (or future Claude/Vitor) can pick up a single section and write it consistently with the rest of the paper without re-reading the whole study trail.

---

## 1 · The paper at a glance

- **Venue:** BRACIS 2026, Main Track. Springer LNAI proceedings.
- **Format:** Springer LNCS, single-blind not allowed — **double-blind**, ≤ 15 pages including references and appendices.
- **Submission window:** 2026-04-20 → 2026-05-04 (extended). Author notification 2026-06-08, camera-ready 2026-06-29.
- **Submission system:** JEMS3 (`https://jems3.sbc.org.br/bracis2026`).
- **Mandatory reviewer commitment:** at least one author must volunteer to review 3 papers (see `docs/BRACIS_GUIDE.md` §3).
- **Working title (locked-default 2026-05-01):** *Beyond Cross-Task Transfer: Check-In-Level Embeddings and the Scale-Sensitive Multi-Task Tradeoff in POI Prediction*.
- **Working tagline:** *Per-visit context wins next-category at every U.S.-state Gowalla split (≥15 pp at smaller anchors, ≥29 pp at the headline scale); multi-task learning then gains a small cat lift and pays a sign-consistent reg cost, but the cost shrinks as data grows on the small-to-medium regime — the textbook MTL tradeoff, with a scale-progression mechanism.*
- **Headline scale (per `PAPER_DRAFT.md §0` D2):** three states **FL/CA/TX**; AL/AZ retained as smaller-scale anchors that surface the scale-progression finding.

The full study lives under `docs/studies/check2hgi/`. The most recent paper-closure synthesis is `PAPER_CLOSURE_RESULTS_2026-05-01.md` and `FINAL_SURVEY.md`. **These supersede the older "+6.48 pp MTL > STL on AL" framing** that you may still see in `PAPER_DRAFT.md` (the old one), `MTL_ARCHITECTURE_JOURNEY.md`, and the F49 docs — those numbers were leak-artifacts (asymmetric C4 leak inflated MTL more than STL). The leak-free closure says: MTL trails STL on `next_region` by 7–17 pp at every one of the five states.

---

## 2 · The story arc (the spine sub-agents must respect)

The user's narrative across three papers, in order:

1. **CBIC 2025 (`articles/CBIC___MTL/`).** First MTL attempt. Hard-parameter sharing, DGI POI embeddings, FiLM modulation, NashMTL. Honest negative result: MTL did not consistently beat STL; gains were within σ. Diagnosis: task dissimilarity (static cat vs. sequential next), restrictive sharing, representation mismatch. Frames a hypothesis: *embedding granularity may be the bottleneck.*
2. **CoUrb 2026 (`articles/CoUrb_2026/`).** Embedding-side response: ST-MTLNet decomposes the monolithic DGI vector into spatial (SIREN/Sphere2Vec-M) + temporal (Time2Vec) + categorical (HGI) sub-encoders. Big cat-side wins (+20–24 pp) across FL/CA/TX. Confirms: *the embedding is a huge part of the story.* But it is concatenation-fusion, and `next-POI` gains are uneven (16/21 cells).
3. **BRACIS 2026 (this paper).** Refines the embedding-as-bottleneck thesis with a single, principled substrate — **Check2HGI**, a check-in-level contextual graph embedding (per-visit, not per-POI). We use Check2HGI for two complementary tasks, **next-category** (7 classes) and **next-region** (~1.1K to ~8.5K classes), and ask the natural follow-up: does multi-task learning over this richer substrate finally deliver bidirectional gains? Five-state, leak-free, paired-Wilcoxon answer: **the substrate carries the cat win; MTL gains a small additional cat lift; MTL costs the harder region task — the classic MTL tradeoff is sign-consistent across all five states**.

The arc is *not* "we extend ST-MTLNet" or "we fix the CBIC negative result" — those are different task pairs. The arc is "embedding choice is the load-bearing lever, and once we lock the right substrate, MTL behaves like the textbook tradeoff says it should."

### Three concrete contributions (in priority order)

- **C1 — Substrate (the strongest survival, scales with data).** Check-in-level Check2HGI lifts next-category macro-F1 over POI-stable HGI under matched-head STL by **+15 pp** at the smaller-scale anchors AL/AZ and **+29–33 pp** at the headline scale FL/CA/TX, head-invariant (linear / GRU / single / LSTM probes), paired Wilcoxon **p = 0.0312** at every state (max significance for n = 5, all 5/5 folds positive). The substrate Δ scales monotonically with data — the embedding gets *more* useful at scale. A POI-pooled counterfactual at AL attributes ~72 % of the cat gap to per-visit context and ~28 % to Check2HGI's training signal.
- **C2 — Scale-sensitive MTL tradeoff.** With Check2HGI fixed as substrate and a cross-attention MTL backbone, joint training lifts next-category by +0 to +2 pp at every state (sign-consistent) and pays a sign-consistent reg cost vs. a matched-head STAN-Flow STL ceiling. The cost magnitude **shrinks as data grows on the small-to-medium regime** — Δreg = −11.04 pp at AL → −12.28 at AZ → **−7.28 at FL** (~5 pp recovery from anchors to smallest headline state) — direct evidence that *MTL begins to generalise on reg as data scales*. CA preserves the regime (−8.93 pp); TX is the honest non-monotone outlier (−16.69 pp), pointing at state-specific factors (transition-graph density, per-user trajectory geometry) beyond raw cardinality. Drop-in alternatives (FAMO, Aligned-MTL, hierarchical-softmax reg head) do not recover the gap.
- **C3 — Methodological note (cross-attention λ = 0 pitfall).** Under cross-attention MTL, a loss-side `task_weight = 0` ablation does not silence the silenced encoder — it co-adapts via attention K/V projections. Encoder-frozen isolation is the only clean architectural decomposition. The note generalises to MulT, InvPT, and any cross-task attention-based MTL.

C3 is short and lives in a methodological note / appendix sub-section. C1 + C2 are the headline.

### What is **not** the contribution

- "We propose Check2HGI" — Check2HGI is a derivative engine in our codebase; the paper *uses* it as the substrate, but the contribution is the empirical claim about per-visit context, not the engine recipe (which is described in the supplement and cited by anonymized self-reference if necessary).
- "We propose per-head learning rates" — H3-alt was a detour. Under leak-free measurement, H3-alt is the universal small-state recipe and B9 the FL-tuned recipe; neither closes the MTL-vs-STL gap on `next_region`. The recipe-selection story belongs in the ablation/appendix band, not the headline.
- "We refute MTL." We do not. MTL still wins on cat at every state, sign-consistent, and is the deployment unit. We position the reg loss honestly as the cost paid for joint single-model deployment.

---

## 3 · Anonymization rules (double-blind)

- No author names, affiliations, e-mails, ORCIDs, funders.
- **Cite prior CBIC/CoUrb papers in third person**, exactly as you would cite an external work — e.g. *"Silva et al. (2025) found that an MTL framework on DGI embeddings did not consistently improve over single-task baselines, attributing the gap to representation mismatch and task dissimilarity."* Do not write "our prior work" or "we previously showed" in the main text. Acknowledgments and the GitHub URL are removed for review and restored at camera-ready.
- Code/data link: use **Anonymous GitHub** (`https://anonymous.4open.science/`) or **Anonymous Dropbox**. Do not paste real GitHub URLs.
- Do not name the lab, the institution, the funder, or the city in the main text. The Gowalla state set (Alabama, Arizona, Florida, California, Texas) is fine — those are dataset partitions, not author origins.
- LLM-usage acknowledgment goes in Acknowledgments at camera-ready (per BRACIS rules). Not in the review version.

---

## 4 · Voice and writing style (calibrated from CBIC + CoUrb)

The user's English (CBIC) and Portuguese (CoUrb) papers share a recognisable voice. Match it.

- **Hypothesis up front, not buried.** Each section begins with a one-sentence claim of what is being established, then evidence. The CBIC intro states the central hypothesis explicitly: *"a standard hard parameter-sharing MTL architecture will face significant limitations…"* — adopt the same up-front framing.
- **Honest framing.** When a result is mixed, say so. CBIC's *"the MTL approach did not consistently demonstrate superior performance"* is an example. The BRACIS 2023 best paper *Embracing Data Irregularities* used the same honest opener with "low computational cost" instead of peak F1. Lean into honesty when the substrate wins and the architecture costs.
- **Numerics always with σ.** Report mean ± std, paired Wilcoxon p, fold-positivity (e.g. *5/5 folds positive at p = 0.0312*). Never bare means in result paragraphs.
- **No polemic.** "Beyond Cross-Task Transfer" is the limit of how aggressive the framing should sound. Do **not** write "Not Transfer" or "MTL is broken." Reviewer-friendly tone wins.
- **Contribution bullets at end of intro** (CBIC pattern, three to four bullets, one sentence each, lettered C1/C2/C3 to mirror this guide).
- **Plain-prose method, equations only when load-bearing.** Cross-attention is described conceptually; we don't reproduce attention equations. STAN-Flow is described as "STAN attention backbone + a learnable scalar α gating a region-trajectory log-transition prior." Equations only for: the joint-loss form, the Δm metric (Maninis 2019), and STAN-Flow's α-gated prior (because it is the load-bearing reg head detail).
- **Per-paragraph load:** state a claim, defend with one number or one mechanism, move on. Do not stack three findings into one paragraph.
- **Tables before figures.** BRACIS reviewers respond to numerics; figures are mechanism aids. Do not commit a figure unless it carries a fact a table cannot.
- **Tense.** Past tense for our experiments ("we trained 5-fold CV"), present tense for our claims and the model's behaviour ("Check2HGI carries per-visit context"), present tense for the literature ("Yang et al. report").
- **No "novel" / "intuitive" / "obvious" / "carefully designed" / "extensive" floweriness.** The CBIC and CoUrb papers do not lean on these words; neither should we.
- **Title pattern.** BRACIS canonical patterns (per `docs/BRACIS_GUIDE.md` §10.3): A (acronym + colon), B (concept-for-task), C (verb-object-domain), D (phenomenon-and-method, 2023 best-paper). Stay in A or D; avoid C (too applied for a methods paper).

---

## 5 · Statistical reporting protocol

The paper inherits the F50/F51/PAPER_CLOSURE protocol. Sub-agents do not negotiate this.

- **Folds:** 5-fold `StratifiedGroupKFold(groups=userid, shuffle=True, random_state=42)` — user-disjoint. Mention this explicitly in §Experimental Setup.
- **Per-fold transition log_T:** `--per-fold-transition-dir`, seed-tagged file. (Cite the leak-free protocol; do not claim leak-free without it.)
- **Multi-seed:** seeds {0, 1, 7, 100} on top of the seed=42 anchor where available — AL+AZ+FL have full multi-seed; **CA+TX multi-seed is in flight on H100 at submission with ~1 h ETA (2026-05-01 PM)**, lifting the n = 5 ceiling at every state once it lands. If it lands direction-consistent with seed = 42, T3/T4 cells upgrade to n = 20 pooled paired Δs. If not landed in time, CA/TX stay seed-42 only (flagged for camera-ready per `PAPER_CLOSURE_RESULTS_2026-05-01.md §8`). Sub-agent A5 must check before T3 commits.
- **Hardware (mandatory in §Experimental Setup):** all headline-scale runs (FL/CA/TX) on a **single NVIDIA H100 80 GB** GPU; AL/AZ additionally validated on Apple-Silicon MPS for cross-platform reproducibility. Per-run wall time: AL/AZ ~10 min, FL ~30 min, CA/TX ~50 min on H100 (5-fold × 50-epoch). Anonymize the platform name (no "Lightning Studio" or any subscription tier — that outs lab funding).
- **Asymmetric baseline coverage (mandatory in §7 Limitations + T5 footnote):** ReHDM (Li et al., IJCAI 2025) reported at AL/AZ/FL only; CA/TX deferred — the dual-level hypergraph's collaborator pool scales quadratically with region cardinality (8.5 K and 6.5 K regions exceeded our H100 per-cell compute budget). Frame as honest baseline-coverage limitation; `BRACIS_GUIDE.md §10.2.7` rewards this pattern.
- **Paired test:** Wilcoxon signed-rank, `alternative='greater'` for the directional claim. p = 0.0312 is the **maximum achievable significance for n = 5 paired samples** (5/5 folds positive); state this once in §Experimental Setup so reviewers know the n = 5 ceiling. Pooled multi-seed (4 × 5 = 20 or 5 × 5 = 25 fold-pairs) gives sub-1e-5 p-values where available.
- **Effect-size axis:** report Δ in percentage points (pp) for F1 / Acc@K / MRR. **Δm (Maninis 2019, Vandenhende 2021)** is the joint score; report leak-free Δm primary (cat F1 + reg MRR) and Δm secondary (cat F1 + reg Acc@10).
- **TOST non-inferiority** at δ = 2 pp / δ = 3 pp where the claim is "tied at scale" (used for CH15 reframing on CA/TX).
- **Sign-consistency framing:** when a Δ is sign-consistent across 5 states, say so explicitly — that is the strongest qualitative claim available at n = 5 single-seed.

---

## 6 · Page budget and what goes where

15-page hard cap. Suggested allocation (sub-agents may negotiate ±0.5 page):

| Section | Pages | What lives here | Claims hit |
|---|---:|---|---|
| 1 — Introduction | 1.5 | Problem, gap, the three-paper arc framed in third person, contributions C1/C2/C3 | — |
| 2 — Related Work | 1.5 | POI embeddings (DGI, HGI, Check2HGI line), MTL methods (NashMTL, PCGrad, FAMO, Aligned-MTL), POI-MTL prior (HMT-GRN, GETNext, MGCL) | — |
| 3 — Method | 2.5 | Check2HGI substrate (one paragraph, conceptual), MTL backbone (cross-attention + static_weight), reg head (STAN-Flow), cat head (next_gru), training protocol | — |
| 4 — Experimental Setup | 1.5 | Five Gowalla state splits (table), folds + seeds, baselines (cat: POI-RGNN, MHA+PE, STL `next_gru`; reg: Markov-1-region, STL GRU, STL STAN, STL STAN-Flow), metrics, statistical protocol | — |
| 5 — Results | 4 | T1 substrate (C1), T2 cat MTL vs STL (C2 cat-side), T3 reg MTL vs STL (C2 reg-side), T4 Δm joint (C2 synthesis), T5 baselines comparison | C1, C2 |
| 6 — Mechanism / Analysis | 1.5 | Per-visit-context counterfactual (~72 % at AL); ablations refuting drop-in fixes (FAMO, Aligned-MTL, HSM); short methodological note on the cross-attn λ = 0 pitfall | C2 robustness, C3 |
| 7 — Discussion & Limitations | 1 | Re-attribution: substrate carries cat, architecture costs reg, sign-consistent across 5 states. Limitations: CA/TX seed = 42 single-seed; FL `next_region` Markov-saturation; encoder enrichment is future work | — |
| 8 — Conclusion + Future Work | 0.5 | Spirit of the paper in 4–6 lines; future directions are over-coming the tradeoff (PLE / Cross-Stitch / dynamic routing / encoder enrichment per `research/F50_HANDOFF_2026-04-28.md`) | — |
| 9 — References | 0.75 | 25–30 refs max, BibTeX style splncs04 | — |
| Appendix / Supplement | (does not count toward 15 pages if outside; if inside, ≤ 0.5 pages) | F-trail, the 28-run paper-closure matrix, raw per-fold tables, the leak-magnitude appendix | — |

The F-numbered experiment trail (F21c → F44 → F45 → F48-H3-alt → F49 → F50 → F51 → paper-closure) does **not** belong in main text. It belongs in supplementary materials as an attribution narrative. Main text gives the *what* (recipe + result), not the *F-number trail*.

---

## 7 · Tables and figures (per `PAPER_STRUCTURE.md` §4-§5 in this folder)

- **T1 — Dataset stats** (5 states × {users, check-ins, POIs, regions, train/val/test split sizes}).
- **T2 — Substrate comparison (CH16, head-invariant).** Cat F1 STL across {Linear probe, `next_gru`, `next_single`, `next_lstm`} × {Check2HGI, HGI} × {AL, AZ, FL, CA, TX}. Source: `FINAL_SURVEY.md` §1-2.
- **T3 — MTL vs STL headline (C2).** Cat F1 + reg Acc@10 + reg MRR for {STL `next_gru`, MTL B9 cat F1; STL STAN-Flow (`next_stan_flow`), MTL B9 reg} × 5 states. Source: `PAPER_CLOSURE_RESULTS_2026-05-01.md` §4a.
- **T4 — Δm joint score (CH22 leak-free).** Δm-MRR / Δm-Acc@10 × 5 states with paired Wilcoxon p. Multi-seed for FL (n = 25 pairs) where available. Source: `CLAIMS_AND_HYPOTHESES.md §CH22 (2026-05-01 reframe)`.
- **T5 — External baselines.** POI-RGNN / MHA+PE for cat; Markov-1-region / STAN / ReHDM for reg; per state. Source: `baselines/next_category/results/<state>.json` and `baselines/next_region/results/<state>.json`.
- **T6 — MTL drop-in ablation (CH22b).** FAMO, Aligned-MTL, HSM-reg-head at FL only — all fail to recover. Source: `research/F50_T1_RESULTS_SYNTHESIS.md`.
- **F1 (optional) — Per-visit mechanism schematic.** AL pooled-vs-canonical bar chart (~72 % of cat gap is per-visit context).
- **F2 (optional) — Architectural cost vs region cardinality.** Δreg pp vs n_regions across the 5 states. Visual evidence the cost is sign-consistent (always negative) but not strictly monotone.

Tables T1–T4 are required; T5 strongly recommended; T6 optional (lives in appendix if cut). Figures are optional — only include F1 if pages allow.

---

## 8 · Pointers to source-of-truth artefacts (under `docs/studies/check2hgi/`)

| What | Where |
|---|---|
| Headline numbers + Δs across 5 states | `FINAL_SURVEY.md`, `PAPER_CLOSURE_RESULTS_2026-05-01.md` |
| Claim catalogue (CH16, CH18, CH19, CH20, CH21, CH22, CH22b) | `CLAIMS_AND_HYPOTHESES.md` (read CH16, CH22 first) |
| Champion recipe (B9 / H3-alt scale-conditional) | `NORTH_STAR.md` |
| F-trail (architecture journey, supplementary) | `MTL_ARCHITECTURE_JOURNEY.md` |
| Statistical artefacts (Wilcoxon JSONs) | `results/paired_tests/F50_T0_delta_m*.json`, `research/PAPER_CLOSURE_WILCOXON.json`, `research/PAPER_CLOSURE_RECIPE_WILCOXON.json` |
| Methodological note (cross-attn λ = 0 pitfall) | `research/F49_LAMBDA0_DECOMPOSITION_GAP.md` |
| Per-visit mechanism counterfactual | `research/SUBSTRATE_COMPARISON_FINDINGS.md`, `CLAIMS_AND_HYPOTHESES.md §CH19` |
| FAMO / Aligned-MTL / HSM ablations | `research/F50_T1_RESULTS_SYNTHESIS.md` |
| Baselines audits | `docs/studies/check2hgi/baselines/{next_category,next_region}/comparison.md` |
| sklearn version reproducibility caveat | `FINAL_SURVEY.md` §8 |

---

## 9 · Operational rules for sub-agents

1. **Do not invent numbers.** Every numeric in a draft section must have a pointer to a study artefact in this guide's §8. If you cannot find the number, leave a `\TODO{cite XX}` and surface it back to the user; do not back-fill.
2. **Do not relitigate the open decisions in §0 of `PAPER_DRAFT.md`.** Those are the user's calls.
3. **Do not narrate the F-number trail in main text.** The reader does not need F21c → F45 → F48-H3-alt to follow the paper. They need: substrate Δ, MTL Δ, sign-consistency, drop-in ablation refusing to recover, methodological note.
4. **Cite leak-free.** Do not quote any reg number that pre-dates the F50/F51 leak-free protocol unless explicitly framed as "the legacy leaky measurement was X; the leak-free measurement is Y." The PAPER_CLOSURE numbers are canonical.
5. **Honour double-blind.** No author names, no `silva2025mtlnet`-style self-cites — cite as third party.
6. **One claim per paragraph.** If you want to land two facts, write two paragraphs.
7. **Default to omitting a comment in the LaTeX.** Comments belong in this guide and in the section docs, not littered through `.tex` files.
8. **If you change the title or abstract**, update both `PAPER_DRAFT.md §0` (decision register) and `samplepaper.tex` in the same commit. Never leave them out of sync.

---

## 10 · Submission checklist (mirrored from `docs/BRACIS_GUIDE.md` §9)

- [ ] Anonymise authors, affiliations, acknowledgments, e-mails, GitHub URL → Anonymous GitHub
- [ ] ≤ 15 pages in the Springer LNCS template (this folder's `samplepaper.tex` shell)
- [ ] Abstract 70–150 words, single paragraph, no footnotes / equations / citations / undefined acronyms
- [ ] Three keywords (LNCS template, separated by `\and`)
- [ ] Bibliography style `splncs04` (the `.bst` is in this folder)
- [ ] Register paper on JEMS3 (paper registration window: 2026-04-13 → 2026-04-27)
- [ ] Volunteer at least one author as reviewer (3 papers)
- [ ] Compare with at least 2–3 external baselines (POI-RGNN, MHA+PE for cat; STAN, ReHDM, Markov-1-region for reg — we already do this)
- [ ] Acknowledge LLM usage in Acknowledgments (camera-ready, after notification)
- [ ] Proofread English carefully; pass through a native or polished editor before submission
