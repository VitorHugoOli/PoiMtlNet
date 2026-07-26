# Chapter 2 (Fundamentals) — Specialist Review Panel Report + Fixes Applied

_Draft 1 reviewed 2026-07-21 by two independent read-only panels; draft 2 reflects the gated fixes.
Read-only means the panels reported findings and altered no prose; every fix below was applied by the
drafting agent and is the author's to accept or revert._

## Panels and headline verdicts

| Panel | Personas | Verdict (on draft 1) | Report |
|---|---|---|---|
| G2 Fact gate | 05 citation, 06 number, 07 claim-honesty | **GATE FAIL** — 2 BLOCKER, 6 SHOULD-FIX, 8 NOTE | `_review/fact_gate_report.md` |
| Domain panel | 10 MTL expert, 11 POI/mobility expert | **SOUND-WITH-CORRECTIONS** — 1 BLOCKER, 5 SHOULD-FIX, 3 NOTE | `_review/domain_review_report.md` |

Both panels re-derived every load-bearing identifier and claim firsthand (OpenAlex / arXiv + the repo
sources of truth) rather than trusting the drafter's self-reports. The fact gate confirmed: 67 distinct
`\cite` keys, 0 dangling; 32 DOI-bearing keys re-resolved (27 exact title match, 1 wrong-paper DOI); 0
retractions; 0 em-dashes / contractions / AI-tell vocabulary; the CBIC null time-indexed; the AL/AZ region
result stated as TOST non-inferiority and not upgraded; MobiWac status wording correct; all 14 balancer-family
attributions supported. The domain panel confirmed the sharing spectrum, the balancer family, the
multi-objective framing, the task distinctions, the sequence-model lineage, and the representation lineage are
all characterized correctly.

## Disposition table (every finding)

Severity is the higher of the two panels where both raised it. "Fixed" = applied in draft 2; "Author" =
recorded for the author to act on at adaptation (bib merge or Ch.5 cross-check); "By design" = intentional
and defended.

| # | Section | Sev | Finding | Disposition |
|---|---|---|---|---|
| B1 | 2.2 | BLOCKER | Check2HGI cited to `silva2025mtlnet` (= the CBIC/MTLnet paper; the MobiWac/Check2HGI work has no bib entry and is submitted/under review). | **Fixed** — cite removed; Check2HGI defers to `\ref{ch:mobiwac}` + the lineage table. No published-status implication. |
| B2 | 2.2 | BLOCKER | "Chapters 3 and 4 use FiLM and these encoders" inverts the arc: MTLnet (Ch.3) uses only the place embedding + per-task FiLM; the decomposed encoders are CoUrb's (Ch.4) contribution. | **Fixed** — FiLM = per-task conditioning; the decomposition is attributed to Ch.4 only and named as the arc's turning point; Space2Vec noted as named-but-not-adopted. |
| B3 | 2.3 / bib | BLOCKER | `misra2016cross` DOI `.434` resolves to the wrong paper; correct Cross-Stitch DOI is `.433`. | **Author** — author-owned CBIC bib; already recorded in `articles/CBIC___MTL/ERRATA.md` item 4 (verified). Sentence itself is accurate. |
| S1 | 2.1 vs 2.3 | SHOULD-FIX | Contradiction: 2.1 cites single-task region prediction (zhu2022drrgnn) as an end target; 2.3 asserts "none predicts the next region as an end target." | **Fixed** — 2.3 universal scoped to the multi-task co-equal setting; single-task region work acknowledged. |
| S2 | 2.3 | SHOULD-FIX | Negative transfer credited to `Zhang2020` (= iMTL next-POI recommender, not negative-transfer literature). | **Fixed** — definition folded onto the already-cited, verified `standley2020tasks`; `Zhang2020` cite removed. |
| S3 | 2.2 | SHOULD-FIX | FiLM described as injecting context; in these models FiLM conditions on task identity and context enters by concatenation. | **Fixed** — corrected to per-task γ/β conditioning + concatenated context (matches GLOSSARY + method sources). |
| S4 | 2.3 | SHOULD-FIX | Missing the canonical scalarization-skeptic anchor (Kurin, "In Defense of the Unitary Scalarization"). | **Fixed** — added `kurin2022scalarization` (arXiv:2201.04122, arXiv record + claim verified firsthand; NeurIPS 2022 venue seen in search only, [VERIFY] flagged in the bib, no DOI field); new bib entry. |
| S5 | 2.4 | SHOULD-FIX | 93% predictability ceiling over-applied to category macro-F1 / region Acc@10. | **Fixed** — rescoped to a next-location bound; the dedicated single-task model named as the operative ceiling. |
| S6 | 2.4 | SHOULD-FIX | "Two datasets serve as the ground" then names three (Foursquare is not used). | **Fixed** — Foursquare marked context-only; "the two" = Gowalla + Massive-STEPS/Istanbul. |
| S7 | 2.5 | SHOULD-FIX | Win half of the result sentence used a non-comparative verb with no superiority test named. | **Fixed** — now "by paired superiority tests, outperforms ... and matches ... by non-inferiority testing"; points to Ch.5. |
| S8 | 2.3/2.4/2.5 | SHOULD-FIX | Banned result verb "beat" x4. | **Fixed** — all reworded to "outperform"/"do not outperform"; DRAFT_LEDGER self-cert corrected. |
| S9 | 2.4 / bib | SHOULD-FIX | `pedregosa2011sklearn` cited for StratifiedGroupKFold (a v1.0/2021 feature). | **Fixed/Author** — single-cite ruling retained (author); splitter behavior stated defensively; date noted in ledger. |
| S10 | 2.3 | SHOULD-FIX | "the structured-sharing topology the later joint model descends from [PLE]" is architecturally imprecise (the joint model is cross-attention, not MoE). | **Fixed** — reworded to the structured-sharing PRINCIPLE the joint model adopts. |
| N1 | 2.4 | NOTE | Food "roughly a third" is a single representative state; addendum provenance points to the POI-count table (32.5%) not the check-in table (34.2%). | **Fixed (prose) / Author (addendum)** — prose now says "in a representative state"; ledger flags the addendum pointer to correct. |
| N2 | 2.5 | NOTE | "four of six datasets" + two-point margin trace to NORTH_STAR §6, not the Ch.5 primary board. | **Author** — confirm against RESULTS_BOARD.md / PAPER_PLAN §3 at adaptation. |
| N3 | 2.4 | NOTE | OOD/unseen-region handling not foreshadowed. | **Fixed** — one clause added (unseen region = miss; full definition in Ch.5). |
| N4 | bib | NOTE | DGI triple-key and Nash-MTL slash-key duplicates will collide on merge. | **Author** — consolidate before compiling the single global bib (in BIB_NOTES §B). |
| N5 | 2.4 / bib | NOTE | `kohavi1995crossval` claim graded PLAUSIBLE (Zenodo re-deposit id, original text not opened). | **Author** — confirm original IJCAI-95 text at adaptation. |
| N6 | 2.1 | NOTE | FPMC (Rendle 2010) absent from the next-place lineage. | **By design** — lineage begins at the deep-learning era; FPMC optional. |
| N7 | 2.4 / bib | NOTE | Massive-STEPS is a 2025 preprint. | **Author** — re-check for a peer-reviewed version before the August 2026 defense. |
| N8 | 2.4 | NOTE | Δm sign convention differs from Maninis' original (drop vs lead). | **By design** — the addendum defines the convention explicitly; ensure Ch.5 matches it. |

## Style pass

No dedicated style specialist (persona 03) was run as a third sub-agent; the style checks that are
mechanically decidable were run by the drafting agent and re-confirmed by the fact gate's persona-07 sweep:
zero em-dashes, zero contractions, zero AI-tell vocabulary across all five sections and the lineage table,
section openers varied, no section ends by restating itself, and no repo codenames. A human or persona-03
distributional read (burstiness, sentence-rhythm) remains available if the author wants it before the
defense; it is not a fail-closed gate item.

## Net effect on draft 2

- 2 BLOCKERS: both fixed (B1, B2); B3 is an author-owned bib DOI already in the CBIC ERRATA.
- 10 SHOULD-FIX: all fixed in the drafts, except the two author-owned bib actions (misra DOI, sklearn
  single-cite ruling), which are recorded.
- 8 NOTES: 2 fixed in prose (OOD foreshadow, Food single-state), 6 recorded for the author at adaptation.
- 1 new verified citation added (`kurin2022scalarization`); ch2 bib now 27 entries, 0 dangling keys, 67
  cite keys all resolving against the project bib universe.

_Full panel reports: `_review/fact_gate_report.md`, `_review/domain_review_report.md`. Per-section
citation/number ledgers: the trailing comment block in each `2.x_*.tex`, consolidated in `DRAFT_LEDGER.md`._
