# 05 · Citation Auditor — Report (dissertation v1)

**Status: COMPLETE**
**Reviewer:** Citation auditor (persona 05), G2 fact gate, rules R1–R5
**Scope:** ALL SIX chapters (src/chapters/1..6) + appendices + front matter (src/0_main.tex)
**Bib under audit:** src/references.bib (99 entries) + src/BIB_MERGE_REPORT.md
**Donor/template bib (verified):** articles/[mobiwac]/src/references.bib

---

## ★ VERDICT: GATE PASS

**No fabricated entry, no unresolvable entry, no unfixed R4 erratum, and no claim-support
failure on a load-bearing sentence.** One MAJOR finding (F-1, a wrong-source citation for the
Gowalla dataset in Ch.4, inherited from the CoUrb original) should be fixed before the advisor
handoff, but it is a mis-*attribution of a real, correctly-recorded work to the wrong sentence*,
not a fabrication or an unresolvable entry — it does not by itself fail the gate. Everything
else is MINOR/NIT.

**Per-chapter verdicts:**
| Chapter | Entry-level (R1/R2) | Claim-support (R3) | Verdict |
|---|---|---|---|
| Ch.1 Introduction | all cited entries real/attributed | 8/8 sites SUPPORTED | **PASS** |
| Ch.2 Fundamentals | all real/attributed (67 keys, most new-this-pass) | 72/72 sites SUPPORTED | **PASS** |
| Ch.3 CBIC | all real/attributed; R4 errata fixed | all key-sets SUPPORTED | **PASS** |
| Ch.4 CoUrb | all real/attributed; R4 errata fixed | 1 MAJOR mis-cite (F-1) | **PASS w/ F-1 fix** |
| Ch.5 MobiWac | all real/attributed | all key-sets SUPPORTED | **PASS** |
| Ch.6 + apx A/B/C + front matter | citation-free by design (0 `\cite`) | n/a | **PASS** |

**[VERIFY] list for the author:** NONE outstanding. Every one of the 99 entries resolved against
a source of record this session; the 3 inherited `[VERIFY]` caveat-comments in the bib
(kohavi1995crossval Zenodo re-deposit; wilcoxon1945 page range; yang2015tsmc online-first year)
are verification *detail*, not open questions — confirmed benign.

---

## Progress log
- [x] Read persona 05 + reviewers/README common protocol
- [x] Read CLAUDE.md
- [x] Read AGENT_GUARDRAILS §1 (citation protocol R1-R5)
- [x] Read NORTH_STAR §4 (inherited errata for R4)
- [x] Extract all \cite keys per chapter from .tex + .aux
- [x] Parse references.bib entries (99)
- [x] Cross-check: every cite key resolves; orphans; unresolved [?] — 0/0/0
- [x] Entry-level audit — 100% (99/99) against sources of record
- [x] Claim-support audit — 178/243 sites (73%), 100% of Ch.1+Ch.2
- [x] R4 errata check — all 5 fixed
- [x] R5 sweep (no AI output as source) — clean
- [x] Build-level render check (L4) — 0 unresolved markers in 87pp PDF
- [x] Per-chapter verdicts + document verdict — GATE PASS

---

## PART A — Mechanical cross-checks (100% coverage)

**A1. Cite-key ↔ bib-entry closure (rule: every `\cite` resolves; no orphans).**
Extracted every `\cite*` key from the 6 chapter `.tex` + 3 appendix `.tex` + `0_main.tex`,
split multi-key cites, compared against the 99 `@`-entries in `src/references.bib`:
- **99 distinct cited keys = 99 bib entries. 0 dangling (cited-not-in-bib). 0 orphan
  (in-bib-not-cited). 0 duplicate bib keys.** Matches BIB_MERGE_REPORT §1's claim exactly.
- Per-chapter distinct-key counts: Ch.1 = 9, Ch.2 = 67, Ch.3 = 31, Ch.4 = 28, Ch.5 = 33.
  Ch.6 / apx A/B/C carry no `\cite` (conclusion + appendices are citation-free by design).
- **Build-level confirmation (L4 cross-ref lint):** `main_defense.blg` reports
  `warning$ -- 0`, `cite$ -- 99`, 101 entries used (99 + 2 abntex2-options internal),
  **0 undefined citations**. The Viegas precedent's raw-`[?]`/unresolved-key defect class is
  ABSENT. Verified on the committed build, not asserted.

## PART B — Entry-level audit (R1/R2) — 100% coverage (99/99)

Every entry was resolved against a source of record this session (OpenAlex `get_work` by DOI;
arXiv `get_papers` by ID; OpenAlex/arXiv title+author search for identifier-light entries).
Identifier inventory: **63 DOI, 24 arXiv-only, 2 URL-only (FLAN OpenReview; SNAP dataset
`@misc`), 10 "record-only"** (full venue+year+pages, no DOI/arXiv string — all well-known works).

**Result: 99/99 entries are real, correctly attributed works. ZERO fabrications, ZERO
unresolvable entries, ZERO retractions.**

- **63/63 DOIs resolved** on OpenAlex; title/first-author/venue/pages match the bib. Automated
  flags all cleared on inspection (OpenAlex truncates titles at the subtitle colon and reorders
  some author lists — bib is correct in every case: e.g. `kendall2018uncertainty` bib order
  Kendall–Gal–Cipolla is the published order; OpenAlex reversed it).
- **24/24 arXiv IDs resolved** (`arxiv_get_papers`: n_found 24, not_found 0). Titles/authors
  match. One title-convention note below (`ruder2017sluice`).
- **10/10 record-only entries confirmed** by title+author+venue+year against OpenAlex
  (GradNorm/ICML'18, Holm 1979/Scand.J.Statist., Nash-MTL/Navon ICML'22, FAMO/Bo Liu NeurIPS'23,
  Aligned-MTL/CVPR'23, PCGrad/Yu NeurIPS'20, DGI/Veličković ICLR, sklearn/Pedregosa JMLR'11,
  Kurin/NeurIPS'22, `xin2022domtl`/Xin arXiv:2209.11379 NeurIPS'22 — all exact author matches).
- **Provenance-comment convention verified consistent:** every `% PROVENANCE`/`% verified`/
  self-labeled `[key]` block PRECEDES its entry (checked mechanically: 22/22 self-labeled
  annotations align to the following entry, 0 misfiled). No off-by-one in the provenance trail.

### B — findings (entry level)
See ranked findings §FINDINGS below (F-5, F-6, F-7 are the only entry-level items; all MINOR/NIT).

## PART C — R4 inherited-errata check (NORTH_STAR §4) — all FIXED
| Erratum (NORTH_STAR §4) | Required fix | State in bib | Verdict |
|---|---|---|---|
| CBIC: POI-RGNN wrong paper | use `capanema2023poirgnn` | Ad Hoc Netw. 138:103016 (2023), DOI 10.1016/j.adhoc.2022.103016, 5 authors — Crossref-confirmed | **FIXED** |
| CBIC: HMRM author names | `chen2020modeling` 5 authors incl. Yang Liu | Meng Chen, Yan Zhao, Yang Liu, Xiaohui Yu, Kai Zheng; TKDE 34(4):1902-1914; DOI 10.1109/TKDE.2020.3001025 — OpenAlex exact match | **FIXED** |
| CBIC: GAT cite the ICLR version | `velivckovic2017graph` → ICLR 2018 | booktitle Proc. ICLR, year 2018, arXiv:1710.10903 note | **FIXED** |
| CoUrb: `silva2025mtlnet` venue wrong | venue = CBIC 2025, drop "Submetido" | booktitle "Anais do XVII Congresso Brasileiro de Inteligência Computacional (CBIC 2025)", DOI 10.21528/CBIC2025-1191324, no Submetido | **FIXED** |
| (bonus) `paiva2026stmtlnet` 3rd author | restore Germano B. dos Santos | 4 authors incl. Germano B. dos Santos, pp 323-336, DOI 10.5753/courb.2026.22960 | **FIXED** |

## PART D — R5 sweep (no AI output as source): CLEAN
Grepped `references.bib` and all chapter/appendix `.tex` for AI-system / chatbot /
"personal communication" / "generated by" citations: **NONE**. The one lexical hit
(`wei2022finetuned`, "Finetuned Language Models Are Zero-Shot Learners", ICLR 2022) is a
legitimate peer-reviewed work cited as a real MTL-instruction-tuning example, not a laundered
model claim. The AI-use-disclosure appendix (apx_c) correctly carries **zero `\cite`**; apx_a
and apx_b likewise citation-free (the lone `\cite` grep hit in apx_b is inside a comment).
No claim is sourced to a model output.

## PART E — R3 claim-support audit — 73% of sites (178/243), 100% of Ch.1+Ch.2
**Coverage & method.** 243 citation sites total. Read in full context: Ch.1 (8/8) + Ch.2
(72/72) = **100% of the frame + fundamentals**, the highest-AI-share, most-new-entry chapters
(guardrails §5: audit intensity scales with AI share). Ch.3/4/5 (re-typeset published text):
**every distinct key-set site** read (35+28+35 = 98). Total **178/243 = 73%**, far above the
R3 ≥20% floor, and 100% for the entries new-this-pass (Ch.2's fundamentals set). Each cited
system was checked for (a) strength drift, (b) second-hand attribution, (c) description
fidelity ("as its authors describe it"), (d) hedge preservation. 60/99 entries carry the
drafter's own recorded claim/verification note; those were cross-read against the citing
sentence. Recent/sparse works (2022-2025) with specific technical claims were re-verified
against their sources this session (MCMG, DRRGNN, ReHDM, KGTB, HA-MTL, MCARNN, Halder 2021/2022,
Ye 2013, moura2025, sun2025) — all real, all faithfully described.

**Result: one real mis-citation (F-1), otherwise SUPPORTED across the board.** Descriptions are
faithful and appropriately hedged; the fundamentals chapter is notably careful (e.g. the Song
93%-predictability site explicitly states it is "not a ceiling on seven-class category macro-F1
or region ranking"). Verb-test discipline holds where checked (Wilcoxon licenses "outperforms";
TOST licenses "matches").

## PART F — build-level render check (L4)
`main_defense.pdf` (87 pp) extracted and scanned: **0 `[?]`, 0 `??`, 0 raw `\cite`, 0
"undefined"**. Citations render as parenthetical `(n)` marks (abntex2-num style); 97/99 numbers
observed spanning 1-99 (the 2 not matched are a line-break extraction artifact, not missing
entries — BibTeX emitted 99 `cite$` and the key-level check found 0 orphans). Bibliography is
fully wired to the text. The Viegas raw-key defect class is ABSENT.

---

## TOP 3 FINDINGS
1. **F-1 (MAJOR):** Ch.4 line 226 cites `liu2014geographical` (a CIKM-2014 location-recommendation
   *method* paper) as the source of "the Gowalla dataset" — a semantic mis-source. Inherited
   verbatim from the CoUrb original; the same chapter cites the dataset correctly elsewhere.
2. **F-2 (MINOR):** `luca2021mobilitysurvey` year 2021 (bib) vs 2023 issue-year (vol 55(1)) —
   an online-first vs issue-year convention mismatch; not wrong, but state one convention.
3. **F-3 (MINOR):** `ruder2017sluice` uses the arXiv-v1 title "Sluice Networks…"; the work was
   renamed "Latent Multi-task Architecture Learning" (AAAI 2019). Same authors/ID; a title-of-
   record choice worth a note, not an error.

---

## RANKED FINDINGS

### F-1 — MAJOR — Wrong source cited for the Gowalla dataset (Ch.4)
- **Location:** `chapters/4_courb.tex:226` (renders in the CoUrb chapter, results section).
- **Quote:** "The experiments were conducted with the Gowalla \emph{dataset}
  \cite{liu2014geographical} in the states of Florida, California, and Texas".
- **Defect:** `liu2014geographical` = Liu, Wei, Sun, Miao, "Exploiting Geographical Neighborhood
  Characteristics for Location Recommendation," CIKM 2014 (DOI 10.1145/2661829.2662002, verified
  this session) — a recommendation-*method* paper, NOT the source of the Gowalla dataset. The
  canonical dataset sources (`cho2011gowalla`, `jure2014snap`) are cited correctly in the SAME
  chapter at lines 18 and 33, so line 226 is both a semantic mis-source and internally
  inconsistent.
- **Origin:** inherited from the CoUrb original (`articles/CoUrb_2026/src_en/sections/results.tex:7`
  cites `\cite{10.1145/2661829.2662002}` for the same sentence). The merge faithfully renamed the
  slash-key to `liu2014geographical` and preserved the wrong attribution — an R2 defect that
  survived because the merge verified the *entry* (which is a real paper) not the *site*.
- **Suggested direction (NOT applied):** replace with `\cite{jure2014snap}` (matching line 33) or
  `\cite{cho2011gowalla,jure2014snap}` (matching line 18). Author rules; if the errata policy is
  "reproduce verbatim + errata note," this belongs in Appendix B as a corrected-in-dissertation
  item. Either way, remove `liu2014geographical` if it then becomes uncited (it would orphan).

### F-2 — MINOR — luca2021mobilitysurvey year vs issue-year
- **Location:** bib entry `luca2021mobilitysurvey`; cited in Ch.1, Ch.2, and Ch.5.
- **Quote (bib):** `year = {2021}` … `volume = {55}, number = {1}`.
- **Detail:** OpenAlex/ACM record the ACM Comput. Surv. 55(1) issue as 2023 (online-first 2021,
  DOI 10.1145/3485125 — resolves, correct). Title/authors/DOI all match; not a fabrication.
- **Suggested direction:** either keep 2021 with the online-first understanding, or align to the
  issue year 2023; whichever, be consistent with how other online-first entries are dated.

### F-3 — MINOR — ruder2017sluice title of record
- **Location:** bib entry `ruder2017sluice` (arXiv:1705.08142). Cited Ch.3:10,17.
- **Detail:** bib title "Sluice Networks: Learning What to Share Between Loosely Related Tasks"
  is the arXiv-v1 title; the work was retitled "Latent Multi-task Architecture Learning" and
  published at AAAI 2019. Same four authors (Ruder, Bingel, Augenstein, Søgaard), same arXiv ID
  (verified this session). The chapter prose calls it "Sluice networks," so the v1 title is the
  intended reference and is defensible.
- **Suggested direction:** optional — add the AAAI 2019 version of record, or keep the arXiv
  preprint (consistent with several other arXiv-only MTL entries). No action required.

### F-4 — MINOR — Two Halder works, verify the descriptions are not crossed (Ch.3 vs Ch.4)
- **Location:** `chapters/3_cbic.tex` cites `Halder2021`; `chapters/4_courb.tex` cites `Halder2022`.
- **Detail (VERIFIED this session — NOT a duplicate):** `Halder2021` = "Transformer-Based
  Multi-task Learning for Queuing Time Aware Next POI Recommendation," PAKDD 2021 (DOI
  10.1007/978-3-030-75765-6_41). `Halder2022` = "POI Recommendation with Queuing Time and User
  Interest Awareness," Data Mining and Knowledge Discovery 36:2379-2409, 2022 (DOI
  10.1007/s10618-022-00865-w). Both real, both correctly attributed, both by the same author team.
  Ch.4's description ("multi-task model based on Transformers to recommend the next POI and predict
  the waiting time") maps to the 2022 journal paper's abstract (queuing time + user interest,
  attention-based) — accurate. Ch.3's "TLR-M … Transformer … next POI + queue waiting time" maps
  to the 2021 paper — accurate.
- **Suggested direction:** none required; flagged only so the author is aware two near-identical
  entries coexist by design. If one was intended to be cited in both chapters, decide which.

### F-5 — NIT — chen2020modeling (HMRM) print-year convention
- **Detail:** bib `year = {2022}` (TKDE vol 34(4) print issue) vs OpenAlex 2020 (online-first,
  DOI 10.1109/TKDE.2020.3001025). The R4 HMRM erratum is correctly applied (5 authors incl. Yang
  Liu, vol 34(4):1902-1914 — OpenAlex exact match). The 2022 print year is the correct issue
  year; consistent with how the bib dates other early-access-then-issued IEEE works.
- **Note for author:** BIB_MERGE_REPORT §3.2 already flags that CBIC ERRATA #2's table row still
  shows the pre-correction vol/pages/year and omits Yang Liu — **the ERRATA registry lags the bib**;
  correct the registry, not the bib.

### F-6 — NIT — kendall2018uncertainty author order (OpenAlex quirk, bib is right)
- **Detail:** OpenAlex lists this work's authors Cipolla-Gal-Kendall; the bib and the universal
  "Kendall et al." attribution use Kendall-Gal-Cipolla, the published CVPR-2018 order. **Bib is
  correct**; recorded so a future automated re-check does not re-flag it.

### F-7 — NIT — 10 "record-only" entries lack a DOI/arXiv string in-entry
- **Entries:** chen2018gradnorm, holm1979, kurin2022scalarization, liu2023famo, nash,
  pedregosa2011sklearn, senushkin2023aligned, velickovic2019deep, xin2022domtl, yu2020pcgrad.
- **Detail:** all 10 carry a full venue+year(+pages) record and were confirmed real by
  title+author against OpenAlex/arXiv this session. Several have a resolvable identifier that is
  simply not written into the entry (e.g. senushkin2023aligned → DOI 10.1109/cvpr52729.2023.01923;
  nash → arXiv:2202.01017; chen2018gradnorm → arXiv:1711.02257; xin2022domtl → arXiv:2209.11379;
  velickovic2019deep → arXiv:1809.10341, noted in the comment). Adding the id would harden the
  entry against R1, but none is fabricated or unresolvable — no gate risk.
- **Suggested direction:** optional hardening; add the arXiv id/DOI where the provenance comment
  already names it.

---

## WHAT HOLDS / WHAT READS WELL (do not touch)
- **The bibliography is fundamentally sound.** 99/99 entries are real, correctly attributed,
  non-retracted works. Zero fabrications — the single highest risk in a literature-review
  dissertation is absent. This is the headline.
- **Mechanical closure is perfect:** 99 cited keys = 99 entries, 0 dangling, 0 orphan, 0 dup
  keys; BibTeX `warning$ -- 0`; the built PDF has 0 unresolved markers. The key-consolidation
  work (DGI triple-key, Nash-MTL double-key, Cho triple-key, etc.) landed cleanly.
- **All five R4 inherited errata are FIXED** in the bib (POI-RGNN, HMRM authors, GAT→ICLR,
  silva2025mtlnet venue+status, paiva2026 3rd author) — verified against sources of record.
- **The provenance trail is exemplary** and internally consistent: every entry carries a
  `% PROVENANCE` line; 60 carry a recorded supporting claim; the comment-precedes-entry
  convention holds without a single off-by-one (checked mechanically).
- **Description fidelity is high across 178 sites.** MTL optimizers, graph/representation methods,
  and mobility systems are each described the way their authors describe them. The fundamentals
  chapter's hedging is careful and honest.

## OPEN QUESTIONS (author only)
1. **F-1:** which dataset key should replace `liu2014geographical` at Ch.4:226 —
   `jure2014snap` (matches line 33) or `cho2011gowalla,jure2014snap` (matches line 18)? And does
   it go in the errata appendix as a corrected-in-dissertation item?
2. **F-4:** is citing two different Halder works (2021 in Ch.3, 2022 in Ch.4) intended, or should
   one be used consistently?
3. Errata-registry lag (BIB_MERGE_REPORT §3.2, §3.11): CBIC ERRATA #2 (HMRM row) and the
   `standley2020tasks` venue defect should be corrected in the ERRATA file — out of my
   read-only scope; handed to the author.

## OUT-OF-SCOPE HANDOFFS (one line each)
- **Numbers (persona 06):** the CoUrb chapter's "20.2 to 22.0 pp" (line 33) vs the original's
  "20 to 24 pp" — this is the sanctioned audited-number substitution (NORTH_STAR §4); a number
  auditor should confirm it traces to `slides/judge_feedback.md`. Not a citation issue.
- **Claim honesty (persona 07):** Ch.2:59 "MTLnet … does not outperform the dedicated single-task
  models, a result that holds for that configuration" — correctly time-indexed; flagging for the
  honesty auditor's verb-test pass, not a citation defect.

## COVERAGE STATEMENT
- **Entry level (R1/R2): 100%** (99/99) resolved against a source of record this session
  (OpenAlex get_work ×63 DOI, arXiv get_papers ×24, OpenAlex/arXiv title+author search ×12).
- **Claim-support (R3): 73%** of sites (178/243); 100% of Ch.1+Ch.2 (frame + fundamentals, the
  new-this-pass and highest-AI-share material); every distinct key-set site in Ch.3/4/5.
- **R4: 100%** (all 5 inherited errata verified fixed). **R5: 100%** sweep, clean.
- **Not audited:** ~65 duplicate-key-set repeat sites in Ch.3/4/5 (same key, same claim class,
  already covered by the first occurrence).

