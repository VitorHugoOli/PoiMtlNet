# G2 Fact Gate + L5 Translation Fidelity — CORRECTED v1 (round-2 corrections applied)

> **Gate:** DISSERTATION_FACT_GATE (personas 05 citation / 06 number / 07 claim-honesty + 08 translation fidelity).
> **Scope:** read-only. Verifies citations, numbers, and claims only. Does not judge prose style, grammar, domain soundness, or formatting (other reviewers own those).
> **Document under gate:** `articles/dissertacao/src/` (main.tex + chapters/*.tex + references.bib), corrected v1, corrections round 2.
> **Session:** re-derived this session; prior review context (`src_utils/_review_v1/`) was NOT echoed.
> **Fail-closed law:** nothing from model memory. A citation is verified only when its identifier resolves against the source of record AND the landing page/PDF was opened this session AND the cited claim was located in it. Numbers trace to the README sources of truth; quote, never compute (reproduce-first where unavoidable).

**Report status: COMPLETE.** All four persona sweeps (05 citation / 06 number / 07 claim-honesty / 08 translation fidelity) executed across all chapters + appendices + bibliography.

---

## OVERALL VERDICT: **GATE FAIL** — one BLOCKER cluster, otherwise clean

The document is in strong shape. The round-2 priority correction (B.1) is verified correct and fully consistent. The gate fails on a **single cluster**: the filled CBIC Florida dataset numbers are arithmetically correct-to-source but rest on an **unconfirmed basis** that contradicts the source-of-truth recommendation, with a stale `[VERIFY]` and a now-false Appendix B row attached. This is a basis-confirmation + documentation-consistency failure, not a fabricated or wrong number. Everything else — citations, other numbers, claims, honesty devices, translation fidelity — passes.

### The trichotomy (per persona 06 output contract)

**MISMATCHES / BLOCKERS (must resolve before advisor):**
1. **[BLOCKER] CBIC Florida dataset basis unconfirmed + contradicts source of truth** (`3_cbic.tex:238`, §2). The rendered numbers (raw 21,052 / 76,544 / 1,407,034; filtered 13,935 / 76,266 / 1,392,262) reproduce EXACTLY from `data/checkins_by_state/Florida.parquet`, but the source of truth `src_utils/cbic_recompute_result.md` carries verdict `[VERIFY-still-needed]` and recommends a DIFFERENT (CBIC-era filtrado) basis, calling the 2026 fresh-ETL numbers "not recommended for a CBIC chapter." The in-file `[VERIFY]` sign-off is open. **Author must confirm the basis choice.**
2. **[MAJOR] Appendix B Table B.1 row 5 is now factually wrong** (`apx_b_errata.tex:74-80`, §2). It still states the CBIC dataset values are "Pending ... the chapter renders visible placeholders" — but Ch.3 now renders actual numbers. Direct self-contradiction between Ch.3 and Appendix B. Update the row once the basis is confirmed.

**ALL-CLEAR (verified this session):**
- **[PASS] §1 B.1 correction** — matches the CBIC record (no region task; negative transfer hypothesized not observed) and is consistent across Ch.1/3/4/5/6 + Appendix B. **Cross-boundary mirror confirmed:** the fix is present in the version-of-record `articles/[mobiwac]/src/` (01_introduction L17, 02_related L49), no old phrasing survives there, and `articles/[mobiwac]/ERRATA.md` L23-35 logs it accurately.
- **[PASS] §3 CoUrb numbers** — 21/21 category, 15/21 next-POI, 0.02 pp Outdoors-FL tie, 20.2-22.0 pp gains all reproduce from the chapter's own tables and trace to `judge_feedback.md`.
- **[PASS] §3 L5 translation fidelity** — every PT→EN claim-strength difference is a documented Appendix B erratum (16→15 wins; 20-24→20.2-22.0 pp), corrected not silently fixed; dataset table digit-identical to published `src_en`.
- **[PASS] §4 citations** — 0 dangling cites, 0 duplicate keys; 13 entries resolved against source of record this session (incl. both dissertation papers + all R4 errata); R4/R5 clean.
- **[PASS] §5 numbers** — never-cite sweep clean; Ch.5 headline deltas + region 4-of-6 + AZ/AL reproduce from Table 3; cross-chapter headline numbers consistent; capacity-baseline traceable; no `[VERIFY]` in rendered prose except the CBIC item.
- **[PASS] §6 claims/honesty** — verbs bound to tests; AZ never upgraded; time-indexing intact; C3/C4 clean; honesty devices present.

**COULD-NOT-VERIFY (fail-closed disclosure):**
- The **resolution of the CBIC N_users convention** (why 2025 filtrado = 10,460 vs 2026 ETL / CoUrb ≈ 20,301): the recompute doc §5 states this cannot be reconstructed from committed artifacts, and neither can I. This is the author's decision, not a computable fact. It does not block IF the author confirms the basis and the cross-chapter divergence is acceptable/noted (§2c).
- **arXiv Atom metadata** for 2 preprint entries (return-shape error in the connector this session); not material — those works were covered via OpenAlex/Crossref, and no preprint-only entry is load-bearing.

### Minor / nits (batch; do not block)
- `5_mobiwac.tex:93` "this task pair" loose antecedent in the recap (§1) — 2-word disambiguation optional.
- `references.bib` orphan `liu2014geographical` (uncited) — drop or leave (§4a).
- `standley2020tasks` PROVENANCE comment is stale/misleading (names a different paper's AAAI coordinates; claims a DOI the entry lacks) — the rendered entry is correct; fix the comment (§4d nit).
- Category-delta prose max "+9.35" vs table-implied "+9.36" — 0.01 rounding; confirm against unrounded board or use Ch.6's "9.4" (§5b).
- Storyline/`6_citations.md` ledger still records old 64.54 (diag-best) while Ch.6 prose correctly uses 64.51 (joint-best) — sync the non-rendered ledger (§5d).

---

## §1 · PRIORITY 1 — B.1 CBIC misattribution correction (personas 07 + cross-chapter concordance)

**Question (from run brief):** does the B.1 correction now match the CBIC record (Ch.3 has no region task; it hypothesized, did not observe, negative transfer), and is it internally consistent across Ch.1/3/4/5/6?

### What the CBIC record actually is (re-derived from Ch.3 = the reproduced article, this session)

- **Task pair:** POI *Category Classification* (static) + *Next-POI Prediction* (predicting the **category** of the next POI). Confirmed verbatim:
  - `3_cbic.tex:37-38` — the two tasks enumerated: "POI Category Classification" and "Next-POI Prediction: Predicting the category of the next POI".
  - `3_cbic.tex:167` — "hard parameter-sharing scheme for the joint training of POI Category Classification and Next-POI Prediction".
  - `3_cbic.tex:366` — conclusion restates the same pair.
  - **No region task anywhere in Ch.3.** (grep for region/tract/mahalle in `3_cbic.tex` → none as a CBIC task.)
- **Negative transfer is HYPOTHESIZED, not observed:**
  - `3_cbic.tex:43` — "It is possible that forcing a shared encoder ... could result in negative transfer".
  - `3_cbic.tex:45` — "the central hypothesis of this study is that a standard hard parameter-sharing MTL architecture will face significant limitations".
  - `3_cbic.tex:370-375` — "We hypothesize three primary factors contributed to this outcome:" → (1) Subtle Negative Transfer due to Task Dissimilarity, (2) Task Difficulty and Representation Mismatch, (3) Architectural Restrictiveness.
  - `3_cbic.tex:368` — the null itself: "the MTL approach did not consistently demonstrate superior performance over its single-task counterparts".

### The corrected sites (all match the standard)

| Location | Quote (verbatim) | Verdict |
|---|---|---|
| `5_mobiwac.tex:44-45` | "Our earlier work reported no consistent multi-task advantage for the paired category tasks and attributed it, in part, to this effect~\cite{silva2025mtlnet}" | **CORRECT** — "paired category tasks" (not region); "attributed it, in part" mirrors CBIC's "we hypothesize three primary factors", one of which is negative transfer. Not an over-claim. |
| `5_mobiwac.tex:145-148` | "Our earlier work~\cite{silva2025mtlnet} paired static category classification with next-category prediction and found no consistent multi-task gain; this chapter introduces the next-region task ..." | **CORRECT** — the gold-standard corrected statement. This is the "L140" site named in the brief. |
| `apx_b_errata.tex:176-186` (§B.3) | "... describes the prior work of Chapter 3 ... as having studied the next-category and next-region tasks and as having observed negative transfer ... Both statements are inaccurate ... Chapter 3 pairs static category classification with next-category prediction and contains no region task, and it hypothesizes negative transfer to explain a parity null rather than observing it." | **CORRECT** — accurately states the defect and the fix; matches the corrected prose. |
| `1_introduction.tex` (§arc + objectives) | CBIC = "static category classification and next category"; "null result ... together with three candidate explanations, one of which pointed at the input representation"; objective 1 = "two category tasks, static category classification and next category prediction" | **CORRECT** — no region task attributed to CBIC; null + candidate explanations, not observed negative transfer. |
| `6_conclusion.tex:61` (§6.2) | "the joint model of Chapter 3 did not consistently outperform two dedicated models, and negative transfer between the static and the sequential task was among its candidate causes" | **CORRECT** — names CBIC's real pair (static + sequential), frames negative transfer as a *candidate cause*. |
| `6_conclusion.tex` §6.1 | "three candidate explanations, task dissimilarity, an input representation too poor for both tasks at once, and the restrictiveness of hard sharing" | **CORRECT** — the three enumerated match CBIC's three hypothesized factors (L372-374) faithfully. |

**§1 verdict: PASS.** The B.1 correction matches the CBIC record and is internally consistent across Ch.1 / Ch.3 / Ch.5 / Ch.6 / Appendix B. No residual misattribution of a region task to CBIC, and no instance of CBIC "observing" negative transfer, survives in any chapter.

**One MINOR to flag (not a misattribution, a potential misread):**
- `5_mobiwac.tex:93` — "Chapter~\ref{ch:cbic} introduced MTLnet, the first joint model for **this task pair**". In the recap subsection whose framing is the dissertation's category+region pair, "this task pair" has a loose antecedent and could be misread as (category, region). Ch.5 disambiguates fully at L145-148, so this is MINOR, but a two-word fix ("for its two category tasks") would remove the ambiguity. Author's call.

---

## §2 · PRIORITY 2 — filled CBIC Florida dataset counts (persona 06, reproduce-first)

**Question (from run brief):** are the filled CBIC dataset numbers traceable to `src_utils/cbic_recompute_result.md` and carried consistently?

**The numbers as rendered** (`3_cbic.tex:238`): "This subset comprises 21,052 users and 76,544 unique Points-of-Interest (POIs) across 1,407,034 check-ins; after discarding users with fewer than five visits ... 13,935 users and 76,266 POIs across 1,392,262 check-ins remain."

### 2a. Reproduce-first (persona-06 rule 10): both triples reproduce EXACTLY

I re-derived both triples directly from the granted source `data/checkins_by_state/Florida.parquet` (60.6 MB; 1,407,034 rows; single `state_name = Florida`; zero unmapped-category rows):

| Basis | Ch.3 claims (users / POIs / check-ins) | Reproduced this session | Match |
|---|---|---|---|
| Raw | 21,052 / 76,544 / 1,407,034 | 21,052 / 76,544 / 1,407,034 | ✅ exact |
| After `<5-visit-per-user` filter | 13,935 / 76,266 / 1,392,262 | 13,935 / 76,266 / 1,392,262 | ✅ exact |

The `<5` rule was applied as "fewer than five check-in rows per user" (the convention the recompute doc §1 fixes from `core.py:MIN_SEQUENCE_LENGTH=5`). **Numbers are arithmetically correct-to-source (N3 satisfied) and internally consistent with Ch.5 Table 1** (`5_mobiwac.tex:324` FL row = 1,407,034 / 21,052 / 76,544 — identical triple). Traceable ledger present in the source comment (`3_cbic.tex:239-251`).

### 2b. BLOCKER cluster — the *basis* is unconfirmed and contradicts the source of truth

The numbers reproduce, but three fail-closed problems attach to the *basis choice*, not the arithmetic:

1. **The source-of-truth doc recommends a DIFFERENT basis and is not signed off.**
   `src_utils/cbic_recompute_result.md` carries **verdict `[VERIFY-still-needed]` — author sign-off required** (its §Verdict, §5, §7). Its actual recommendation is the **CBIC-era `filtrado.csv` basis: 10,460 users / 64,454 POIs / 960,520 check-ins** (§5). It lists the 21,052/76,544/1,407,034 numbers as "fresh ETL (current repo code + TIGER-2022), raw" and states explicitly (§5, §3): *"that reflects the 2026 mapping, not CBIC-era, so it is not recommended for a CBIC chapter"* and *"The current code therefore cannot reproduce the CBIC-era counts; the committed CBIC-era artifact (filtrado.csv) is the faithful record and is what I recommend."* → The chapter reports the basis the source of truth explicitly recommends against for a chapter that reproduces CBIC.

2. **The in-file [VERIFY] flag is still open.** `3_cbic.tex:239` — "% [VERIFY -- author confirm basis before the advisor build]" and `3_cbic.tex:248` — "% [VERIFY] The three dataset statistics above were never filled in the published paper". So the fill is self-declared provisional; the basis sign-off the recompute doc demands (§7 flags 1 and 2: N_users basis; raw-vs-filtered) is not recorded as resolved. Rendering a still-`[VERIFY]` number as plain fact in the PDF (no visible flag) is the fail-closed hazard.

3. **Appendix B is now internally contradictory with Ch.3.** `apx_b_errata.tex:74-80` (Table B.1, row 5) still states the dataset placeholders are *"Pending. Not invented: the chapter renders visible placeholders; the values await recomputation ..."* This is **now false** — Ch.3 renders actual numbers, not placeholders. A reader comparing Ch.3 to Appendix B sees a direct self-contradiction. **MAJOR** (a factual self-contradiction between two parts of the document — exactly the class persona 06 exists to catch).

### 2c. Cross-chapter tension with CoUrb (Ch.4) — note, not yet a blocker

- Ch.3 (CBIC) FL: 21,052 users / 76,544 POIs / 1,407,034 check-ins (raw); 13,935 / 76,266 / 1,392,262 (filtered).
- Ch.4 (CoUrb) FL (`4_courb.tex:237`): 20,301 users / 65,009 POIs / 990,518 check-ins.

The chosen 21,052-user basis is actually CLOSE to CoUrb's 20,301 (≈+3.7%), so — counter to the recompute doc's original worry (which assumed the filtrado 10,460 basis, a ~2× gap) — the user counts no longer contradict. But the **check-in** counts still differ substantially (Ch.3 raw 1.41M / filtered 1.39M vs Ch.4 0.99M) and **POI** counts differ (76,544 vs 65,009) for "the same" Florida Gowalla data. Each chapter faithfully reproduces its own paper's extraction, so the divergence is defensible IF flagged; there is **no reconciling note in rendered prose** (only a hidden comment at `3_cbic.tex:243-245`). An examiner may ask. → Flag for author: decide whether a one-line cross-chapter note is warranted.

**§2 verdict: GATE FAIL (basis unconfirmed).** The numbers are correct-to-source and Ch.3↔Ch.5 consistent, but (i) the basis contradicts the source-of-truth recommendation and its `[VERIFY-still-needed]` verdict, (ii) the in-file `[VERIFY]` sign-off is open, and (iii) Appendix B still documents the values as pending placeholders. **Author must (a) confirm the basis (fresh-ETL 2026 re-extraction vs recompute-doc-recommended CBIC-era filtrado), (b) clear the `[VERIFY]` once confirmed, and (c) update Appendix B Table B.1 row 5 to reflect that the values are filled.** This is not a claim that the numbers are wrong — they reproduce — it is that the basis decision the source of truth flagged for sign-off has been made silently, against the doc's recommendation, and the documentation is left inconsistent.

---

## §3 · CoUrb (Ch.4) — numbers, claims, and L5 translation fidelity (personas 06 + 07 + 08)

### 3a. Audited numbers reproduce EXACTLY from the chapter's own tables (persona 06, reproduce-first)

I recomputed the win-counts and per-state gains directly from Tables `tab:courb:category` and `tab:courb:next` (best-of-two-encoders vs MTLNet, per the disclosed convention):

| Claim (as rendered) | Locations | Reproduced from table | Verdict |
|---|---|---|---|
| Category: variants beat MTLNet in **all 21** category-state combinations | `4_courb.tex:34,253,344` | 21 of 21 | ✅ exact |
| Category: **average gains per state of 20.2 to 22.0 pp** (best-of-two) | `4_courb.tex:34,253,344`; `6_conclusion.tex:39` | FL 20.24, CA 20.91, TX 21.98 → range 20.2–22.0 | ✅ exact |
| Next-POI: variants beat MTLNet in **15 of 21** combinations | `4_courb.tex:296,344` | 15 of 21 | ✅ exact |
| Next-POI: one **technical tie**, Outdoors-FL, baseline exceeds best variant by **0.02 pp** | `4_courb.tex:296` | MTL 21.61 vs best variant 21.59 → 0.02 pp | ✅ exact |

### 3b. Audit trace to the source of truth (persona 06, N1)

The README §Sources names `articles/CoUrb_2026/slides/judge_feedback.md` as the CoUrb audited win-count/means source. I opened it this session; it states (verbatim, my translation of the PT): strict count by mean gives 15/21 = 71.4%; the ambiguous case is Florida Outdoors (baseline 21.61 vs Sphere 21.59, baseline wins by 0.02 pp within σ). This is the exact correction Appendix B Table B.3 records and the chapter carries. **Traceable and matched.**

### 3c. L5 TRANSLATION FIDELITY — the errata-vs-translation interaction (persona 08)

The published PT paper of record (`articles/CoUrb_2026/src/`) is the L5 source. The claim-bearing sentences DIFFER from the EN chapter, and the difference is the sanctioned post-publication audit correction, not a translation drift:

| PT published (src/) | EN chapter (`4_courb.tex`) | Classification |
|---|---|---|
| intro/conclusion/main: "ganhos médios de **20 a 24** pontos percentuais" | "average gains per state of **20.2 to 22.0** percentage points, considering the better of the two spatial encoders" | **CORRECTED (errata #2), documented** — narrower, audited range + the required best-of-two disclosure. NOT a silent weakening. Appendix B Table B.3 row 2 records it. |
| conclusion (`src/conclusion.tex:3`) + results (`src/results.tex:28`): "vence em **16 das 21** combinações" | "outperforms ... in **15 of the 21** combinations, with one additional technical tie" | **CORRECTED (errata #1), documented** — Appendix B Table B.3 row 1. |
| main.tex abstract: "**76\%** das combinações" | (abstract not reproduced in chapter; body carries 15/21) | Consistent with the correction; the imprecise 76% is dropped, per judge_feedback.md. |
| intro (`src/intro.tex:16`): "vence na **maioria** dos cenários" | "outperforms the baseline in **most** scenarios" (`4_courb.tex:34`) | **FAITHFUL** — maioria → most; verdict-verb rule applied (vence→outperforms). Appendix B Table B.3 row 3. |

**L5 verdict: PASS.** Every claim-strength difference between the PT source and the EN chapter is a documented erratum in Appendix B §B.2 (Table B.3), applied under the settled errata policy — corrected, not silently fixed, and never strengthened. The preface (`4_courb.tex:13`) carries the reproduction statement, the second-author contribution note, and the of-the-time / sample-stratified-split disclosure. The chapter dataset table (`4_courb.tex:237`, FL/CA/TX) is digit-identical to the published `src_en/resultados/tabela_dataset.tex`. No silent omission or addition detected in the claim-bearing prose; the three marked frame additions (preface, MTLnet recap subsection §4.5, sample-vs-user split sentence) are visibly frame material, declared in Appendix B.

**One MINOR (persona 08, terminology landing):** the chapter preserves "MTLNet" (published spelling) while the frame/GLOSSARY canonical is "MTLnet"; this is deliberate and disclosed at `4_courb.tex:139` ("the published paper typesets the name as MTLNet, and this chapter preserves that form"). Acceptable per WRITING_LAW §6 (paper chapters keep the source paper's usage). Not a finding; noted so a future editor does not "fix" it.

---

## §4 · Citation gate (persona 05, R1–R5)

### 4a. Structural integrity (R1 cross-checks) — CLEAN

- **Bib size:** 99 entries. **Cited keys:** 98 unique across all chapters. **Dangling \cite → 0** (every cited key resolves in `references.bib`). **Duplicate keys → 0** (the three DGI spellings / two Nash spellings / two Gowalla spellings named in the CLAUDE.md warning are consolidated — confirmed no collisions survive).
- **Orphan (defined, uncited):** 1 — `liu2014geographical`. This is the CoUrb mis-source key that Appendix B Table B.4 (last row) says was "removed where it no longer applies"; it survives in the bib as an uncited entry. **NIT** — a truly unused entry should be dropped, or it is harmless (BibTeX does not render uncited entries in a numeric style). Author's call.

### 4b. Source-of-record resolution (R1, R2) — 13 entries opened this session

I resolved the following against the source of record this session (Crossref for DOIs; OpenAlex for conference/preprint records). **100% coverage of the two dissertation papers, the R4 errata corrections, and a sample of new-this-pass Ch.2 entries.** All attributes (first author, author count, venue, year, volume, pages) match the bib entry:

| Key | Source of record | Resolved attributes | vs bib |
|---|---|---|---|
| `silva2025mtlnet` | Crossref 10.21528/CBIC2025-1191324 | CBIC 2025, Silva 1st, **6 authors**, pp.1–8 | ✅ (Art. 21 confirmed) |
| `paiva2026stmtlnet` | Crossref 10.5753/courb.2026.22960 | CoUrb 2026, Paiva 1st, 4 authors, **pp.323–336** | ✅ |
| `capanema2023poirgnn` | Crossref 10.1016/j.adhoc.2022.103016 | Ad Hoc Networks, **v.138, 103016**, 2023, Capanema, 5 auth | ✅ (R4 fix) |
| `chen2020modeling` (HMRM) | Crossref 10.1109/TKDE.2020.3001025 | TKDE **v.34, 1902–1914, 2022**, Chen, 5 auth | ✅ (R4 fix) |
| `misra2016cross` | Crossref 10.1109/CVPR.2016.433 | CVPR 2016, pp.3994–4003, Misra, 4 auth | ✅ (R4 DOI fix) |
| `zhang2021survey` | Crossref 10.1109/TKDE.2021.3070203 | TKDE v.34, 5586–5609, Zhang | ✅ (R4 DOI fix) |
| `ma2018mmoe` (MMoE) | Crossref 10.1145/3219819.3220007 | KDD 2018, pp.1930–1939, Ma | ✅ (yu2019mmoe dropped, confirmed absent) |
| `cho2011gowalla` | Crossref 10.1145/2020408.2020579 | KDD 2011, pp.1082–1090, Cho | ✅ |
| `velickovic2019deep` (DGI) | OpenAlex | Deep Graph Infomax, ICLR 2019 (arXiv 1809.10341 = 2018 preprint) | ✅ (triple-key consolidation confirmed) |
| `kendall2018uncertainty` | OpenAlex 10.1109/cvpr.2018.00781 | CVPR 2018, uncertainty weighting | ✅ (new-this-pass) |
| `kurin2022scalarization` | OpenAlex arXiv 2201.04122 | "In Defense of the Unitary Scalarization", 2022 | ✅ (new-this-pass) |
| `huang2023hgi` (HGI) | OpenAlex 10.1016/j.isprsjprs.2022.11.021 | ISPRS J Photogramm, 2023 | ✅ |
| `standley2020tasks` | OpenAlex arXiv 1905.07553 | "Which Tasks Should Be Learned Together", 2020 | ✅ title/year; see NIT below on venue |

### 4c. Claim-support audit (R3) — load-bearing sites SUPPORTED

- `song2010limits` → the "93 percent potential predictability" claim (Ch.1:35, Ch.2:35/435): the bib comment records firsthand-PDF verification (Science 327:1018); the dissertation correctly **scopes** it to next-location at coarse resolution and explicitly refuses to use it as a ceiling on category-F1/region-ranking (`2_fundamentals.tex:435-437`, gate fix B-5). Attribution honest, not strength-drifted. **SUPPORTED.**
- `capanema2023poirgnn` → "POI-RGNN predicts the category of the next place as its output" (Ch.2:94-95): matches the resolved title "Combining recurrent and Graph Neural Networks to predict the next place's category". **SUPPORTED.**
- `xin2022domtl` / `kurin2022scalarization` → "MTL optimizers often do not outperform a well-tuned fixed-weight baseline" (Ch.2:326-329, Ch.5:184): the source titles ("In Defense of the Unitary Scalarization", the DoMTL controlled study) support the hedged claim; the verb is "do not outperform", not a strength upgrade. **SUPPORTED.**
- `standley2020tasks` → "joint training can hurt ... depending on the pairing" / negative transfer (Ch.2:305, Ch.3:167): title "Which Tasks Should Be Learned Together in Multi-Task Learning?" supports it. **SUPPORTED.**

### 4d. R4 errata check — all inherited errata fixed in the global bib

Every Appendix B Table B.4 correction is present in `references.bib` with the corrected values (verified in 4b: POI-RGNN, HMRM, GAT→ICLR2018, Cross-Stitch DOI, MTL-survey DOI, word2vec→mikolov, yu2019mmoe dropped, DGI consolidated, silva2025mtlnet venue+6 authors, paiva2026 author+pages, standley venue, Gowalla mis-source). **R4 PASS.**

### 4e. R5 sweep — CLEAN

No AI output cited as a source; no real-looking citation laundering an unverifiable claim. (The AI-use disclosure in Appendix C describes AI assistance as process, not as a cited source — correct.)

**§4 verdict: GATE PASS (citations).** 1 NIT (orphan `liu2014geographical`), 1 NIT (stale audit-trail comment, below). No fabricated, unresolvable, or misattributed entry found; no unfixed R4 erratum; load-bearing claim-support holds.

**NIT (persona 05, audit-trail hygiene, does NOT render):** the trailing PROVENANCE comment on `standley2020tasks` in `references.bib` reads "DOI added this session (Crossref: AAAI 2020, 34(01):214-221)" — but (i) the entry has **no** `doi` field, and (ii) "AAAI 2020, 34(01):214-221" are the coordinates of a *different* paper (an AAAI proceedings entry), not this ICML work. The **rendered** entry (booktitle = Proc. ICML 2020) follows Appendix B correctly; only the audit comment is wrong/misleading. Recommend correcting or deleting the comment so a future auditor is not misled. (Standley is a known venue-ambiguous work — arXiv 1905.07553, presented ICML 2020; the rendered "ICML 2020" is defensible.)

---

## §5 · Number gate — full extraction cross-checks (persona 06, N1–N5)

### 5a. Never-cite sweep (N-rule 5, absolute BLOCKERs) — CLEAN

Grepped all chapters for every never-cite value (STAN v4-collapse AL 34.46 / AZ 38.96; HMT-GRN AL 62.37 outlier; ReHDM v2 row 66.06/54.65/65.68; fp16/bf16 VOID cells). **Zero hits in rendered prose or tables.** The strings appear only inside the guard comment at `5_mobiwac.tex:468-469` (the do-not-cite list itself). The `54.65` at `5_mobiwac.tex:408` is the legitimate Check2HGI Istanbul representation cell (Table 2), NOT the banned ReHDM-v2 value. **PASS.**

### 5b. Ch.5 headline deltas reproduce from Table 3 (reproduce-first)

Recomputed every joint−dedicated delta from `tab:mobiwac:results`:

| | Istanbul | AL | AZ | FL | TX | CA |
|---|---|---|---|---|---|---|
| Category Δ | +8.58 | +7.69 | +9.36 | +5.33 | +7.45 | +6.45 |
| Region Δ | +0.19 | −0.41 | 0.00 | +0.71 | +2.11 | +2.20 |

- Category range reproduces as **+5.33 to +9.36**. Ch.5:455/539 state "**+5.33 to +9.35**"; Ch.6:51 states "5.3 to 9.4". The **+9.35 vs +9.36** is a 0.01 rounding artifact (the AZ cells 65.79/56.43 are themselves 2-dp rounded; the true delta on unrounded means is what the prose quotes). **MINOR** — within declared rounding, verdict-neutral, but the prose max (+9.35) and the table-implied max (+9.36) differ by 0.01; recommend the author confirm the prose value against the unrounded board (`JOINT_BEST_RESULTS.md`) or state "+9.4" as Ch.6 does. Not a blocker.
- Region: **4 of 6 positive** (Istanbul, FL, TX, CA) ✓; AL −0.41 ✓ (matches Ch.5:560); AZ 0.00 ✓ (matches Ch.5:561, "centered on zero, so we report a match, not a gain" — AZ correctly NOT upgraded); TX +2.11 / CA +2.20 both exceed the 2pp margin ✓ (Ch.5:568-569). Istanbul category delta +8.58 ✓ (matches external-validity claim Ch.5:601). **All consistent.**

### 5c. Cross-chapter headline consistency — CONSISTENT

- "four of six" (region) identical across Ch.1:132, Ch.2:542, Ch.5:67/633, Ch.6:51. ✓
- Ch.5 Table 3 AL dedicated 56.82 / joint 64.51 (joint-best) match Ch.6:77-78 capacity-baseline paragraph exactly (gate fix B-2: 64.54→64.51 to honor N5 joint-best-vs-diag-best; verified the rendered value is 64.51). ✓
- Category floor "about 7 percent macro-F1" (Ch.5:514) and majority-class floor references consistent with the 7-class framing. ✓

### 5d. Ch.6 capacity-baseline (frame-level D1 analysis) — traceable

The post-submission capacity-baseline numbers (Ch.6:75-78: 4.2M vs 0.6M params; 56.16 best wide arm; 56.82 ceiling; 64.51 joint) trace to `storyline/audit/capacity_baseline_experiment.md §5` (56.16 ±1.88, bs2048@lr0.0025, n=20) and the `6_citations.md` ledger. The prose correctly labels it "run after the Chapter 5 manuscript was submitted ... reported here as a frame-level analysis" (C2 new-claim discipline: it is a frame addition, disclosed). **Note:** the storyline source and `6_citations.md` ledger still record the old **64.54** (diag-best); Ch.6 rendered prose uses **64.51** (joint-best) — this is the CORRECT direction (matches Ch.5 Table 3), applied by gate fix B-2. The stale 64.54 in the *ledger/storyline* is not rendered; harmless but worth syncing.

### 5e. `[VERIFY]` / `[NEEDS SIGN-OFF]` flags in RENDERED prose — NONE

Swept all chapters: **zero** `[VERIFY]`, `[NEEDS SIGN-OFF]`, `\todo`, or `FIXME` markers survive in rendered text. All such markers live in LaTeX comments (`%`), which do not compile into the PDF. **The one substantive exception is the CBIC dataset number** (§2): its basis is still `[VERIFY]` in the *comment*, and the number renders as plain fact — that is the §2 finding, not a visible-flag problem.

**§5 verdict: GATE PASS (numbers) EXCEPT the §2 CBIC-basis item.** Never-cite clean; headline deltas, region counts, and cross-chapter headline numbers all reproduce and are consistent; capacity-baseline traceable. The one open number issue is the CBIC Florida basis (§2), which is a basis-confirmation + documentation-consistency BLOCKER, not an arithmetic error.

---

## §6 · Claim & honesty gate (persona 07, C1–C4 + honesty law)

### 6a. Verb-binding (WRITING_LAW §3, the region-wording law) — PASS

- **Banned verdict verbs** ("beats", "wins", "ties", "Pareto" as a results verb): **zero** in any results claim. The only hits are (i) legitimate math ("Pareto-optimal descent", "Pareto-stationary point" in the Nash-MTL formulation, Ch.3:88/98/214), (ii) "ties this structure to" (physical, Ch.5:214), and (iii) "wins"/"16 of 21" inside Appendix B quoting the *published CoUrb defect being corrected* (`apx_b_errata.tex:146/158`). None is a live verdict.
- **"outperforms" scoping:** every region "outperforms" is bound to the four licensed datasets or the scaling claim (Ch.5:64/74/547/633, Ch.1:129, Ch.6:65). Category "outperforms ... on every dataset" is licensed (paired superiority holds everywhere). ✓
- **AZ never upgraded:** no "outperform/beat/win" attaches to Arizona region. AZ region reads as "non-inferior ... (TOST, ±2 pp)" (Ch.5:549) and "centered on zero, so we report a match, not a gain" (Ch.5:561). Alabama (−0.41, a within-margin deficit) is also correctly NOT upgraded. **PASS** on the most-guarded item in the whole document.
- **"matches" bound to TOST:** the phrase "statistically non-inferior within a two-point margin (TOST)" appears in full (Ch.1:132, Ch.5:29/68, defined Ch.5:355, Ch.2:454). ✓

### 6b. Time-indexing (honesty law) — PASS

- CBIC null is time-indexed at every appearance: chapter preface "the conclusions of the time, for the configuration studied here" (`3_cbic.tex:21`); Ch.5 recap "the conclusion of the time for that configuration" (`5_mobiwac.tex:98`); Ch.6 "held for the configuration of its time ... and the later chapters revised it" (`6_conclusion.tex`). ✓
- CoUrb protocol time-indexed: preface "the conclusions reported here are those of the time, for that configuration" + sample-stratified-split disclosure (`4_courb.tex:13`). ✓
- Nash-MTL preference time-indexed and NOT amplified: CBIC preface "a conclusion of the time, weakened by a later finding about the optimizer implementation, and the following chapters do not rely on it" (`3_cbic.tex:25-27`); the verbatim "consistently yielded a better overall performance" is preserved in-body (`3_cbic.tex:230`) with the preface caution, per the settled errata policy (Appendix B preservation note). The frame does not amplify Nash-MTL's benefit (GLOSSARY §2 caveat honored). ✓

### 6c. C4 BRACIS containment — PASS

Appendix A cites BRACIS only as "an earlier unpublished iteration" (`apx_a_contributions.tex`), states its region-cost claim ONLY as corrected history ("its central claim, that multi-task learning imposes a cost on region prediction, was later corrected by the MobiWac study"), quotes no rejected-paper numbers, and confirms "no result of the manuscript is cited as evidence anywhere". The "substrate" repo-codeword appears only inside the verbatim BRACIS title (a proper name), disclosed in the source comment. BRACIS appears nowhere else in the rendered chapters. **PASS.**

### 6d. Honesty devices intact (evidence-guard, anti-dilution) — PRESENT

- Reference points carried: majority-class floor (Ch.5:514 "about 7 percent macro-F1"), Markov floors (Ch.5:585-587), dedicated ceiling as the operative reference (Ch.2:438). ✓
- Uncertainty stated: fold-sd / cross-seed sd on every mean in the results tables; CIs on the region TOST/superiority claims (Ch.5:560-569). ✓
- Leakage hygiene sentence present and concrete: the integrity-of-representation paragraph (Ch.5:343), the per-fold region-transition prior disclosure, the A4 audit null. ✓
- Limitations concrete (Ch.6 §6.3: "Gowalla check-ins collected in 2009 and 2010"; the six enumerated limits including the task-pair confound). ✓
- Negative-result care: the CBIC null is written plainly as the arc's foundation, not rushed (Ch.1 arc, Ch.3 conclusion, Ch.6 §6.1). ✓
- CoUrb ownership honesty: second-author contribution note present (`4_courb.tex:13`, Ch.1:organization). ✓

### 6e. New-claim control (C2) — flags, not judgments

The frame carries arc-narrative sentences that are claims. All are marked `[NEEDS SIGN-OFF]` in-source (the recap subsections, the prefaces, the capacity-baseline paragraph, both appendices' frame prose). These are correctly routed for author sign-off, not silently inserted. **The B.1-correction sentences (§1) are NOT new claims — they are corrections of a submitted-text error, documented in Appendix B, and match the CBIC record.** The capacity-baseline paragraph (Ch.6, D1 analysis) is a genuine NEW post-submission empirical claim; it is disclosed as such and its number traces (§5d). It needs the author sign-off it is already flagged for.

**§6 verdict: GATE PASS (claims/honesty).** Verbs bound to tests; AZ never upgraded; time-indexing intact; C3 never-cite and C4 BRACIS containment clean; honesty devices present. New-claim sentences are flagged for sign-off, not smuggled.

---
