# N4 + R3 FACT-GATE REPORT — assembled dissertation v1

> Fresh-eyes audit (AGENT_GUARDRAILS L6), 2026-07-23/24. Auditor drafted none of the text under
> review; read-only (this report is the only file written). Scope: the nine chapter files in
> `articles/dissertacao/src/chapters/` + `src/0_main.tex` front matter + `src/references.bib`.
> Protocol: reviewers/README.md Common protocol; AGENT_GUARDRAILS §1 (R1–R5) and §2 (N1–N5).

## VERDICT: **FAIL (conditional)** — 3 BLOCKER-class items, none a new fabrication

The document's numbers are in exceptionally good shape: every table is byte-identical to its
source of record, every prose numeral traces to a ledger line, and no MISMATCH exists anywhere.
The three blockers are (1) the three *declared* dataset placeholders in Ch.3 that the chapter's
own ledger marks as blocking handoff, and (2)–(3) two claim-support defects **inherited verbatim
from the published CBIC/CoUrb texts** — under the errata policy (NORTH_STAR §5.7) these route to
an author decision + Appendix B listing, not a silent fix. No sampled citation is fabricated;
all 99 bib entries resolve; attribute fidelity of the 19 identifier-gained entries checks out.

---

## Part A — N4 numeral-extraction audit

### A.1 Method

Scripted extraction over the nine chapter files: LaTeX comments stripped (unescaped `%` to end
of line), arguments of `\cite/\ref/\label/\url/\includegraphics/...` masked, every remaining
numeral+unit captured with context. Table cells verified separately as **ordered numeral
sequences** against the source-of-record table files (not just multisets). Front matter
(Resumo/Abstract in `0_main.tex`) extracted the same way.

### A.2 Counts

| File | Numerals | Classification |
|---|---|---|
| 1_introduction.tex | 14 | 14 MATCHED (ledger `storyline/drafts/1_citations.md` + venue verifications of 2026-07-23) |
| 2_fundamentals.tex | 5 | 5 MATCHED (`fundamentals/DRAFT_LEDGER.md`; 93% verified firsthand there) |
| 3_cbic.tex | 355 | 352 MATCHED + **3 ORPHAN-BY-DESIGN placeholders (BLOCKER, declared)** |
| 4_courb.tex | 361 | 361 MATCHED (incl. 4 DECLARED-AUDITED values per ERRATA #1/#2) |
| 5_mobiwac.tex | 371 | 371 MATCHED |
| 6_conclusion.tex | 19 | 19 MATCHED (2 DECLARED-ROUNDING; 1 declared-partial value, see A-5) |
| apx_a_contributions.tex | 4 | 4 MATCHED (dates; owner list CLAUDE.md §1) |
| apx_b_errata.tex | 60 | 60 MATCHED (restate ERRATA/judge_feedback/Crossref records; spot-checked) |
| apx_c_ai_disclosure.tex | 3 | 3 MATCHED (CNPq Portaria 2.664/2026; AGENT_GUARDRAILS §6) |
| **Total** | **1,192** | **0 MISMATCH; 3 ORPHAN (declared placeholders)** |

Front matter (Resumo + Abstract): numerals {2026, 5.3, 9.4, "twenty repetitions (four random
initializations, five folds)"} — consistent with Ch.1/Ch.6 exactly; 5.3/9.4 is the
NORTH_STAR-sanctioned rounding of 5.33/9.35 (declared; see A-4). PT/EN pair carries the same
numbers with the same hedges (claim parity holds).

### A.3 Table verification (source-of-record, cell-by-cell, ordered)

| Chapter table | Source of record | Result |
|---|---|---|
| tab:cbic:category (126 cells) | `articles/CBIC___MTL/tables/category_result.tex` | ordered-identical; bold set identical (21 bold cells; HMRM never bolded, as published — caption states the convention, per ledger B7) |
| tab:cbic:next (126 cells) | `articles/CBIC___MTL/tables/next_result.tex` | ordered-identical; bold + underline sets identical |
| tab:cbic:convergence (9 cells) | `articles/CBIC___MTL/tables/converge_result.tex` | ordered-identical |
| tab:courb:dataset (9 cells) | `articles/CoUrb_2026/src_en/resultados/tabela_dataset.tex` | ordered-identical |
| tab:courb:category (126 value cells) | `.../tabela_comparativa_f1_category.tex` | ordered-identical; 21 bold cells identical (the only sequence diff is a header-layout token: source `\multicolumn{2}` vs chapter's `\multirow{7}` — layout, not data) |
| tab:courb:next (126 value cells) | `.../tabela_comparativa_f1_next.tex` | ordered-identical; 21 bold cells identical, incl. the FL-Outdoors baseline bold kept as published (ledger A4) |
| tab:mobiwac:datasets (54 cells) | `articles/[mobiwac]/src/tables/tbl1_datasets.tex` | ordered-identical; caption verbatim |
| tab:mobiwac:representation (32 cells) | `.../tbl2_substrate.tex` | ordered-identical; caption + coincidence footnote verbatim |
| tab:mobiwac:results (96 value cells) | `.../tbl3_results.tex` | ordered-identical (diff is only the booktabs `\cmidrule{3-6}/{7-11}` tokens replacing the source's `\multicolumn{11}` footnote row wrapper); footnote text verbatim; bold/`\sd`/↑/≈/†/‡ all verbatim |

Ch.5 deep trace (N1 chain): the tbl3 joint cells (63.32/64.51/65.79/79.84/77.24/77.05; region
75.35/69.70/59.46/77.41/67.06/65.69) were re-found in
`docs/studies/closing_data/joint_best/JOINT_BEST_RESULTS.md`; dedicated ceilings (54.74/56.82/
56.43/74.51/69.79/70.60; 75.16/70.11/59.46/76.70/64.95/63.49) in
`docs/studies/closing_data/v17_completion/CEILINGS_N20_FINAL.md`; HMT-GRN row in
`docs/baselines/next_region/{hmt_grn,comparison}.md`; ReHDM row in `.../rehdm.md` (v4,
version-uniform — the never-cite v2 row 66.06/54.65/65.68 appears nowhere in the chapter);
STAN row incl. TX 4/5 and CA 2/5 partial-fold disclosures in `.../stan.md`; Markov-K and
POI-RGNN rows in `docs/baselines/next_category/comparison.md`; Table 2 arms in
RESULTS_BOARD.md + `v17_completion/h3_istanbul/RESULTS.md` (Istanbul 54.65/26.56/+28.09).
Never-cite check: no STAN v4-collapse numbers (34.46/38.96), no HMT-GRN AL 62.37 outlier,
no fp16/bf16 cells anywhere in the chapter.

### A.4 Prose-numeral sweeps (exhaustive per chapter)

- **Ch.3**: every prose numeral found verbatim in `articles/CBIC___MTL/sections/*.tex` or the
  sanctioned ERRATA.md values (34.97 s, 2.3×). The only numerals NOT in the published sections
  are 2.315/0.012/0.234 — they are in the published *convergence table* and are quoted into the
  B3-reconciled prose exactly (ledger-sanctioned).
- **Ch.4**: every prose numeral found in `src_en` or in `slides/judge_feedback.md` / ERRATA.md
  (15/21 + 1 tie; 0.02 pp; 20.2–22.0 with the best-of-two-encoders qualifier — the 22.0 is the
  ERRATA-sanctioned rounding of 21.98, declared in the ledger).
- **Ch.5**: all 96 distinct prose numerals found in `articles/[mobiwac]/src/sections/*.tex` /
  `figs/*.tex` except the EDAS number and "2026" in the preface — owned by CLAUDE.md §1
  (EDAS #1571313639 confirmed there). MATCHED.
- **Ch.6**: 20.2/22.0 (NORTH_STAR §2 audited); 5.3–9.4 (NORTH_STAR §2, rounding of the Ch.5
  5.33/9.35 — the rounding is IN the source, not agent-computed); 4.2M/0.6M/56.16/56.82/64.54
  and the CA partial 15-of-20 (capacity_baseline_experiment.md §5.1–5.4, all verbatim; 0.6M =
  644,359, declared); +0.001 cosine with its full scope (verbatim scope in `02_related.tex`);
  freeze control at AL/AZ/FL (`06_results.tex`); 2009–2010 vintage (DATASETS.md / CoUrb
  conclusion). MATCHED.
- **Cross-checks**: abstract-vs-body consistent; captions-vs-table verified (A.3); prose-vs-
  statistic conventions named (macro-F1/Acc@10/TOST margins present at each claim site).

### A.5 N4 findings

**[N4-1] BLOCKER (declared, pre-existing) — Ch.3 dataset placeholders.**
`3_cbic.tex:235`: "comprises a total of [$N_{\text{users}}$; VERIFY: recompute per ERRATA.md]
users, [$N_{\text{poi}}$; ...], and [$N_{\text{checkins}}$; ...] check-ins."
The adaptation ledger (B1/D1) already marks this as **blocking handoff**; the sanctioned path
(repo-committed recompute over the CBIC-era Florida pipeline, author-approved) has not run.
Direction: run the sanctioned script or ship with the placeholders only if the author
explicitly accepts that for the advisor draft. *No values were invented — correct fail-closed
behavior; the blocker is that the gate cannot pass until resolved.*

**[N4-2] MAJOR (declared-partial) — Ch.6 California capacity value.**
`6_conclusion.tex:79`: "A partial California run, fifteen of twenty repetitions at the time of
writing, shows the same direction." Matches the source (job 4cff4b00, seeds {0,1,7} = n=15/20,
68.35 ±0.53, same direction) and the text discloses partiality — but the Ch.6 ledger's own
"must change before final" list requires replacing it with the final n=20 verdict. Direction:
swap in the final value when the job completes; re-run this numeral check then.

**[N4-3] MINOR — Ch.3 prose year vs bib year for HMRM.**
`3_cbic.tex:247`: "introduced by Chen et al. (2020) \cite{chen2020modeling}" — the merged bib
entry (correctly, per Crossref) is TKDE 34(4):1902–1914, **2022**; the rendered reference list
will say 2022 against the prose's "(2020)". Inherited from the published paper (DOI is
10.1109/TKDE.2020.3001025, online-first 2020). Direction: author decision — either "(2020)" →
"(2022)" as a listed erratum, or leave and note in Appendix B.

**[N4-4] NOTE — declared roundings inventory (all sanctioned, none agent-computed).**
5.33→5.3 / 9.35→9.4 (Ch.6:51, Abstract `0_main.tex:258`, Resumo :169; source NORTH_STAR §2);
21.98→22.0 (Ch.4:33, ERRATA #2); 644,359→0.6M (Ch.6:75, capacity file §5.1). Each rounding
exists in its source document; the chapters quote it.

**[N4-5] NOTE — Table 2 caption "seed 0" convention.** `tab:mobiwac:representation` correctly
carries the matched-recipe seed-0×5-fold convention from the source table and does not blur
with Table 3's n=20 convention (N5 check passes).

---

## Part B — R3 citation claim-support sample

### B.1 Sample design and coverage

- Bib universe: 99 entries, 257 `\cite` instances across the six body chapters (Ch.6 cites
  nothing — verified). Every provenance comment parsed.
- **Sampled: 66 keys / 137 unique citing sentences ≈ 198 of 257 instances (77%)** — far above
  the ≥20% floor. Composition: **100% of the 19 entries that gained identifiers this build**
  (BIB_MERGE_REPORT §2), **100% of every entry whose provenance is NOT the MobiWac verified
  donor** (56 entries: CBIC/CoUrb/Ch.2 donors + the `nash` seed entry), plus a random sample of
  12 MobiWac-donor keys. All six citing chapters covered (Ch.6 has no cites by design).
- Method: for each sampled key, the source of record was opened this session — Crossref (DOI),
  arXiv API (id), or OpenAlex (keyed; `api_key` from env, no `mailto`) — attributes compared
  against the bib entry, and each citing sentence judged against the opened
  title/venue/abstract; where the abstract could not settle a mechanism-level claim, the full
  text was fetched when reachable (Time2Vec 1907.05321, DWA 1803.10704, Attentive Single-Tasking
  1904.08918 — all opened and the claims located). One fetch failed at source: MDPI returned 403
  to non-browser clients even after the domain grant; AAAI OJS likewise (proxy handshake).
  Fetch log: `handoff/meta_fetched.json`; per-sentence raw verdicts: `handoff/verdicts_llm.json`
  (screening pass), adjudicated below by the auditor against the actual clause each key anchors.

### B.2 Attribute fidelity of the 19 identifier-gained entries (R1/R2 re-check)

All 19 DOIs/arXiv ids resolve and the records match the bib attributes (spot-confirmed:
Halder2021 = PAKDD 2021 LNCS 510–523 ✔; chen2020modeling = TKDE 34(4):1902–1914 2022, 5 authors
incl. Yang Liu ✔; sener2018mgda NeurIPS 2018 ✔; lakens2017tost SPPS 8(4):355–362 ✔;
perez2018film AAAI 2018 ✔; lin2021ctle AAAI 35(5):4241–4248 ✔; wu2024torchspatial arXiv
2406.15658 ✔; kohavi1995crossval Zenodo re-deposit resolves in OpenAlex with the IJCAI'95
title/author/year ✔ — and its abstract, opened this session, DOES state the ten-fold stratified
recommendation, closing the donor's own residual [VERIFY]). **No attribute defect found.**

### B.3 Adjudicated verdict tally (per unique citing sentence, n=137)

| Verdict | n | Notes |
|---|---|---|
| SUPPORTED | 101 | incl. 6 upgraded from the screening pass after full-text checks (Time2Vec linear+periodic formula located; DWA rate-of-change definition located; Δm average-per-task-drop located; Aligned-MTL condition-number clause verified; Vaswani MHA-for-MT attribution correct; baxter clause re-scoped) |
| PARTIAL (nuance drift) | 22 | listed below, B.5 |
| NOT-SUPPORTED | 2 | **B-1, B-2 below (both inherited published text — BLOCKER-class, errata route)** |
| UNVERIFIED-BLOCKED | 12 | closed-access sources where the abstract cannot settle a mechanism detail (B.6); none contradicted |

### B.4 BLOCKER-class findings (NOT-SUPPORTED)

**[R3-1] BLOCKER (inherited, errata route) — `standley2020tasks` cited for the opposite of its
thesis.** `3_cbic.tex:187` (published CBIC text, reproduced verbatim): "In practice, hard
parameter sharing frequently matches or exceeds the performance of more complex architectures
on many benchmarks, while offering faster training and inference \cite{standley2020tasks}."
The source's abstract (opened, DOI-resolved) argues the reverse: multi-task training "often
leads to inferior overall performance as task objectives can compete," and proposes splitting
tasks across networks. The citation supports neither "matches or exceeds" nor the efficiency
clause. Because this is the published paper's own sentence, the fix is an ERRATA.md + Appendix B
listing (author decision), not a silent edit. Candidate for the CBIC ERRATA list alongside the
existing #11 (`standley2020tasks` venue defect already recorded in BIB_MERGE_REPORT §3.11).

**[R3-2] BLOCKER-leaning / UNVERIFIED-BLOCKED (inherited, both papers) — `Xia2020` (MTPR)
described as LSTM-based.** `3_cbic.tex:102`: "MTPR ... combines LSTMs and adversarial learning";
`4_courb.tex:75`: "MTPR \cite{Xia2020} jointly models location and temporal context through
geographic LSTMs and adversarial learning." The MDPI abstract (opened via Crossref/the
article-fetch tool) describes a **GAN-based** multi-task recommender using temporal check-ins
and geographical locations; it never mentions LSTMs. The full text could not be opened (MDPI
403 to non-browser clients, after a granted domain request), so an internal LSTM component
cannot be excluded — but the abstract gives the "LSTM" attribute no support. Both sentences are
verbatim published text (confirmed against `CBIC___MTL/sections/basis.tex` and the CoUrb PT
original). Direction: author opens the MTPR full text once; if no LSTM, list in both papers'
ERRATA + Appendix B ("LSTMs" → "a GAN-based multi-task architecture").

### B.5 PARTIAL findings (nuance drift; MAJOR→NIT, ranked)

1. **MAJOR — `Xu2023` in Ch.2** (`2_fundamentals.tex:49–50`, frame text, NOT inherited): "A
   fourth, non-sequential task, category classification, labels a POI with its semantic
   category from static features rather than from a sequence \cite{Xu2023}." The TME abstract
   (opened) says existing methods "fail to fully exploit the contextual information in the
   check-in sequences" — TME itself uses sequence context for annotation. The *task-target*
   framing (per-POI label) is fine; "from static features rather than from a sequence" as a
   claim carried by this citation is not. Direction: either drop the cite from that clause or
   reword ("whose target is a per-place label rather than a next event").
2. **MAJOR — `Xu2023` in Ch.3** (`3_cbic.tex:104`, inherited): "TME ... using graph-based
   encoders" — TME is *tree-guided* multi-task embedding (hierarchical category structure), not
   graph-based encoders. ("Treat prediction and classification separately" reads acceptably:
   TME does not do joint next-POI prediction.) Errata candidate.
3. **MINOR — `mikolov2013word2vec` in Ch.4** (`4_courb.tex:175`): "skip-gram strategy with
   negative sampling \cite{mikolov2013word2vec}" — arXiv 1301.3781 introduces skip-gram, but
   negative sampling (SGNS) is the companion paper (Mikolov et al., NeurIPS 2013, "Distributed
   Representations of Words and Phrases"). The ERRATA #6 substitution fixed the wrong-paper
   problem but landed on the half-right Mikolov paper. Direction: cite the NeurIPS 2013 paper
   (or both) for SGNS.
4. **MINOR — `wu2024torchspatial` in Ch.1** (`1_introduction.tex:48`): "first validated on
   geospatial tasks such as species recognition and remote sensing
   \cite{mai2023...,wu2024torchspatial}" — "first validated" is licensed by the Sphere2Vec
   paper; TorchSpatial is a later benchmark, so the pair-cite slightly overstates its role
   ("first"). Same pattern at `4_courb.tex` (two sites, inherited): SIREN/Sphere2Vec-M
   "originally validated" citing the benchmark. NIT-to-MINOR; wording tweak or accept.
5. **MINOR — `ruder2017sluice` in Ch.3** (`3_cbic.tex:186`, inherited): "hard sharing acts as a
   regularizer ... \cite{ruder2017sluice}" — the sluice-networks paper is about *learning* the
   sharing architecture; the regularization claim belongs to `caruana1997multitask`/
   `baxter2000model` (both already cited nearby). Errata-candidate (soft).
6. **MINOR — `chen2020modeling` in Ch.3** (`3_cbic.tex:247`, inherited): "HMRM ... is designed
   for POI category classification" — HMRM is a general trajectory-attribute representation
   model (abstract opened); category classification is the *evaluation use* here. Acceptable as
   baseline framing; consider "used here for POI category classification."
7. **MINOR — `yu2024survey`/`zhang2021survey` "five methodological dimensions"**
   (`3_cbic.tex`, inherited): the five dimensions listed in the CBIC prose do not map 1:1 onto
   either survey's own taxonomy (Zhang & Yang's five categories differ). Survey-paraphrase
   drift; errata-candidate (soft).
8. **MINOR — `sun2024transtarec` in Ch.4** (inherited): "unites representations of the POI, the
   timestamp, and user preferences" — the translation triplet is (timestamp, user, next-POI);
   close but attribute-imprecise.
9. **NIT — `rußwurm2024...` "The SIREN model"** (Ch.4, inherited): the paper's contribution is
   spherical-harmonic + SirenNet combinations; calling the whole encoder "SIREN" simplifies.
   The correct origin (`sitzmann2020implicit`) is separately cited in Ch.2/Ch.4.
10. **NIT — `sitzmann2020implicit` in Ch.2**: "the general basis for periodic coordinate
    encoders" — a generalization beyond the paper's own claim; acceptable didactic register.
11. **NIT — `sun2020go` in Ch.4** (inherited): cyclical-regularity claim (meal times) carried
    by an LSTPM cite whose abstract stresses long/short-term preference, not cyclicity.
12. **NIT — `perez2018film` in Ch.3** (inherited): FiLM "aimed at ... mitigating negative
    transfer" — negative-transfer purpose is the CBIC authors' use, not FiLM's own framing
    (the co-cite `standley2020tasks` partially covers task interference).

(Remaining PARTIALs are of the same class or weaker; full per-sentence table in
`handoff/verdicts_llm.json`.)

### B.6 UNVERIFIED-BLOCKED (source unreachable at the needed depth; none contradicted)

- `huang2023hgi` mechanism-level claims (Ch.2 "maximizing MI among the levels"; Ch.5 "match its
  real neighborhood and reject a shuffled one"): ISPRS full text closed-access (Elsevier
  blocked; no arXiv version found). Record verified (ISPRS 196:134–145, 2023); the MobiWac
  donor bib carries a web-verified comment. The mechanism wording matches the DGI/HGI family's
  standard description and the author's own implementations; recommend one author-side
  full-text confirmation, else accept on donor verification.
- `sokolova2009measures` (Ch.2 macro-F1 invariance framing): closed access; donor bib carries
  the claim-level verification comment from the Ch.2 fact gate. Accept on that record.
- `lin2021ctle` "masking and reconstructing parts of a user's check-in sequence" (Ch.5,
  submitted-paper text): AAAI OJS unreachable (proxy). Abstract confirms contextual per-visit
  embeddings + pre-training; the masking detail is unconfirmed this session. Author's own
  paper text; low risk.
- `Halder2021`/`capanema2023poirgnn` sentence-level details beyond their titles: records
  verified via Crossref (PAKDD 2021; Ad Hoc Networks 138:103016) — titles alone already support
  the citing sentences' attributions (queue-time multi-task; next place's category via
  recurrent+graph networks).
- `jure2014snap` (dataset website, @misc): URL not fetchable from the sandbox; entry is a
  dataset-collection pointer — acceptable identifier class for a dataset citation.
- `feng2017poi2vec` "geographic binary tree" detail (Ch.4, inherited): AAAI PDF unreachable;
  abstract confirms geographical-influence latent representation. Binary-tree detail is
  standard POI2Vec description; author-side confirmation recommended.

### B.7 What holds (R3)

- **Zero fabricated references.** All 99 keys resolve; every sampled identifier opened maps to
  the right work with the right authors/venue/year.
- The MobiWac donor discipline shows: all 12 randomly sampled MobiWac-donor keys came back
  SUPPORTED or benign-PARTIAL, and the donor's `% verified:` quotes were accurate every time
  they were checked (moura2025mobilityaware, sun2025kgtb, lim2022hmtgrn, huang2023hgi...).
- The high-risk frame chapters (Ch.1, Ch.2, Ch.6) are clean at the BLOCKER level: Ch.6 cites
  nothing (as designed); Ch.1's eight breadth anchors all check out against their opened
  abstracts (song2010limits 93% wording verified again this session via the Science record —
  "93% predictability across the whole user base"); Ch.2's one defect is the Xu2023 clause
  (B.5-1), already the weakest link its own draft ledger circled.
- The bib merge's errata work (ERRATA #2–#11) is *correct*: every correction spot-checked
  (HMRM 5-author TKDE record; Cross-Stitch .433; zhang2021survey .3070203; MMoE consolidation;
  CBIC/CoUrb self-records with pages 323–336 and 6/4-author lists) matches the sources of
  record opened this session.

---

## Out-of-scope handoffs (one line each)

- Ch.4 chapter-title wording ("Point-of-Interest Representations" vs the paper's
  "Representations of Points of Interest") — open [VERIFY] in the Ch.4 ledger; concordance/
  author question, not a fact-gate item.
- `\:` in the Ch.4 chapter title renders as thin space, not colon (ledger flag) — persona 18.
- Ch.3 figure `cbic_mtlnet_arch.png` is a ~200 dpi bitmap (ledger A8 flag) — persona 18.
- The three benign [VERIFY] caveat comments in the bib (kohavi/wilcoxon/yang2015tsmc) — the
  kohavi one can now be marked closed (abstract opened this session confirms the stratified
  ten-fold recommendation; see B.2).
- MobiWac chapter re-sync obligation before final gate (`[mobiwac]/src/` refined in parallel)
  — process item, ledger already tracks it.

## Open questions for the author

1. B1 placeholders (N4-1): run the sanctioned recompute now, or ship the advisor draft with
   visible placeholders?
2. R3-1/R3-2 and the B.5 inherited PARTIALs: which go into the papers' ERRATA.md + Appendix B
   for this build, and which are accepted as faithful reproduction defects?
3. Xu2023 in Ch.2 (B.5-1): reword the clause or drop the cite? (Frame text — freely editable.)
4. SGNS citation (B.5-3): add Mikolov et al. 2013b (NeurIPS) to the bib?

*Sampling artifacts for independent re-audit: `src/handoff/citation_sample.json` (sample plan),
`src/handoff/meta_fetched.json` (opened records), `src/handoff/verdicts_llm.json` (raw
screening verdicts before adjudication).*
