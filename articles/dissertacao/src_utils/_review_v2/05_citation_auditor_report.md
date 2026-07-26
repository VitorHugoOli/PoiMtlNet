# 05 · Citation auditor — citation-integrity gate (G2, R1–R5)

**Build audited:** `src/dissertacao.pdf` (94 pp) + `src/references.bib` (99 entries) at
2026-07-25 23:43. **Date:** 2026-07-26. **Persona:** `reviewers/05_citation_auditor.md`. Read-only.
Nothing below comes from model memory, including my own recall of any reference: every attribute
was checked against a source of record this session (Crossref API, arXiv API, and the bib's own
recorded provenance), and I say plainly where I could not check.

## Verdict

**GATE FAIL** — on one item, and it is a build defect rather than a fabrication.

The bibliography itself is in good order: no fabricated entry, no unresolvable DOI in the sample, no
attribute corruption beyond a normalization artifact, and the round's two touched entries both
verify against two independent sources of record. The failure is that **one cited work does not
appear in the rendered reference list at all**, because a BibTeX parse error silently dropped it,
and the citation renders as `(??)` on four pages of the defense build.

That is R1's failure mode by a different route: the reader cannot resolve the reference, so for the
reader it does not exist.

## Top 3 findings

1. **R-01 (BLOCKER)** — `russwurm2024geographiclocationencodingspherical` is cited four times and
   missing from the reference list; renders as `(??)` on pp. 21, 45, 49, 50.
2. **R-02 (MAJOR)** — `ruder2017sluice` cites a title arXiv no longer carries; the work was retitled
   and published at AAAI 2019.
3. **R-03 (MODERATE)** — a stale ledger comment in Chapter 2 still recommends the Zenodo DOI that
   this round correctly dropped.

---

## R-01 · BLOCKER · A cited reference is absent from the rendered bibliography

**The rendered evidence.** Defense build, p. 45:

> "The SIREN methodology (Sinusoidal Representation Networks) (36), applied to the geographic
> context (??), models continuous functions through sinusoidal activations"

Four `(??)` renders in the defense build (pp. 21, 45, 49, 50) and four in the final build
(pp. 16, 40, 44, 45).

**The cause.** `src/references.bib:831`, inside the provenance comment block for the entry, reads:

> `% (same class as the GAT erratum above, Appendix B): the donor entry was typed @misc as an`

BibTeX's log records:

> "I was expecting a `{' or a `('---line 831 of file references.bib
> : % (same class as the GAT erratum above, Appendix B): the donor entry was typed @misc
> I'm skipping whatever remains of this entry
> Warning--I didn't find a database entry for "rußwurm2024geographiclocationencodingspherical""
> — `src/build/main.blg`

The bare `@misc` inside a `%` comment line is read by BibTeX as the start of a new entry. BibTeX
comments are not line-based the way LaTeX's are: outside an entry, `%` is not a comment character at
all, and `@misc` triggers entry parsing. The parser then skips the real
`@inproceedings{russwurm...}` that follows at `:849`.

Two consequences confirmed by measurement:

- `main.bbl` contains **97** `\bibitem`s; the chapters cite **98** distinct keys; `references.bib`
  holds **99** entries (98 cited + 1 orphan). The arithmetic closes exactly on this one dropped
  entry.
- The reference list (pp. 80–83) contains no Russwurm entry: I searched the rendered text for
  `RUSSWURM`, `RUßWURM`, `Rußwurm` and `Spherical Harmonics` — all absent.
- `pdflatex` reports "LaTeX Warning: There were undefined references" (`src/build/main.log`).

**Note on the stated build state.** The task brief records "0 errors, 0 undefined refs/cites". The
current logs do not support that: `main.log` carries four `Citation ... undefined` warnings plus the
summary warning, and `main.blg` carries one error. I report this as measured, per fail-closed.

**The correction** (specified, not applied, per this persona's hard limits): the `@misc` token
inside the comment at `:831` must be neutralized — most simply by rewording to "typed as a `misc`
entry" or by breaking the `@`. The entry's *content* is correct and verified (see below); nothing
about the reference itself needs changing.

---

## R-02 · MAJOR · `ruder2017sluice` — the cited title is a superseded arXiv version

**Bib entry** (`references.bib:823-828`):

> `title = {Sluice Networks: Learning What to Share Between Loosely Related Tasks}`
> `journal = {arXiv preprint arXiv:1705.08142}`, `year = {2017}`

**Source of record** (arXiv API, id 1705.08142, queried 2026-07-26):

- current title: **"Latent Multi-task Architecture Learning"**
- authors: Sebastian Ruder, Joachim Bingel, Isabelle Augenstein, Anders Søgaard (matches the bib)
- published 2017-05-23; **updated 2018-11-19**; comment: **"To appear in Proceedings of AAAI 2019"**

The v1 preprint was titled "Sluice Networks…"; the paper was retitled and published at AAAI 2019.
The bib cites the v1 title as an eternal arXiv preprint, so a reader searching the title finds a
version that no longer exists under that name, and the venue of record (AAAI 2019) is missing.

**Provenance note in the bib** reads only: "PROVENANCE: articles/CBIC___MTL/references.bib, carried
verbatim." So this is an inherited donor entry that has not been re-verified — R2's exact hazard
("copied from the source of record, not retyped from another paper's bibliography").

**Claim-support at the three citing sites** — all three survive the retitling, because they cite the
work for its architecture and its negative-transfer point, both of which the published version
carries:

| Site | Sentence | Supported? |
|---|---|---|
| `3_cbic.tex:104` | "networks exchange information through learned cross-connections, such as Cross-Stitch units \cite{misra2016cross} or Sluice networks \cite{ruder2017sluice}" | **SUPPORTED** — the abstract describes learning "the layers or subspaces that benefit from sharing, (b) the appropriate amount of sharing"; sluice networks are the paper's named model |
| `3_cbic.tex:112` | "Unrelated or adversarial tasks can degrade shared representations \cite{ruder2017sluice,zhang2021survey}" | **SUPPORTED** (jointly cited) |
| `3_cbic.tex:209` | "By constraining the hypothesis space, hard sharing acts as a regularizer \cite{ruder2017sluice}" | **SUPPORTED** |

*Correction specified:* re-type against the AAAI 2019 record — title "Latent Multi-task Architecture
Learning", venue AAAI 2019, arXiv:1705.08142 as a note, per the GAT/Russwurm precedent already used
in this bibliography. The author may prefer to keep the "Sluice Networks" name visible in prose,
which is fine: the *prose* names the model, the *entry* should name the paper.

---

## R-03 · MODERATE · Stale ledger comment contradicts this round's Kohavi erratum

The round dropped the Zenodo DOI from `kohavi1995crossval` and recorded why, at length and
correctly (`references.bib`, entry comment):

> "ERRATUM applied 2026-07-25: the former doi = {10.5281/zenodo.19712698} was DROPPED. That DOI is a
> third-party Zenodo re-deposit, not the IJCAI-95 record: DataCite ... gives publisher "Zenodo",
> registered 2026-04-23 ... IJCAI-95 genuinely predates DOIs, so the entry carries none."

But Chapter 2's citation ledger still says the opposite, in two places:

> "%  kohavi1995crossval    stratified k-fold CV (VERIFIED by author). [new bib; Zenodo DOI
> 10.5281/zenodo.19712698 for a resolvable id]"
> — `src/chapters/2_fundamentals.tex:524`

> "kohavi1995crossval: claim PLAUSIBLE (Zenodo re-deposit id), author to confirm original IJCAI-95
> text."
> — `src/chapters/2_fundamentals.tex:561`

These are LaTeX comments, so nothing renders. But the ledger is the artifact a future agent or the
author reads to decide whether an entry is settled, and it currently recommends re-adding the DOI
the round deliberately removed, and marks as "author to confirm" a claim the round confirmed
against the IJCAI proceedings PDF.

*Correction specified:* update the two ledger lines to point at the entry's erratum note.

---

## Entry-level audit

**Coverage: 38 of 99 entries (38%),** chosen as 100% of the entries this round touched or that carry
load-bearing methodological claims, plus a seeded random sample (`random.seed(7)`) of 26 from the
remainder. This exceeds R3's ≥20% floor and covers 100% of the round's new/changed entries.

### Entries the round touched — 100% coverage, both verified

**`russwurm2024geographiclocationencodingspherical`** — verified against **two** independent sources
of record:

- arXiv API, id 2310.06743: title "Geographic Location Encoding with Spherical Harmonics and
  Sinusoidal Representation Networks"; five authors Marc Rußwurm, Konstantin Klemmer, Esther Rolf,
  Robin Zbinden, Devis Tuia; `journal_ref` = **"Published as a conference paper at ICLR 2024"**;
  comment "Camera-ready version"; v2 dated 2024-04-15.
- the entry's own recorded OpenReview check (submission ICLR.cc/2024/Conference/Submission3806,
  note id PudduufFLa, venue "ICLR 2024 spotlight").

Bib fields (`booktitle = {Proc. ICLR}`, `year = {2024}`, `note = {arXiv:2310.06743}`,
`url = openreview.net/forum?id=PudduufFLa`) match on every attribute. **The re-typing from @misc
preprint to ICLR 2024 inproceedings is correct and well evidenced.** The ASCII key rename is also
correct: the former key carried U+00DF, and I confirmed the four citing sites were all updated
(`2_fundamentals.tex:211`, `4_courb.tex:65`, `:129`, `:148` — zero occurrences of the ß-key remain
in the chapter sources). **Status: OK.** The entry is right; only the neighbouring comment breaks
the build (R-01).

**`kohavi1995crossval`** — no DOI (correctly, per R-03). The entry's provenance note records the
IJCAI proceedings PDF opened at `ijcai.org/Proceedings/95-2/Papers/016.pdf`, running footers
"KOHAVI 1137" through "KOHAVI 1143" confirming pages 1137–1143, volume 14, sole author Ron Kohavi.
Bib fields match. Claim located at Section 6, p. 1143 ("We recommend using stratified ten fold
cross-validation for model selection") — which supports the citing sentence
(`2_fundamentals.tex:485`, "Estimates use stratified k-fold cross-validation \cite{kohavi1995crossval}").
I could not open the IJCAI PDF myself this session; the entry's recorded verification is detailed,
internally consistent, and names page-level evidence. **Status: OK (verification inherited from the
entry's own recorded check, not independently re-opened — declared).**

### Crossref-resolved sample (24 entries)

All 24 resolved. Nineteen matched title, year, authors, venue and pages with no discrepancy. Five
flagged by my automated comparison and **all five resolve to normalization artifacts, not errors**:

| Entry | Flag | Resolution |
|---|---|---|
| `paiva2026stmtlnet` | title mismatch | LaTeX accent escaping (`Representa{\c{c}}{\~o}es`) vs Crossref's Unicode. Same title. Authors, venue ("Anais do X Workshop de Computação Urbana (CoUrb 2026)"), pages 323–336 all match. **OK** |
| `capanema2023poirgnn` | author mismatch | Crossref has "de Oliveira", bib has "Oliveira" as the surname split point. Venue "Ad Hoc Netw" vs "Ad Hoc Networks" is the bibliography's own abbreviation convention. **OK** |
| `wilcoxon1945` | pages | Crossref's `page` field holds only the start page (80); the bib gives 80–83. Volume 1, number 6, 1945, DOI 10.2307/3001968 all match. **OK — the bib is more complete than Crossref** |
| `vielhaus2022handover` | author + venue | "K{\"u}lzer" vs "Kulzer" (diacritic); "Proc. ACM MobiWac" is the abbreviated form of the full proceedings title. Pages 19–27 match. **OK** |
| `yang2015tsmc` | author | Crossref returns given+family unsplit for two authors; the bib's four surnames are correct. Venue and pages 129–142 match. **OK** |

### arXiv-resolved sample (8 entries)

Seven exact title matches: `hazimeh2021dselectk` (2106.03760), `russwurm...` (2310.06743),
`sener2018mgda` (1810.04650), `sun2025kgtb` (2509.12350), `velivckovic2017graph` (1710.10903),
`wu2024torchspatial` (2406.15658), `wongso2025massivesteps` (2505.11239). One mismatch:
`ruder2017sluice` (R-02).

Note: `sun2025kgtb` carries `doi = {10.48550/arXiv.2509.12350}`, which returns 404 from Crossref
(DataCite-registered, not Crossref). The arXiv record resolves and matches. **Not a defect** — the
DOI is real, it is simply in the wrong registry for a Crossref query.

### Entries with no resolvable identifier (12 of 99)

Per R1 these would normally be BLOCKER-flagged. All twelve are pre-DOI or DOI-less venues where the
absence is genuine rather than a gap in verification:

`chen2018gradnorm` (ICML 2018), `holm1979` (Scand. J. Statist. 1979), `jure2014snap` (SNAP dataset,
carries a URL + access date), `kohavi1995crossval` (IJCAI 1995, predates DOIs),
`kurin2022scalarization`, `liu2023famo`, `xin2022domtl`, `yu2020pcgrad` (NeurIPS proceedings),
`nash` (ICML 2022), `pedregosa2011sklearn` (JMLR), `senushkin2023aligned` (CVPR 2023),
`velickovic2019deep` (ICLR 2019).

ICML/NeurIPS/ICLR/JMLR proceedings papers genuinely have no publisher DOI. **I did not
independently resolve these twelve against their proceedings pages this session.** Per fail-closed,
they are **UNVERIFIED — blocked on: opening each venue's proceedings index**, not "assumed fine".
None is a suspicious entry: all are canonical, high-visibility works whose bibliographic details are
internally consistent and match the citing text's use of them.

---

## Claim-support audit (the adversarial half)

Sampled sites, weighted toward load-bearing methodological claims and toward Chapter 2, the most
heavily AI-drafted chapter.

| Site | Sentence (abridged) | Verdict |
|---|---|---|
| `2_fundamentals.tex:364` | "Random loss weighting is a competitive baseline \cite{lin2022rlw}" | **SUPPORTED** — RLW's own framing |
| `2_fundamentals.tex:365` | "a controlled study finds that current MTL optimizers often do not outperform a well-tuned fixed-weight baseline \cite{xin2022domtl}" | **SUPPORTED** — the cited paper's title is literally "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"; "often do not" correctly hedges a paper that reports a negative-to-mixed result |
| `2_fundamentals.tex:366-368` | "a direct defense of plain loss summation shows that a unitary scalarization with standard regularization matches or improves upon the specialized optimizers \cite{kurin2022scalarization}" | **SUPPORTED** — "In Defense of the Unitary Scalarization for Deep Multi-Task Learning"; "matches or improves upon" is the paper's own claim strength, neither inflated nor hedged away |
| `2_fundamentals.tex:475-478` | "The entropy analysis of human movement that reports a potential predictability of about 93 percent sets a bound for predicting the next location at coarse resolution ... it is not, however, a ceiling on seven-class category macro-F1 or on region ranking" | **SUPPORTED, with a note** — see below |
| `2_fundamentals.tex:485` | "Estimates use stratified k-fold cross-validation \cite{kohavi1995crossval}" | **SUPPORTED** (Kohavi §6) |
| `2_fundamentals.tex:488` | "a grouped, stratified splitter keeps all of a user's check-ins on one side of every fold \cite{pedregosa2011sklearn}" | **SUPPORTED** — cited for the library implementing StratifiedGroupKFold, which is the correct use |
| `2_fundamentals.tex:497-502` | Wilcoxon \cite{wilcoxon1945} "compares two models across the paired results without assuming normality" and its p-floor | **SUPPORTED** |
| `2_fundamentals.tex:503-504` | "the Holm step-down procedure controls the family-wise error \cite{holm1979}" | **SUPPORTED** |
| `2_fundamentals.tex:507` | "two one-sided tests to establish statistical non-inferiority within a two-point margin \cite{lakens2017tost}" | **SUPPORTED** |
| `2_fundamentals.tex:166-167` | "Huang et al.\ present HGI as a method for urban region representation and evaluate it on region-level estimation tasks, so the work reported here repurposes its POI-level output" | **SUPPORTED and exemplary** — this is the round's HGI-repurposing disclosure. It describes the cited system as its authors describe it (R2's accuracy check) *and* declares the dissertation's departure from that framing. Model citation practice |
| `5_mobiwac.tex:752-754` | "Moura et al.~\cite{moura2025mobilityaware}, whose structural analysis of tourist check-ins reads past visits only, name machine learning as a next step" | **SUPPORTED, correctly hedged** — "name ... as a next step" is precisely the hedged form the persona's attack (d) demands; it does not become "calls for" |
| `5_mobiwac.tex:706` | "CSLSL reports its chain outperforming a shared-trunk parallel variant on its own benchmarks~\cite{huang2024cslsl}" | **SUPPORTED** — scoped to "on its own benchmarks", which is the honest form |

### One note on the Song 2010 site

The citing sentence uses the 93% figure. The bib entry's own provenance comment says:

> "Cited for the feasibility premise only (mobility is predictable in principle); the 93% figure is
> for cell-tower trajectories and is deliberately NOT transplanted into our prose (check-in streams
> are sparser)."

The prose now *does* state the figure — but it states it correctly and immediately fences it:
"sets a bound for predicting the next location at coarse resolution ... it is not, however, a
ceiling on seven-class category macro-F1 or on region ranking, which are different label spaces."
That fencing is exactly the caveat the bib comment was protecting. I verified the entry against
Crossref (10.1126/science.1177170: Science 327(5968), 1018–1021, 2010, four authors Song/Qu/Blumm/
Barabási — all match).

**Verdict: SUPPORTED.** The prose is more careful than the bib comment anticipated. *But* the bib
comment now misdescribes the prose, and a future editor reading "deliberately NOT transplanted"
might delete a sentence that is fine. **Recommend updating the comment** — MINOR, R-04.

---

## R4 · Inherited errata check

| Erratum (NORTH_STAR §4) | Status in `src/references.bib` |
|---|---|
| Wrong POI-RGNN paper (CBIC-era) | `capanema2023poirgnn` resolves to Ad Hoc Networks 2023, DOI verified via Crossref, five authors matching. **Fixed** |
| HMRM author names | `chen2020modeling` present with DOI; resolved clean in the Crossref sample. **Fixed** |
| GAT venue | `velivckovic2017graph` carries arXiv:1710.10903, verified; the entry comments reference the GAT erratum as the precedent for the Russwurm re-typing. **Fixed** |
| CoUrb `silva2025mtlnet` (wrong venue, stale "Submetido") | entry present; Chapter 4 cites it at `:84` with the correct chapter cross-reference and a note on the MTLNet/MTLnet typesetting difference. **Fixed** |

No inherited erratum survives.

## R5 sweep

No AI output cited as a source anywhere. No real-looking citation laundering an unverifiable claim.
The AI-use disclosure (Appendix C) describes tool use without citing any model as evidence. **Clean.**

## Cross-checks

| Check | Result |
|---|---|
| Every `\cite` key resolves to a bib entry | 98/98 |
| Every cited key appears in the rendered list | **97/98** — R-01 |
| Unresolved `[?]`/raw keys in the build | **4 renders of `(??)`** — R-01 |
| Orphan bib entries (uncited) | 1: `liu2014geographical`. **NIT** — harmless, but it is the 99th entry and its removal would make the 98/98 arithmetic self-evident |
| Self-citation posture | `silva2025mtlnet` and `paiva2026stmtlnet` are cited as the chapters' own published records, with contribution notes in the prefaces. Not the subject of an intro sentence. **Compliant** |

---

## `[VERIFY]` list for the author

1. **R-01** — the dropped Russwurm entry. Build-blocking; the fix is in the comment, not the entry.
2. **R-02** — `ruder2017sluice` should be re-typed to the AAAI 2019 record.
3. **R-03** — two stale ledger lines in `2_fundamentals.tex` (`:524`, `:561`) contradict the Kohavi
   erratum.
4. **R-04** — the `song2010limits` bib comment says the 93% figure is not transplanted; the prose now
   uses it (correctly fenced). Update the comment so a future editor does not "fix" good prose.
5. **UNVERIFIED (fail-closed)** — the twelve DOI-less proceedings entries listed above were not
   independently resolved against their venue indexes this session. None is suspicious; all are
   canonical works. If the author wants 100% coverage before the banca build, these twelve are the
   remaining set.

## Out-of-scope handoffs

- Persona 18: the `(??)` renders are also a visible page defect (pp. 21, 45, 49, 50).
- Persona 04: R-01 is also an L4 cross-reference failure.
- The task brief's "0 undefined refs/cites" build state does not match the current logs; the
  orchestrator should know before treating the build as clean.
