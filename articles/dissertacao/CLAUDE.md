# CLAUDE.md — Dissertação de Mestrado (UFV / PPGCC) — working folder

> **What this folder is.** The working folder for Vitor H. O. Silva's **master's dissertation** at
> UFV / PPGCC (Ciência da Computação, Campus Florestal / NESPeD-LAB), advisor Fabrício A. Silva.
> Format: **coletânea de artigos** (UFV Normas §2.3(iii)/§2.6), **English frame**, defense target
> **August 2026**. This file is the landing: read it first, then the doc in the read-order below
> that matches your task.

## 0 · Read-first order (by task)

| Your task | Read, in order |
|---|---|
| Any writing at all | This file → [`WRITING_LAW.md`](WRITING_LAW.md) → [`GLOSSARY.md`](GLOSSARY.md) → [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) → [`NORTH_STAR.md`](NORTH_STAR.md) §3–§4 + §6 (chapter map, preface/time-capsule rules, per-chapter errata, story spine) |
| Planning / structure | [`NORTH_STAR.md`](NORTH_STAR.md) → [`PLAN.md`](PLAN.md) |
| LaTeX / formatting | [`TEMPLATE.md`](TEMPLATE.md) → [`UFV_COMPLIANCE.md`](UFV_COMPLIANCE.md) |
| Submission mechanics / deadlines | [`UFV_COMPLIANCE.md`](UFV_COMPLIANCE.md) → [`PLAN.md`](PLAN.md) |
| Imitating the praised example | [`exemples/VIEGAS_ANALYSIS.md`](exemples/viegas/VIEGAS_ANALYSIS.md) |

**Non-negotiable for every agent:** the guardrails in [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md)
(citation protocol, number protocol, claim registry, review gates) apply to every edit in this
folder. No chapter text goes to the author or advisor without passing its gates.

## 1 · Current state (2026-07-24 — v1 ASSEMBLED + corrections round 2)

- **v1 lives in [`src/`](src/)** — the single working copy; the draft folders (`storyline/`,
  `fundamentals/`) are **frozen** (freeze pointers in each). **One source, two builds** from a
  single `main.tex`: `make defense` → `build/main.pdf`, copied to
  [`src/dissertacao.pdf`](src/dissertacao.pdf) (87 pp, full front matter, the banca PDF);
  `make final` → `build/main_final.pdf` (83 pp, AcademicoPG body-only). 0 errors, 0 undefined
  refs/cites, `make check` lint 0. Build recipe: [`src/src_utils/README_SRC.md`](src/src_utils/README_SRC.md).
- **`src/` layout (restructured round 2):** LaTeX source + `chapters/` + `figures/` + `tables/`
  + the one `dissertacao.pdf` at the root; **`src_utils/`** holds all non-LaTeX (README, lint,
  reports, the review outputs `_review_v1/`, gate reports `_gates/`, the CBIC recompute result,
  and the pt_BR decisions doc); **`build/`** holds all compile output (gitignored).
- **What was built:** skeleton from the Germano tree (Times, numeric cites); Chapters 3/4/5
  re-typeset (errata applied + ledgered → Appendix B); frame Chapters 1/2/6 imported; one global
  [`src/references.bib`](src/references.bib) (99 entries, 0 dangling); front/back matter +
  Appendices A/B/C. Full-document gates + the **18-persona review suite** ran (reports in
  `src/src_utils/_review_v1/`; consolidated at `.../CONSOLIDATED_REVIEW_REPORT.md`).
- **Corrections round 2 (this session, all committed):** title set to the working option
  *From Representations to a Single Joint Model: …* (alternates commented in `0_main.tex`;
  final call still with the advisor); Ch.3/4/5 headings shortened (fixes the header-padding +
  TOC-wrap defect, Germano precedent); **B.1 CBIC misattribution corrected in BOTH the
  dissertation Ch.5 AND the version-of-record [`articles/[mobiwac]/src/`]** (author-authorized
  cross-boundary edit; logged in the MobiWac ERRATA + Appendix B + Ch.5 ledger); `src/`
  restructured; three Locus exemplar dissertations added under `exemples/` + calibration note
  deepened; the three configured specialist profiles (BANCA\_SIMULATOR, DISSERTATION\_FACT\_GATE,
  DISSERTATION\_REVIEWER) re-run on the corrected v1.
- **Author actions before the advisor build** (full ranked list:
  [`src/src_utils/HANDOFF_v1.md`](src/src_utils/HANDOFF_v1.md); PT decisions doc:
  `src/src_utils/DECISOES_PENDENTES_ptBR.md`): (1) **title** — confirm/replace with the advisor;
  (2) **CBIC dataset counts** — recomputed via the Gowalla ETL this round
  (`src/src_utils/cbic_recompute_result.md`), confirm + wire into Ch.3 (still `[VERIFY]`);
  (3) approve the queued `[NEEDS SIGN-OFF]` items (Resumo/Abstract, AI-disclosure, claim-scope
  rewordings). **Tell the advisor** the v1 was machine-assembled (reading map + Appendix C
  disclosure accompany the PDF).
- **Review-suite model deviation:** the plan mandated the Fable model for every reviewer persona;
  Fable tokens were exhausted mid-run, so on the author's call the suite ran on `claude-opus-4-8`.
  Logged here, in the handoff, and in the AI-use disclosure (Appendix C).

### History — 2026-07-18, evening (decisions round closed)

- **Bases established** (this doc set) + **story settled** (NORTH_STAR §6 spine). Next: skeleton
  + mass drafting launch 2026-07-19 ([`PLAN.md`](PLAN.md)). **Chapter drafting is blocked until
  the skeleton + [`GLOSSARY.md`](GLOSSARY.md) exist** (PLAN Day 0–1).
- **The three articles** (arc mapped in [`NORTH_STAR.md`](NORTH_STAR.md)):
  - **CBIC 2025** — *An Investigation into Multi-Task Learning for Point-of-Interest Category
    Classification and Next-POI Prediction* (EN, Vitor 1st author). **PUBLISHED — DOI
    `10.21528/CBIC2025-1191324` (verified resolves 2026-07-18 → sbia.org.br, CBIC 2025, 8 pp.).
    Satisfies Art. 21 — file the comprovante with the secretariat.**
  - **CoUrb 2026 (SBRC)** — *ST-MTLNet: Representações Espaço-Temporais de Pontos de Interesse
    para Aprendizado Multitarefa* (PT, **Tarik S. Paiva 1st author, Vitor 2nd**, presenter
    2026-05-25). **PUBLISHED — DOI `10.5753/courb.2026.22960` (verified resolves 2026-07-18 →
    SBC SOL, Anais do CoUrb 2026, pp. 323–336).** In the coletânea as a full chapter,
    translated to EN (ledger below).
  - **MobiWac 2026** — *Predicting the Next Category and Region of a Visit: A Check-in-Level
    Multi-Task Study on Mobility Data* (EN, Vitor 1st author). **Uploaded to EDAS 2026-07-09,
    under review** (EDAS #1571313639; an 8-page build existed by the 07-11 deadline — confirm
    which PDF is the version of record). Present as "submitted" — allowed by UFV §2.3(iii).
    *This bullet list is the owner of venue/status/date facts; other docs point here.*
- **BRACIS 2026 is NOT a chapter**: rejected (notification 2026-06-08), unpublished, its material
  absorbed into MobiWac and its central region-cost claim corrected by MobiWac. Cite at most as an
  intermediate iteration in the frame narrative.
- **Prior folder** [`../[TESE]_MTL_POI/`](../%5BTESE%5D_MTL_POI/) (2026-06-05) is **superseded by
  this folder**: its `NORMAS_UFV.md` extraction was re-verified 2026-07-18 and folded into
  [`UFV_COMPLIANCE.md`](UFV_COMPLIANCE.md) (one material change: new regimento of 2026-07-09); its
  README's decisions on frame language (PT) and article order (CBIC→CoUrb→BRACIS) are superseded
  by the ledger below. Do not extend that folder.

## 2 · Decisions ledger

**Settled (author, 2026-07-18 — second round closed the structural decisions) — do not
silently reopen:**

| Decision | Ruling |
|---|---|
| Format | Coletânea de artigos (UFV §2.6): Introdução Geral → article chapters → Conclusão Geral. |
| Frame language | **English** — author-settled; advisor sign-off pending (Normas §1.3 puts language at the Comissão's discretion). Portuguese only where mandated (Resumo; system fields). |
| Articles + **order** | **CBIC → CoUrb → MobiWac** (author confirmed 2026-07-18; chronological = intellectual order). BRACIS excluded (rejected/superseded). |
| CoUrb inclusion | **Full chapter.** Norms check: nothing in Normas §2.3/§2.6 or the regimento requires first authorship — articles need only be pertinent to the research and published/accepted/submitted; CoUrb qualifies (published, DOI). Advisor/Comissão sign-off still recommended (unregulated area; the Viegas precedent used 1st-author works only). Chapter carries a contribution note (Vitor: 2nd author, presenter, author of the baseline MTLnet). |
| CoUrb language | **Translate to EN** (author will launch a translation agent; fidelity gate AGENT_GUARDRAILS L5 mandatory). The chapter states it is a translated reproduction of the CoUrb 2026 paper, with the original DOI — permitted under §2.6 free formatting; §2.6.3 makes mixed language legal anyway, so translation is a style choice, not a compliance need. |
| MobiWac chapter version | **Current working build in `articles/[mobiwac]/src/`** (author: "the last one in the src"; being refined in parallel — re-sync before the final gate pass). |
| Bibliography | **Single global, Viegas-style** (numeric, one consolidated list). |
| AI-use disclosure | **Proceed** (author OK). Recommended placement: short appendix ("AI-use disclosure") — it survives both build modes; drafted from git provenance (AGENT_GUARDRAILS §6). |
| Art. 21 | **Satisfied by CBIC** — DOI `10.21528/CBIC2025-1191324` verified 2026-07-18. Remaining action (PLAN Day 0): file the comprovante with the secretariat + confirm banca logistics. |
| Defense target | **August 2026**, compressed: **complete v1 to advisor 2026-07-24**; advisor round Jul 27–31; to banca ≈ Aug 1 → defense from ≈ Aug 21 ([`PLAN.md`](PLAN.md)). |
| Typesetting model | Viegas-style **hybrid**: papers re-typeset as unified chapters (not stapled PDFs), consistent numbering. Base template per [`TEMPLATE.md`](TEMPLATE.md). |
| Writing law + terms | [`WRITING_LAW.md`](WRITING_LAW.md) (inherits the MobiWac GLOSSARY honesty rules and AI-tell bans) + [`GLOSSARY.md`](GLOSSARY.md) (the dissertation's own term registry — canonical names, model lineage, acronyms, PT equivalents). |

**Still open (small; full context in [`NORTH_STAR.md`](NORTH_STAR.md) §5):**

1. **Dissertation title** — author: decide after the arc/text firms up (candidates in
   NORTH_STAR §5.8; must match the folha de rosto exactly). Latest sensible moment: before the
   defense-build front matter (≈ Jul 23).
2. **Errata policy** (NORTH_STAR §4): default = fix in the re-typeset chapters + Appendix B
   listing, unless the advisor objects.
3. **Advisor sign-offs bundle** (one conversation): frame language EN, CoUrb inclusion,
   defense date + banca names, title shortlist.

## 3 · Doc map (this folder)

| Doc | Contents |
|---|---|
| [`NORTH_STAR.md`](NORTH_STAR.md) | Thesis question, the three-paper arc, chapter map, per-chapter adaptation notes + errata, title candidates, open decisions. |
| [`WRITING_LAW.md`](WRITING_LAW.md) | The writing law: register, canonical names, honesty rules, AI-tell bans (2026 state), Viegas-derived style patterns. |
| [`GLOSSARY.md`](GLOSSARY.md) | The term registry: tasks + per-paper mapping, model lineage, protocol terms, metrics, acronyms, PT equivalents. Fail-closed: a term not in the registry may not be used. |
| [`AGENT_GUARDRAILS.md`](AGENT_GUARDRAILS.md) | Process law for AI-assisted writing: citation/number/claim protocols, review gates, long-form countermeasures, AI-use disclosure. |
| [`UFV_COMPLIANCE.md`](UFV_COMPLIANCE.md) | UFV/PPGCC rules distilled + verified links: format spec, coletânea rules, defense prerequisites, AcademicoPG pipeline, deadlines. |
| [`TEMPLATE.md`](TEMPLATE.md) | LaTeX template decision + adaptation checklist + the two build modes (defense PDF vs. AcademicoPG upload). |
| [`PLAN.md`](PLAN.md) | Backwards-planned milestones to the August 2026 defense; risks and fallback. |
| [`docs/`](docs/) | Official PDFs (submission manual 04_2026) + [`docs/research/`](docs/research/) (raw 2026-07-18 research records with all source URLs: AI-writing evidence, template survey, norms verification). |
| [`exemples/`](exemples/) | [`viegas/`](exemples/viegas/): Viegas 2026 PDF + [`VIEGAS_ANALYSIS.md`](exemples/viegas/VIEGAS_ANALYSIS.md) (the quality bar). [`germano/`](exemples/germano/): Germano 2024 (same advisor, defended, EN) — full WORKING LaTeX source, the skeleton candidate ([`TEMPLATE.md`](TEMPLATE.md) §0). |
| [`reviewers/`](reviewers/) | The reviewer persona suite (2026-07-20): **18 invocable review agents** — banca simulator, MTL + POI domain experts, stats/leakage skeptic, the G2 fact-gate trio (citations, numbers, claims), the G3 style gate, cold reader, concordance checker, line editor, L5 translation gate, UFV compliance, the pre-application adversarial advisor, plus (added later same day) the readability editor (15), AI-credibility reviewer (16), excellence assessor (17), and visual-presentation reviewer (18). Start at [`reviewers/README.md`](reviewers/README.md) (roster, pipeline, common protocol); research provenance in `docs/research/` (banca evaluation, MTL/POI criteria, AI-detection landscape, dissertation-excellence criteria). |

## 4 · Where the science lives (do not duplicate it here)

- **MobiWac numbers**: single source of truth is
  [`docs/studies/closing_data/RESULTS_BOARD.md`](../../docs/studies/closing_data/RESULTS_BOARD.md)
  (§1 headline, §3 file map) + the claim whitelist in
  [`articles/[mobiwac]/PAPER_PLAN.md §3`](../%5Bmobiwac%5D/PAPER_PLAN.md). Never quote MobiWac
  numbers from memory or from prose.
- **CBIC / CoUrb numbers**: the published tables in `articles/CBIC___MTL/` and
  `articles/CoUrb_2026/` are the source — reproduce, never recompute (errata handled per
  NORTH_STAR §4).
- **Repo-wide guide**: [`/CLAUDE.md`](../../CLAUDE.md) (architecture, canonical versions, traps).
- **MobiWac paper folder**: [`articles/[mobiwac]/CLAUDE.md`](../%5Bmobiwac%5D/CLAUDE.md).
