# AGENT_GUARDRAILS.md — process law for AI-assisted writing (v1, 2026-07-18)

> **Scope.** How humans and agents WORK on this dissertation: what an agent may and may not do,
> the verification gates every chapter passes, and the disclosure obligations. The word-level
> law is [`WRITING_LAW.md`](WRITING_LAW.md); this file is about process. It is evidence-based:
> §8 lists the 2025–2026 measurements each rule answers to. The rules are deliberately
> fail-closed — when in doubt, an agent STOPS and flags, it does not improvise.

---

## 0 · The three prime directives

1. **Agents draft; the author owns.** No chapter, claim, or number is "done" until the author
   (Vitor) has read and approved it. AI cannot be an author (CNPq/ICMJE/COPE consensus); the
   author is accountable for every word, so every word must be verifiable by him.
2. **Nothing enters the document from model memory.** Every citation, number, name, date, and
   quoted claim traces to a file in this repo or a source opened during the session. If the
   provenance is "the model recalled it", it does not go in.
3. **Fail closed.** A claim that cannot be verified is removed or flagged `[VERIFY: …]` — never
   smoothed over. An agent that is uncertain says so in its handoff note; self-reported success
   is not trusted (the author audits independently — standing project feedback).

## 1 · Citation integrity protocol

**Why (measured):** GPT-class models fabricate ~20% of citations in literature-review settings
(6–29% by topic sparsity; ~7–8% even with web search); fabricated citations in published papers
rose about tenfold from 2023 to early 2026 (1/2,828 papers → 1/277), still accelerating; 100
hallucinated citations were found in accepted NeurIPS 2025
papers that 3–5 reviewers each missed; arXiv now bans authors over it. Existence-checking alone
misses the two dominant subtler classes: partial attribute corruption (~27%) and
claim-not-supported ("semantic") errors.

**The rules:**

- **R1. No bib entry from memory, ever.** A reference enters `references.bib` only with (a) a
  resolvable identifier (DOI / arXiv ID / ACM-DL / SBC-SOL / publisher landing page) checked
  against the source of record (Crossref/OpenAlex/publisher), AND (b) the PDF or landing page
  actually opened, AND (c) the cited claim located in the source (page/section noted in a bib
  comment). The MobiWac bibliography (every entry web-verified, with quotes in bib comments) is
  the template — and the preferred donor: reuse its entries verbatim where the dissertation
  cites the same works.
- **R2. Attribute fidelity.** Author list, venue, year, pages copied from the source of record,
  not retyped from another paper's bibliography. Venue names per the bibliography style chosen
  in [`TEMPLATE.md`](TEMPLATE.md); one accuracy check per citation ("describe cited systems as
  their authors describe them").
- **R3. Claim-support audit.** Before any advisor handoff, an adversarial pass samples ≥20% of
  citations (100% for new-this-pass entries) and verifies the SENTENCE citing them is actually
  supported — not merely that the reference exists.
- **R4. Inherited errata stay fixed.** The known CBIC/CoUrb citation errata (owner list:
  NORTH_STAR §4) are corrected in the dissertation bibliography regardless of the errata policy
  chosen for chapter text (NORTH_STAR §5.7).
- **R5. AI output is not a source.** Never cite a model or its answer as evidence; never launder
  a model claim through a real-looking citation.

## 2 · Number integrity protocol

**Why (measured):** models corrupt numbers even when given correct input (numeric spans are a top
faithfulness-error class); in whole-paper generation studies, the most polished agent drafts
carried the MOST fabrications (>10/paper), and 57% of one system's papers contained wrong or
hallucinated numerical results. Polish anti-correlates with groundedness — which is exactly the
failure mode a good-looking dissertation chapter invites.

**The rules:**

- **N1. Single source of truth per chapter.**
  - Ch.5 (MobiWac): `docs/studies/closing_data/RESULTS_BOARD.md` §1 (via its §3 file map to the
    JSONs) + the claim whitelist in `articles/[mobiwac]/PAPER_PLAN.md §3`. Never from prose, never
    from memory.
  - Ch.3 (CBIC) and Ch.4 (CoUrb): the published papers' own tables (with the documented errata,
    NORTH_STAR §4, as the only sanctioned corrections — CoUrb's audited win-count/means come from
    `articles/CoUrb_2026/slides/judge_feedback.md`).
  - Frame chapters: may only repeat numbers already sourced in a chapter, with the same hedges.
- **N2. Agents quote; they do not compute.** No mental arithmetic, rounding, aggregation,
  percentage conversion, or delta-taking in prose. Derived quantities come from a script
  committed to the repo (or the existing analysis scripts), then are quoted.
- **N3. Every numeral is traceable.** Any number an agent writes must be traceable to its source
  file in the handoff note (a "numbers ledger": value → file → field). Numbers without a ledger
  line fail the gate.
- **N4. Numeral-extraction audit (the gate).** Before advisor handoff, extract every numeral+unit
  from the chapter (script or manual sweep) and match each against its ledger source, exact or
  declared-rounding. Orphan numbers block the handoff. Also cross-check: abstract vs body,
  captions vs table content, prose interpretation vs the statistic named (the sciwrite-lint
  check classes).
- **N5. Convention named.** Every reported cell states its convention (metric, selection rule,
  n, seeds×folds) per WRITING_LAW §3; the MobiWac joint-best vs diagnostic-best distinction must
  never blur.

## 3 · Claim registry

- **C1. The whitelist governs.** Scientific claims about MobiWac results come only from
  `PAPER_PLAN.md §3` (CAN-say / must-NOT-say) + the decisions ledger in
  `articles/[mobiwac]/CLAUDE.md §3`. Claims about CBIC/CoUrb come from their published texts,
  time-indexed per WRITING_LAW §3.
- **C2. New claims need sign-off.** Any claim not derivable from the registry (including
  "obvious" connective claims in the frame, e.g. about what the arc "shows") is proposed in the
  handoff note and enters the text only after the author approves. The Introdução Geral's
  arc-narrative sentences are claims — they get the same treatment.
- **C3. Never-cite lists are absolute** (inherited from the MobiWac board): STAN v4-collapse
  numbers, ReHDM v2 row, VOID fp16/bf16 collapsed cells, pre-bugfix findings flagged in
  `docs/PAPER_FINDINGS.md`.
- **C4. BRACIS containment.** **No BRACIS result, number, or claim appears anywhere in the
  dissertation**, and its C2-era region-cost claim is never reissued. The rejected manuscript is not
  disclosed to the reader at all.
  > **Amended 2026-07-27 by author decision** (NORTH_STAR §5 item 11; the matching edit was owed to
  > this file and is applied here). C4 previously required the *opposite* of its current form: it
  > mandated a containment device, citing BRACIS "only as an earlier unpublished iteration". Appendix
  > A §A.2, which carried that disclosure, was removed on the author's grounds that the trail of a
  > rejected-then-reworked manuscript adds reading complexity without serving the reader, and that
  > reworking after a reject is common practice with the conclusion unchanged. The *prohibition* half
  > of C4 survives and is now the whole rule; the *disclosure* half is void. Nothing in the document
  > asserts a correction relative to that manuscript (swept 2026-07-27: every "earlier" or
  > "corrected" passage names CBIC, CoUrb, the submitted MobiWac manuscript, or a development-time
  > data preparation as its own antecedent).

## 4 · Long-form failure-mode countermeasures

**Why (measured):** long outputs degrade (repetition in ~45% of long generations; quality
collapse past ~2k words in most models); models retrieve mid-context constraints unreliably
(context rot), so "keeping the whole thesis in context" silently drops notation defined two
chapters earlier; models reuse discourse skeletons across sections; register drifts across
model versions over a months-long project.

**The rules:**

- **L1. Bounded drafting.** No agent drafts a whole chapter in one pass. Work section-by-section
  against the approved outline (NORTH_STAR §3), ≤ ~1,500 words per drafting unit, each unit
  reviewed before the next.
- **L2. Consistency lives in files, not context.** Notation, canonical names, and claim scopes
  are enforced by checking against WRITING_LAW §2, the term registry
  [`GLOSSARY.md`](GLOSSARY.md) (a term not in the registry may not be used; agents propose,
  the author approves, the entry lands BEFORE the term does), and the claim registry — never by
  trusting that the agent "remembers" earlier chapters. Every session re-reads the law files
  first (CLAUDE.md §0 order).
- **L3. Cross-chapter duplication check.** Before advisor handoff: n-gram/near-duplicate sweep
  across chapters (paper-chapters legitimately share background; the frame must not repeat
  itself or them beyond the sanctioned recap subsections).
- **L4. Cross-reference lint.** Every `\ref`/`\cite`/section pointer resolves; no "as discussed
  in Section X" pointing at the wrong target (a known Viegas defect class).
- **L5. Translation fidelity gate** (if CoUrb is translated): a separate verification pass
  compares source PT and target EN sentence-by-sentence for claim-strength drift — quantifiers,
  hedges, and numbers must map 1:1.
- **L6. Fresh-eyes audits.** Style and consistency audits are run by an agent that did NOT write
  the text under audit (or by the author), never self-certified by the drafting agent.

## 5 · Review gates (the pipeline every chapter passes, in order)

```
G0 OUTLINE   author approves the section outline (scope + claims to be made)
G1 DRAFT     agent drafts per L1; handoff note lists: numbers ledger, new-claim proposals,
             [VERIFY] flags, sources opened
G2 FACT GATE (fail-closed) citation protocol §1 + number protocol §2 + claim registry §3
             + cross-ref lint L4. Any failure returns to draft.
G3 STYLE GATE (statistical, separate pass, fresh eyes) WRITING_LAW §7 checklist: AI-tell sweep,
             idiom sweep, variance/burstiness read-aloud, discourse-skeleton variety, register
G4 AUTHOR    Vitor reads and approves (edits welcome; approval recorded in git)
G5 ADVISOR   only after G2–G4 are green
```

- Gates G2 and G3 are **separate passes** (fact ≠ style; merging them measurably weakens both).
- Audit intensity scales with AI share: a chapter that is mostly re-typeset published text gets
  the standard pass; heavily AI-drafted frame prose gets the full adversarial treatment
  (contamination is bimodal — heavy-reliance documents carry most fabrications).
- Git discipline supports provenance: AI-drafted and author-drafted content land in
  distinguishable commits (`draft(ai): …` vs `edit(author): …`), so the disclosure statement
  (§6) is reconstructible from history rather than remembered.

## 6 · AI-use disclosure (required, not optional)

**The landscape (verified 2026-07-18):** no binding UFV/PPGCC rule yet, but (a) **CNPq Portaria
nº 2.664/2026** mandates declaring any generative-AI use (tool + purpose) for CNPq-linked
researchers and forbids submitting AI-generated content as human-authored; (b) UFV/DPE published
a recommended declaration format (03/2026); (c) CAPES directives are converging on the CNPq
policy; (d) every major publisher (ICMJE, Elsevier, Springer, IEEE, ACM) requires disclosure;
(e) PPGCC separately requires an **anti-plagiarism certificate** for the defense.

**The rules:**

- **D1.** The dissertation carries an AI-use disclosure note (placement: open decision,
  NORTH_STAR §5.9) naming: the tools and model versions, the scope of use per part (drafting,
  editing, formatting, code), and the human-verification steps applied (this file's gates).
  Honest, specific, one page maximum.
- **D2.** The disclosure is drafted from the git provenance trail (§5), not from recollection.
- **D3.** Raise with the advisor EARLY (it is also his risk); if he wants committee
  pre-authorization, obtain it before mass drafting, not after.
- **D4.** The anti-plagiarism certificate is a defense blocker — schedule it in PLAN.md, and
  remember AI-assisted text still must not lift verbatim prose from sources (paraphrase +
  citation discipline as usual; the coletânea's own papers are exempt self-material, stated in
  the organization section).

## 7 · Known agent biases this file counters (name them to catch them)

| Bias | Counter |
|---|---|
| **Sycophancy** (agreeing with the author's slip instead of checking — e.g. the CoUrb→CBIC order) | Evidence beats instruction on facts; discrepancies are flagged with sources, decisions stay the author's (this file exists because the check caught exactly that). |
| **Plausible confabulation** (citations, numbers, "recalled" details) | §1–§2 fail-closed protocols. |
| **Polish over grounding** (the best-looking draft carries the most errors) | G2 before G3; polish never substitutes for a ledger line. |
| **Overclaiming / verdict inflation** (upgrading "matches" to "outperforms", widening scopes) | Claim registry §3 + WRITING_LAW §3 verb-test binding. |
| **Padding** (length as a proxy for quality) | Outline-bound drafting (L1); every section must earn its pages; the Viegas example is ~100 pages total — that is the calibration, not a target to exceed. |
| **Fake cohesion** (template transitions, uniform section shapes) | G3 skeleton-variety check; WRITING_LAW §4.4. |
| **Variance compression in edit passes** (homogenizing the author's voice) | Edits preserve burstiness; a pass that only smooths is rejected (WRITING_LAW §4.3). |
| **Self-certification** (agent declares its own output verified) | L6 fresh-eyes rule; author audits independently. |
| **Trusting the tolerant tool** (two checks disagree; the one reporting success is believed) | The source did not compile for six commits while `build.sh` reported "104 pp, 0 overfull, 0 undefined": under `-interaction=nonstopmode` pdflatex recovers from an error and still writes a PDF, and the checker never looked for TeX errors. `make` (`-halt-on-error`) produced nothing the whole time. **Rule: `tex_errors=0` is part of every build claim; a PDF existing is not evidence the source is correct; when two tools disagree about one artifact, distrust the one reporting success.** (2026-07-28, §2.3b of `science/AGENT_HANDOFF.md`.) |
| **A gate that has never fired** (a check whose passing carries no information) | Validate every new gate in BOTH directions before trusting it: run it against a tree where the defect is present and confirm it fails, then against the fixed tree and confirm it passes. Four of this repository's checkers were wrong at least once by being tuned only on the case in front of them. |
| **Silent correction** (fixing a published number/claim without a trail) | Errata policy (NORTH_STAR §5.7): every departure from a published source is listed and approved. |

## 8 · Evidence base (why these rules; verified 2026-07-18)

Citation fabrication rates and cases: Lancet/Columbia corpus study (fabricated citations 1/2,828
papers 2023 → 1/277 early 2026); JMIR Mental Health 2025 (GPT-4o 19.9% fabricated, 6–29% by
sparsity); GPTZero NeurIPS-2025 audit (100 fabricated refs in 53 accepted papers); Ansari 2026
taxonomy (66% total fabrication, 27% attribute corruption); Springer retraction of a fabricated-
citation ML book (2025); arXiv 1-year ban policy (2026). Number corruption: NAACL-Findings 2025
multi-doc faithfulness; AI-Scientist evaluation (57% papers with wrong numbers); PaperRecon/U-Tokyo
2026 (polish↔hallucination trade-off). Long-form: LongGenBench (repetition in ~45% of long
outputs); HelloBench (quality collapse past ~2k words); context-rot / Ref-Long (mid-context
constraint loss); QUDsim (discourse-skeleton reuse); syntactic-template detection (Shaib et al.);
"Voice Under Revision" (variance compression, Claude 78% of features). Tells: Kobak et al.
Science Advances 2025 (excess vocabulary, ≥13.5% of 2024 abstracts); Matsui 2025 (tell-avoidance
already measurable); Terčon & Dobrovoljc 2025 survey (POS-profile tells); refsmmat per-model word
rates (Claude "genuinely" ~10×). Policy: CNPq Portaria 2.664/2026; UFV/DPE guide 03/2026; CAPES
GT 2025 (+ NT 3/2025 via secondary sources — verify before citing verbatim); ICMJE 04/2025;
publisher policies (Elsevier/Springer/IEEE/ACM); U. Georgia / U. Toronto thesis policies; Unifesp
Res. 17/2025; Unicamp PRPG 2025. **Full findings with every URL:**
[`docs/research/ai_writing_evidence_2026-07-18.md`](docs/research/ai_writing_evidence_2026-07-18.md)
(kept verbatim; also the source pool for the dissertation's own disclosure appendix if needed).
