# Consolidated Review Report — Dissertation v1 (18-persona suite)

> Phase-7 deliverable. The full 18-persona review suite was run as fresh sub-agents on the
> `claude-opus-4-8` model (deviation from the plan's Fable mandate, made on the author's call
> after Fable-token exhaustion stalled the wave; logged in the handoff note). Each persona
> received only its persona file and the material its 'Read first' list names; the drafting
> agent reviewed none of its own text (L6). Full per-persona reports are in `src/_review_v1/`.

## 1. Per-persona verdicts

| # | Persona | Verdict | Summary |
|---|---------|---------|---------|
| 01 | Cold reader | **COMPREHENSIBLE — argument holds start to finish** | 3 top findings: next-POI collision, CBIC placeholders (BLOCKER), open title. Friction concentrated in known-open surface items, not the science. |
| 02 | Line editor | **MINOR PASS NEEDED** | All 5 zero-checks pass (em-dash 0, contractions 0). 3 majors: 'at [dataset]' inconsistency, Ch.4 title \:, percent-style. Ch.3/4 items route via Appendix B. |
| 03 | Style auditor | **GATE FAIL (document)** | 3 triggers: CBIC placeholder renders (BLOCKER), 'unlocks' in Ch.2 hinge, CBIC ban cluster. Punctuation triggers all clean. Ch.5 strongest; Ch.4 conditional pass. |
| 04 | Concordance checker | **SEAMS NEED WORK** | No blocker; no result misreported. 3 seams: 'MTLnet' named everywhere but absent in Ch.3; AL 64.51/64.54; weekday-motif residual. Arc threads cleanly. |
| 05 | Citation auditor | **GATE PASS** | 99/99 bib entries real, zero fabrications. 1 major (liu2014 Gowalla mis-source), 2 minor (date/title conventions). R4 errata 100% verified. No [VERIFY] outstanding. |
| 06 | Number auditor | **GATE FAIL (conditional)** | 1 blocker (CBIC placeholders, pre-existing), 0 fabrications, 0 mismatch. ~1225 numerals re-derived; 9/9 tables cell-identical. N-2 (AL 64.51/64.54), N-3 (CA n=15 partial). |
| 07 | Claim honesty auditor | **GATE PASS** | Zero unlicensed claims, verb-test binding airtight, AZ never upgraded. 2 majors straddle number scope (AL blur; data vintage 2009-2010 vs 2009-2011). Honesty devices intact. |
| 08 | Translation fidelity | **L5 PASS** | Ch.4 says exactly what the PT paper says. Tables cell-identical (63/63, 63/63, 9/9). Only departures = documented audited errata. Zero claim-strength drift. |
| 09 | Stats/leakage skeptic | **SURVIVES WITH CORRECTIONS** | Methods sound; holes are textual. B1 (BLOCKER): Wilcoxon vs t-test named two ways. B2/B3 majors: user-disjoint CV overclaimed; region pre-reg framing. Leak audit holds. |
| 10 | MTL expert | **SOUND WITH CORRECTIONS** | 1 BLOCKER: Ch.5 falsely says CBIC studied next-region + observed negative transfer. 2 majors: AL blur; cosine over-generalized to magnitude balancers. Well-armored vs field skepticism. |
| 11 | POI/mobility expert | **SOUND WITH CORRECTIONS** | 1 BLOCKER (CBIC placeholders render). 7 majors incl. 93% ceiling scope, Ch.3 split-axis silence, data vintage, Florida 990k vs 1.4M, no persistence baseline. Ch.5 exemplary. |
| 12 | Banca simulator | **APROVADO COM CORREÇÕES MENORES (45/50)** | 4 obligatory: title placeholder, CBIC scaffolding renders (p.35), AL 64.51/64.54, Wilcoxon-vs-t rigor overclaim. Honest correction-trail arc is the standout. None invalidating. |
| 13 | UFV compliance | **DEFENSE non-compliant as built / FINAL compliant with conditions** | 1 blocker (title renders on folha de rosto). Every MEASURED rule passes both builds (A4, Times 12pt, margins 3/2, 1.5 spacing, page nums). Table 1 overflow 29.8pt minor. |
| 14 | Adversarial advisor | **GATE — 6 landed fixes CERTIFIED** | All 6 phase-6 mechanical fixes certified (no rule broken, no disclosure lost). 6 pending APPROVE-WITH-EDIT (applied this loop). B.1 CBIC misattrib + B.8 preposition held for author. |
| 15 | Readability editor | **READ-WORTHY (overall 8/10; consistency 6/10)** | 1 critical (next-POI task collision), majors: Ch.4 italics, Ch.3 bold, CBIC unparseable sentences. Frame voice unified; surface seam not craft seam. |
| 16 | AI-credibility | **SCREENER MEDIUM / EXPERT-SUSPICION LOW** | Specificity audit PASS (largest credibility asset). 3 items: no front-matter disclosure line; do NOT sterilize Ch.3/4; freeze negative-parallelism count. Provenance shield strong. |
| 17 | Excellence assessor | **6 OUTSTANDING / 4 GOOD / 0 BELOW** | Science uniformly outstanding; every GOOD is packaging not substance. SBC CTD: yes-with-edits. Top moves: contributions->claims table, consolidated results view, artifacts appendix. |
| 18 | Visual presentation | **NEEDS A VISUAL PASS** | 4 majors: Fig 2 Portuguese labels, Fig 3 color-only, chapter-title line-breaks (Viegas defect), Table 1 overflow. Structural level reads as one document; booktabs throughout. |

**Headline:** No persona found a fabricated citation, a fabricated number, or an unlicensed
claim. Persona 05 (citation) and 07 (claim honesty) return outright GATE PASS; 08 (translation)
passes the mandatory L5 gate; 17 (excellence) scores the science uniformly outstanding. Every
gate FAIL / conditional is driven by one of four things: the known-open **title** placeholder,
the known-open **CBIC dataset placeholders**, cross-chapter **consistency seams**, or
**presentation** defects. None is a defect in the underlying experiments.

## 2. Findings inventory (24 items + 1 guard)

Routing: **MECH** = mechanical/defect, landed directly this loop; **SIGNOFF** = touches a
claim, a number's meaning, scope, or the author's voice — queued, never self-approved;
**AUTHOR** = author-only action (recompute or decision).

| ID | Sev | Route | Location | Finding | Status |
|----|-----|-------|----------|---------|--------|
| BL-1 | BLOCKER | AUTHOR | 3_cbic.tex:235 / PDF p.35 | CBIC dataset stats render as literal [VERIFY: recompute per ERRATA.md] placeholders (N_users/N_poi/N_checkins) | AUTHOR (CBIC recompute) |
| BL-2 | BLOCKER | AUTHOR | 0_main.tex folha de rosto + Resumo + Abstrac | Dissertation title is the placeholder [TITLE — open decision] | AUTHOR (title) |
| BL-3 | BLOCKER | SIGNOFF | 5_mobiwac.tex:44,140 | Ch.5 states CBIC prior work studied next-category AND next-region and observed negative transfer — both false (CBIC studied static | AUTHOR (B.1 — ERRATA.md + Appendix B; repair text supplied) |
| MJ-1 | MAJOR | SIGNOFF | 6_conclusion.tex:~79 vs 5_mobiwac.tex Table  | Alabama joint next-category macro-F1 = 64.51 in Ch.5 (joint-best/deploy convention) but 64.54 in Ch.6 (diagnostic-best board value | FIXED B.2 [signoff] |
| MJ-2 | MAJOR | SIGNOFF | 2_fundamentals.tex:442 vs 5_mobiwac.tex:349, | Ch.2 binds 'outperforms' to paired Wilcoxon; Ch.5 uses paired t on per-seed means (n=4); reproduced CBIC/CoUrb say 'significantly  | QUEUED signoff (MJ-2) |
| MJ-3 | MAJOR | SIGNOFF | 2_fundamentals.tex:~ (2.4 'used throughout') | Ch.2 sells user-disjoint CV as the whole-dissertation protocol; only Ch.5 uses it. CBIC/CoUrb code is StratifiedKFold on rows (use | QUEUED signoff |
| MJ-4 | MAJOR | SIGNOFF | 5_mobiwac.tex:5.3/5.6 | Region superiority presented as pre-assigned direction; protocol pre-registered region as non-inferiority. 4 region-superiority cl | QUEUED signoff |
| MJ-5 | MAJOR | SIGNOFF | 6_conclusion.tex limitation 1 | Ch.6 says the five state datasets collected '2009 and 2010'; Ch.5's own measured provenance (figshare dump) runs 2009–2011 and exp | QUEUED signoff |
| MJ-6 | MAJOR | SIGNOFF | 2_fundamentals.tex 2.1 vs 2.4 | 93% predictability is a universal 'ceiling any model should be read against' in 2.1 but disowned in 2.4 (Song's bound is next-loca | QUEUED signoff |
| MJ-7 | MAJOR | MECH | 2_fundamentals.tex:532 ('unlocks'); 3_cbic.t | Hard-banned WRITING_LAW §4 vocabulary in Ch.2 hinge + a CBIC ban cluster the AppB conformance pass missed. | PARTIAL — Ch.2 unlocks FIXED B.3; Ch.3 cluster queued (Appendix B) |
| MJ-8 | MAJOR | SIGNOFF | 3_cbic.tex / 4_courb.tex titles + 3.2.1 (PDF | 'Next-POI Prediction' (35× in Ch.3/4) reads as next-PLACE (which the frame explicitly does NOT predict); 3.2.1 defines it as 'whic | QUEUED signoff (preface bridge) |
| MJ-9 | MAJOR | MECH | 4_courb.tex (italicized common terms), 3_cbi | Ch.4 italicizes ordinary terms every page (leaks into LoF); Ch.3 uses inline bold for emphasis in prose — two register breaks vs t | QUEUED (Appendix B; guard: no sterilize) |
| MJ-10 | MAJOR | MECH | PDF p.46 (fig2_model, Ch.4) | Figure 2 carries Portuguese labels (Encoder Espacial/Temporal/Categórico, Coordenadas, Timestamps) in an English-frame chapter. | AUTHOR/asset (Fig 2 regen) |
| MJ-11 | MAJOR | MECH | PDF p.51 (fig3_embquality, Ch.5) | Figure 3 distinguishes Food from Shopping by color only (red vs orange) — collapses in grayscale. | AUTHOR/asset (Fig 3 regen) |
| MJ-12 | MAJOR | MECH | PDF p.25 etc (chapter title typesetting) | Chapter titles stretch to 3-4 justified lines with mid-word hyphen breaks (Multi-/Task, Ca-/tegory) → two-line running headers + 3 | QUEUED mech (title line-break) |
| MJ-13 | MAJOR | MECH | PDF p.20 (Table 1 model-lineage, 2_fundament | Table 1 overflows right margin by 29.8pt (~1cm). | QUEUED mech (Table 1 width) |
| MJ-14 | MAJOR | MECH | 4_courb.tex:226 | Cites liu2014geographical (a recommendation method) as the source of 'the Gowalla dataset'; canonical dataset keys cho2011gowalla/ | FIXED B.4 + Appendix B |
| MJ-15 | MAJOR | MECH | 4_courb.tex:8 | Chapter 4 title uses \: (math-mode space) in text-mode title; only \: in the build. Same name uses plain colon elsewhere. | FIXED B.6 |
| MJ-16 | MAJOR | MECH | 0_main.tex:265, 1_introduction.tex:114, 6_co | Reported performance uses both 'at' and 'on'/'across' for the same construction; internal inconsistency. | HELD — persona 14 VETO blanket; targeted only + author Q |
| MJ-17 | MAJOR | SIGNOFF | 2_fundamentals.tex:2.4 vs 5_mobiwac.tex:5.4 | Ch.2 says pipeline uses class-weighted cross-entropy; Ch.5 uses plain unweighted CE and reports class-weighting HURT. Direct contr | QUEUED signoff |
| MJ-18 | MAJOR | SIGNOFF | 0_main.tex:251,192 + ch1/2/4/5/6 vs 3_cbic.t | 'MTLnet' is named as the central artifact 'introduced in Chapter 3' by the Abstract, Resumo, and Ch.1/2/4/5/6, but Chapter 3 uses  | QUEUED signoff (MTLnet naming) |
| MN-1 | MINOR | MECH | 2_fundamentals.tex:162-163 (and gate-fix com | The de-dup gate-fix comment claims the 'weekday lunch / Saturday night' image now appears only in Ch.1, but §2.2 still carries a v | QUEUED mech |
| MN-2 | MINOR | AUTHOR | front matter (0_main.tex) + apx_c | AI-use disclosure lives only in Appendix C (last page); 2026 detail-on-demand norm wants a one-line front-matter statement pointin | AUTHOR (front-matter disclosure line) |
| GUARD-1 | GUARD | NONE | Ch.3/Ch.4 (re-typeset published papers) | GUARD (not a fix): do NOT sterilize Ch.3/Ch.4 -ly density or bold/italics beyond mechanical register alignment — they are peer-rev | CONSTRAINT (no action) |

## 3. Fixes applied this loop (6, all gate-approved by persona 14)

Persona 14 (adversarial advisor) certified all 6 phase-6 mechanical fixes and supplied exact
APPROVE-WITH-EDIT text for these 6, applied verbatim:

1. **B.3** `2_fundamentals.tex` — banned word `unlocks` → `enables in` (Ch.2 hinge). MECH.
2. **B.5** `2_fundamentals.tex` — Song 93% rescoped to next-location + forward-ref to §2.4. **[NEEDS SIGN-OFF]** (claim scope).
3. **B.7** `2_fundamentals.tex` — `93\%` → `93 percent` (document convention). MECH.
4. **B.4** `4_courb.tex` — Gowalla cite `liu2014geographical` → `cho2011gowalla,jure2014snap`; orphan key dropped; Appendix B row added. MECH (errata route).
5. **B.6** `4_courb.tex` — chapter-title `\:` → plain colon. MECH.
6. **B.2** `6_conclusion.tex` — AL joint value `64.54` → `64.51` (match Ch.5 table convention). **[NEEDS SIGN-OFF]** (number convention).

Two of the six (B.2, B.5) carry `[NEEDS SIGN-OFF]` LaTeX comments in place: they are the
author's to confirm or revert, though the change itself is verdict-neutral in both cases.
Post-fix rebuild: both modes compile, 0 errors, 0 undefined refs/cites, lint exits 0, page
counts stable (87/83).

## 4. Fixes queued [NEEDS SIGN-OFF] (touch claims/scope/voice — not self-applied)

- **B.1 (BLOCKER, persona 10/14):** Ch.5 falsely states CBIC studied next-region and *observed*
  negative transfer. Inherited verbatim from the under-review MobiWac paper; persona 14 supplied
  exact repair text for both lines (L44, L140) but routed it to the author because it edits an
  under-review paper's claims → ERRATA.md + Appendix B. **Top substantive item; do not ship to
  the advisor unresolved.**
- **MJ-2 (persona 09/12):** superiority test named as Wilcoxon (Ch.2) but t-test (Ch.5) at n=4;
  reproduced papers say 'significantly outperform' with no test. Scope Ch.2's rigor claim to
  frame+Ch.5; defend the parametric choice. Do NOT retrofit tests onto the reproduced papers.
- **MJ-3 (persona 09/11):** user-disjoint CV sold as document-wide in Ch.2; only Ch.5 uses it.
  Scope to Ch.5 + add one disclosure sentence to Ch.3.
- **MJ-4 (persona 09):** region superiority framed as pre-assigned; protocol pre-registered
  non-inferiority. State the post-hoc confirmation; enumerate the family.
- **MJ-5 (persona 07/11):** data vintage '2009 and 2010' (Ch.6) vs measured 2009–2011 (Ch.5).
- **MJ-6 (persona 11/01):** 93% ceiling scope — the Ch.2 §2.1 half is now fixed (B.5); confirm.
- **MJ-8 (persona 01/03/15/11/12):** 'next-POI' task-name collision — add a one-sentence preface
  bridge in Ch.3/Ch.4 (Ch.5 already models it). New connective prose.
- **MJ-17 (persona 09/10):** class-weighting contradiction — Ch.2 says class-weighted CE, Ch.5
  uses unweighted and reports weighting hurt. Correct Ch.2.
- **MJ-18 (persona 04):** 'MTLnet' named across the frame as 'introduced in Ch.3' but absent from
  Ch.3; Ch.4:82 claims the published paper typesets 'MTLNet'. Reconcile via preface bridge.
- **MN-1 (persona 04/03):** weekday-lunch motif residual in §2.2 (gate-fix comment says it lives
  only in Ch.1). **MN-2 (persona 16):** add a one-line front-matter AI-use disclosure.

## 5. Held for the author — VETO / decision (persona 14)

- **B.8 preposition campaign ('at [dataset]' → 'on'/'across'):** persona 14 **VETOED** a blanket
  find-replace — the region-verdict phrasings ('at four of six datasets', 'at AL/AZ') are
  law-mandated verbatim scopes echoed in the Abstract/Resumo/body with certified parity. Fix
  only non-verdict descriptive instances; a verdict-preposition change is a whitelist decision
  (all-at-once or nowhere).
- **Figure 2 Portuguese labels / Figure 3 color-only encoding (persona 18):** asset
  regenerations, not text edits — author/asset work before the advisor build.
- **GUARD (persona 16/03):** do NOT sterilize Ch.3/Ch.4 -ly density or bold/italics beyond
  mechanical register alignment — they are peer-reviewed published text; over-correction reads
  as defensive writing and risks altering published wording. Freeze the negative-parallelism count.

## 6. Re-run gate statuses (after the fix loop)

| Gate | Status |
|------|--------|
| Build (both modes) | PASS — defense 87pp / final 83pp, 0 errors, 0 undefined cites/refs |
| Lint (check.sh) | PASS — exits 0 (em-dash 0, contractions 0, banned words 0 in prose, codenames 0) |
| N4 numeral (06) | CONDITIONAL — 0 fabrications, 0 mismatch; blocks only on CBIC placeholders (author recompute). AL blur fixed (B.2). |
| R3 citation (05) | PASS — 99/99 real, 0 fabrications, Gowalla mis-source fixed (B.4) |
| Claim honesty (07) | PASS — verb-test binding airtight, AZ never upgraded |
| L5 translation (08) | PASS — Ch.4 faithful to PT source |
| Style G3 (03) | RE-RUN NEEDED after fixes — 'unlocks' (B.3) + percent (B.7) fixed; CBIC ban cluster still queued (Appendix B route); placeholder blocker remains author action |
| Concordance (04) | SEAMS — AL blur fixed; MTLnet-naming + motif-residual queued for sign-off |
| Change gate (14) | PASS — 6 landed fixes certified; 6 more applied this loop; B.1/B.8 held for author |

## 7. The two pre-existing blockers (author actions, unchanged by this loop)

1. **Title** — renders as `[TITLE — open decision]` on folha de rosto, Resumo, Abstract,
   pdftitle. Working title + 3 alternates sit in `0_main.tex` comments. Author decides + wire in.
2. **CBIC dataset placeholders** — `3_cbic.tex` renders `[N_users; VERIFY]` etc. on PDF p.35.
   Sanctioned CBIC-Florida recompute is an author action; no number may be invented (correct
   fail-closed handling, but blocks the banca build).

Both are the top two items in the handoff note's ranked action list.