# HANDOFF — Dissertation v1 (assembled 2026-07-24)

> The single handoff document for the machine-assembled v1. Author owns and approves every
> word; this note is the map, the flag ledger, and the ranked to-do list. Nothing here was
> self-approved past the gates. Repo working copy: `articles/dissertacao/src/`.

## 0. READ THIS FIRST — the two things that block the banca build

1. **Decide the title.** It renders as `[TITLE --- open decision]` on the folha de rosto,
   Resumo, Abstract, and PDF metadata. A **working title + three alternates** are already
   reconciled in `src/0_main.tex` (header comment) and `src/chapters/1_introduction.tex`
   (your 2026-07-23 decision block). Pick one; update all four echo points (they are listed
   in the `0_main.tex` comment).
2. **Run the CBIC dataset-count recompute.** Chapter 3 §3.4.1 (PDF p.35) prints three literal
   `[N_users; VERIFY: recompute per ERRATA.md]` placeholders instead of the Florida corpus
   size. No number was invented (correct fail-closed handling). Run the sanctioned CBIC-era
   Florida recompute, approve the values, and fill them in. **This is the single most visible
   defect in the built PDF and every fact/style/domain/banca reviewer flagged it.**

**⚠ AND — tell the advisor about how v1 was produced.** The plan assumed six human drafting
days (Jul 19–24); the build was instead done by the assistant in one automated session and
lands as a *machine-assembled v1 for you to read and own*. The reading map (this note) + the
AI-use disclosure (Appendix C) must accompany the PDF. The review suite ran on `claude-opus-4-8`
rather than the planned Fable model (Fable tokens were exhausted mid-run; you approved the
switch). This is disclosed here, in PLAN.md rev-3, in CLAUDE.md §1, and in Appendix C.

## 1. Phase-0 gap table — final state

Every planned component, its Phase-0 status, and where it landed in v1.

| Component | Phase-0 | v1 state | Where |
|---|---|---|---|
| Skeleton (Germano tree, 2 builds) | missing | BUILT, compiles | `src/0_main.tex + main_{defense,final}.tex` |
| Ch.1 Introduction | draft exists | IMPORTED | `src/chapters/1_introduction.tex` |
| Ch.2 Fundamentals (+lineage table) | draft exists | IMPORTED (inlined) | `src/chapters/2_fundamentals.tex` |
| Ch.3 CBIC re-typeset | paper source | RE-TYPESET | `src/chapters/3_cbic.tex (+ ADAPTATION_LEDGER)` |
| Ch.4 CoUrb EN re-typeset | EN source | RE-TYPESET, L5 PASS | `src/chapters/4_courb.tex (+ ADAPTATION_LEDGER)` |
| Ch.5 MobiWac re-typeset | VoR source | RE-TYPESET, re-synced (no drift) | `src/chapters/5_mobiwac.tex (+ ADAPTATION_LEDGER)` |
| Ch.6 Conclusion | draft exists | IMPORTED | `src/chapters/6_conclusion.tex` |
| Global bibliography (numeric) | 3 donor bibs | MERGED, 99 entries, 0 dangling | `src/references.bib (+ BIB_MERGE_REPORT)` |
| Front matter (Resumo/Abstract/lists) | planned | DRAFTED [sign-off] | `src/0_main.tex` |
| Appendix A (BRACIS contribution) | planned | DRAFTED [sign-off] | `src/chapters/apx_a_contributions.tex` |
| Appendix B (errata reconciliation) | 3 ERRATA.md | COMPILED | `src/chapters/apx_b_errata.tex` |
| Appendix C (AI-use disclosure) | planned | DRAFTED [sign-off] | `src/chapters/apx_c_ai_disclosure.tex` |
| Full-document gates (N4/R3/L3/L4/§7) | planned | RUN + fixed | `src/_gates/{N4_R3_REPORT,L3_L4_STYLE_BUILD_REPORT}.md` |
| 18-persona review suite | planned | RUN (Opus 4.8) | `src/_review_v1/*.md + CONSOLIDATED_REVIEW_REPORT.md` |
| Title | OPEN by design | STILL OPEN (blocker #1) | `placeholder + candidates in 0_main.tex` |
| CoUrb figure distribuicao_estados.png | thought absent | PRESENT (plan note was stale) | `used in Ch.4` |

**No BLOCKING gap prevented assembly** (Phase-0 verdict was GO). The two remaining blockers
(title, CBIC counts) are author actions known-open from the start, not assembly failures.

## 2. What was built where (+ ledger pointers)

- **Build:** `cd src && make defense` (→ `main_defense.pdf`, 87 pp, banca PDF) / `make final`
  (→ `main_final.pdf`, 83 pp, AcademicoPG upload). Recipe + TeX-tree notes: `src/README_SRC.md`.
- **Numbers/citations ledgers:** per-chapter `src/chapters/{3,4,5}_*_ADAPTATION_LEDGER.md`
  (every departure from the published text, feeding Appendix B); `src/BIB_MERGE_REPORT.md`
  (99-entry key-mapping table + provenance + errata applied); Ch.1/6 citation ledgers in the
  frozen `storyline/drafts/{1,6}_citations.md`; Ch.2 in `fundamentals/DRAFT_LEDGER.md`.
- **Gate + review evidence:** `src/_gates/` (N4 numeral, R3 citation, L3/L4/style/build);
  `src/_review_v1/` (18 persona reports + `CONSOLIDATED_REVIEW_REPORT.md`).
- **Freeze:** `storyline/` and `fundamentals/` are FROZEN (READMEs carry the pointer); `src/`
  is the single working copy. Edit `src/chapters/` and rebuild — never the draft folders.
- **Commits (this build, `draft(ai):` prefix):** phase0b `415c5cd3`, skeleton `a735b8f3`,
  ch3-5 `1a29b545`, frame `70e794f1`, bib `80fb133c`, front/back `0ccbdebe`, phase-6 gate fixes
  `49a67996`, phase-7 fix loop `5374d511`, reviews `9742ea5c`, phase-8 docs `bb64e220`.

## 3. Ranked flag ledger — [VERIFY] and [NEEDS SIGN-OFF] (title first)

### Tier 1 — BLOCKERS (must clear before the banca build)
1. **[TITLE]** open decision — folha de rosto + Resumo + Abstract + pdftitle. Candidates in source. **Author decides.**
2. **[VERIFY] CBIC dataset counts** — `3_cbic.tex:235`, renders in PDF p.35. Sanctioned recompute. **Author runs.**
3. **[NEEDS SIGN-OFF] B.1 Ch.5 CBIC misattribution** — `5_mobiwac.tex:44,140` say CBIC studied
   next-region and *observed* negative transfer (both false; inherited from the under-review
   paper). Repair text ready in `_review_v1/14_adversarial_advisor_report.md` §B.1 → ERRATA route. **Author approves.**

### Tier 2 — [NEEDS SIGN-OFF] drafted content (approve or revise; the text compiles as-is)
4. **Resumo (PT) + Abstract (EN)** — `0_main.tex:174,236`, claim-parity pair from Ch.1/6 (certified parallel by 03/08).
5. **Appendix C AI-use disclosure** — `apx_c_ai_disclosure.tex:11`, one page, from the git trail; confirm scope + add the Opus-4.8 deviation line.
6. **Appendix A (BRACIS)** + **Appendix B (errata)** — `apx_a:17`, `apx_b:20`, new frame prose around quoted content.
7. **Ch.5 preface + dual recap + restored embquality figure** — `5_mobiwac.tex:18,82,417`, mandated new-to-chapter text.
8. **Gate-fix rewordings:** B-5 Song-93%-scope `2_fundamentals.tex:40`; B-2 64.51 convention `6_conclusion.tex:79`;
   L3 dedup A-1/A-2 `1_introduction.tex:79` + `2_fundamentals.tex:512`. All verdict-neutral; revert paths in the comments.

### Tier 3 — queued fixes not yet applied (review suite; need your call)
9.  **MJ-2 superiority-test naming** — Ch.2 says Wilcoxon, Ch.5 uses paired t at n=4. Scope to frame+Ch.5; defend the choice.
10. **MJ-3 user-disjoint CV scope** — Ch.2 sells it document-wide; only Ch.5 uses it. Scope + add one Ch.3 disclosure sentence.
11. **MJ-4 region pre-registration framing** — state region was pre-registered non-inferiority; superiority confirmed post-hoc.
12. **MJ-5 data vintage** — Ch.6 '2009 and 2010' vs measured 2009–2011. **MJ-6 93% ceiling** — Ch.2 §2.1 half fixed (B.5); confirm.
13. **MJ-8 'next-POI' task-name bridge** — one preface sentence in Ch.3/Ch.4 (Ch.5 already models it).
14. **MJ-17 class-weighting** — Ch.2 says class-weighted CE; Ch.5 unweighted + reports weighting hurt. **MJ-18 'MTLnet' naming seam** (persona 04).
15. **Visual:** Fig 2 Portuguese labels (regen EN); Fig 3 color-only Food/Shopping (regen grayscale-safe); chapter-title line-breaks; Table 1 overflow (~1cm).
16. **VETOED — do NOT run mechanically:** the 'at [dataset]' preposition sweep (persona 14 VETO — collides with the frozen verdict scopes; fix only non-verdict instances).

## 4. Gate statuses (after the fix loop)

| Gate | Status |
|---|---|
| Build both modes | **PASS** — 87/83 pp, 0 errors, 0 undefined refs/cites |
| Lint (check.sh) | **PASS** — exits 0 (em-dash 0, contractions 0, banned words 0, codenames 0) |
| N4 numeral (06) | **CONDITIONAL** — 0 fabrications, 0 mismatch; blocks only on CBIC placeholders |
| R3 citation (05) | **PASS** — 99/99 real, 0 fabrications, Gowalla mis-source fixed |
| Claim honesty (07) | **PASS** — verb-test binding airtight, AZ never upgraded |
| L5 translation (08) | **PASS** — Ch.4 faithful to the PT source |
| Style G3 (03) | **RE-RUN NEEDED** — 'unlocks'/percent fixed; CBIC ban cluster queued (Appendix B) |
| L3/L4 + WRITING_LAW §7 | **PASS** — em-dash 0, cross-refs resolve, parity certified |
| Concordance (04) | **SEAMS** — AL blur fixed; MTLnet-naming + motif residual queued |
| Change gate (14) | **PASS** — 6 landed fixes certified, 6 more applied, B.1/B.8 held for you |
| Two-build (UFV 13) | defense non-compliant ONLY on the title placeholder; final compliant with conditions; all measured rules pass |

## 5. Review-suite verdicts (18 personas, on claude-opus-4-8)

Full per-persona detail: `src/_review_v1/CONSOLIDATED_REVIEW_REPORT.md`. Headline: **no
fabricated citation, number, or unlicensed claim anywhere.** GATE PASS from 05 (citation), 07
(claim honesty), 08 (L5 translation); 17 (excellence) scores the science 6 OUTSTANDING / 4 GOOD
/ 0 BELOW; 12 (banca) returns *aprovado com correções menores* (45/50). Every FAIL/conditional
traces to the title, the CBIC placeholders, cross-chapter seams, or presentation — not the
experiments. Optional excellence moves (persona 17, for the SBC-CTD lens): a contributions→claims
table in §1.6, a consolidated cross-chapter results table in §6, and an artifacts/reproducibility
appendix. None is a defect; all are frame-only enhancements.

## 6. Author to-do list (in order)

1. **Decide the title** and wire it into the four echo points in `0_main.tex`.
2. **Run the CBIC Florida recompute**, approve, and fill the three counts in `3_cbic.tex:235`.
3. **Approve or revise B.1** (Ch.5 CBIC misattribution) — repair text in the persona-14 report; log to ERRATA.md + it flows to Appendix B.
4. **Read + approve the Tier-2 [NEEDS SIGN-OFF] content** (Resumo/Abstract, Appendices A/C, Ch.5 new-to-chapter text, gate rewordings).
5. **Rule on the Tier-3 queued fixes** (superiority test, CV scope, region framing, data vintage, task-name bridge, class-weighting, figures).
6. **Message the advisor** about the machine-assembled v1 (this note + Appendix C) and the model deviation; confirm banca + Art. 22 timing.
7. **Re-sync note:** Ch.5 currently has NO drift vs `[mobiwac]/src/`; if you edit the paper after today, re-run the diff before the advisor build.

_All writes are under `articles/dissertacao/`. Source paper folders and law-file rules were not edited._