# Calibration re-check — assembled v1 vs exemplars + Locus findings (2026-07-24, Phase 8)

> Phase-8 verification step. The Locus survey ran pre-skeleton (Phase 0b,
> `locus_calibration_2026-07-23.md`); this note confirms the skeleton-time choices held through
> assembly. No new web search was run (the timebox was spent in Phase 0b); this is a re-check of
> the ASSEMBLED document against the Phase-0b note, VIEGAS_ANALYSIS.md, and the Germano tree.

## Verdict: all five Phase-0b fold-in decisions held; no ledger conflict; nothing to reopen.

| Phase-0b consequence | Assembled v1 state | Held? |
|---|---|---|
| 1. Germano tree as base | `abntex2-UFV.sty` + `abntex2-num.bst` + memoir machinery, content stripped | YES |
| 2. Coletânea layout (Introdução → artigos → Conclusão Geral) | `\include` order: 1 intro, 2 fundamentals, 3 CBIC, 4 CoUrb, 5 MobiWac, 6 conclusion; Conclusão Geral present | YES |
| 3. Catalog-header Resumo (AUTHOR, M.Sc., UFV, month year. Title. Orientador:) | Rendered verbatim in both Resumo and Abstract (title still the open placeholder) | YES |
| 4. Single global numeric citations (Viegas precedent; author-year also accepted) | `abntex2cite [num]`, one `references.bib`, 99 entries, 0 dangling | YES |
| 5. "Estrutura da Dissertação" section at end of Intro | Present in Ch.1 | YES |

## Exemplar cross-check (assembled v1)

- **Viegas (numeric, coletânea, same advisor, EN):** the v1 matches its structural devices —
  coletânea declared in the introduction, one bold research question, one global numeric
  bibliography, per-chapter figure/table numbering. Viegas omits the PT Resumo (pre-deposit
  build); the v1 defense build correctly INCLUDES Resumo PT + Abstract EN, which is the right
  choice for the defense PDF (the AcademicoPG final build strips them per UFV_COMPLIANCE §1, and
  the v1's `main_final.tex` does exactly that).
- **Germano (defended 2024, same advisor, EN body):** the v1 keeps its front-matter machinery
  and the non-article review chapter precedent (persona 13 confirmed the Fundamentals chapter has
  precedent in this exact tree). No divergence.

## Improvements folded in (none conflicting with the decisions ledger)

None required. Every concrete improvement the Locus survey suggested was already a settled
decision; the assembled document realizes them. Persona 17 (excellence) proposed three
value-add artifacts (contributions→claims table, consolidated cross-chapter results view,
artifacts appendix) — these are enhancements beyond the calibration baseline, listed in the
handoff as optional author moves (SBC CTD lens), NOT calibration gaps.

## Confirmed against the decisions ledger

No skeleton-time choice was reopened. The one place program practice differs from the settled
decision (Canesche uses ABNT author-year; the v1 uses numeric) was already resolved in Phase 0b:
both schemes have accepted UFV PPGCC precedent, and decision #5 (numeric, Viegas-style) stands.
