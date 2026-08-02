# ERRATA — CoUrb 2026 / SBRC (this folder's article)

> **Article:** "ST-MTLNet: Representacoes Espaco-Temporais de Pontos de Interesse para Aprendizado
> Multitarefa." CoUrb 2026 (SBRC workshop), DOI 10.5753/courb.2026.22960, pp. 323-336. Published (PT).
> Tarik S. Paiva 1st author; Vitor 2nd author and presenter.
>
> **Purpose.** Defects known in the published article, corrected **silently in the re-typeset /
> translated dissertation chapter** (Ch.4) and listed in Appendix B (NORTH_STAR §4, decision #7).
> This file is the living record for this folder.

## Number / claim errata (use the audited numbers in the dissertation)

| # | Place | Defect | Fix |
|---|---|---|---|
| 1 | Results prose | Text says "16/21 (76%)" wins on the sequential task (its "next-POI prediction" = canonical next category). | The internal audit recounted **15/21 strict wins + 1 technical tie**. Use the audited count. |
| 2 | Results prose | Text says "+20-24 pp" category gains. | Per-state means are **+20.2 ... +22.0 pp**. Use the audited range. |
| 3 | `silva2025mtlnet` bib entry | Venue name wrong: "Proceedings of the Brazilian Conference on Intelligent Systems (CBIC)". That conflates two venues (BRACIS = Brazilian Conference on Intelligent Systems; **CBIC = Congresso Brasileiro de Inteligencia Computacional**). Also `note = {Submetido}` is stale (CBIC 2025 is published). | Venue → CBIC (Congresso Brasileiro de Inteligencia Computacional), 2025; DOI 10.21528/CBIC2025-1191324; drop the "Submetido" note. |

## Honesty items (state, do not hide — they strengthen the arc)

- The CoUrb split is **stratified by sample, not user-disjoint** (weaker than Ch.5's protocol). Say so explicitly
  in the dissertation; it is part of the honest arc, not a defect to conceal.
- Nash-MTL caveat as in Ch.3 (solver-bug time-indexing).
- No external baselines in the CoUrb study — scope the claims accordingly.

## Authorship note
State Vitor's contribution where the Comissao wants it: the baseline model MTLnet is his 1st-author work (CBIC),
and he presented the CoUrb paper.
