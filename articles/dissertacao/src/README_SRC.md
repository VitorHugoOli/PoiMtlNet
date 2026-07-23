# src/ — the dissertation working copy (v1 assembly)

Skeleton derived from the Germano tree (`../exemples/germano/`, defended 2024, same advisor)
per TEMPLATE.md §0. The kept-vs-stripped ledger is the header comment of `0_main.tex`.

## Build

Two build modes (UFV_COMPLIANCE §1), one source:

```
make defense   # -> main_defense.pdf  (full front matter; the banca's PDF)
make final     # -> main_final.pdf   (AcademicoPG upload: lists -> sumario -> body)
make check     # lint hook: em-dashes, "this paper", contractions, banned words,
               # repo codenames, unresolved \ref/\cite
```

Requirements: TeX Live (tested: 2026 basic, `/Library/TeX/texbin`) with `abntex2`, `newtx`,
`fontaxes`, `xstring`, `enumitem`, `multirow`, `adjustbox`, `collectbox`, `xkeyval`, `subfig`,
`lastpage`, `textcase`, `xurl`, `kastrup` available. On this machine they live in a usermode
tree; export `TEXMFHOME` pointing at it before `make` if TeX cannot find `abntex2.cls`.

## Compliance decisions baked in

- Font: `newtxtext,newtxmath` (Times; manual §8); abnTeX2 heading fonts remapped to Times bold.
- Citations: `abntex2cite [num]` — the settled single global numeric scheme (decision #5).
- Margins: 3 cm top/left, 2 cm bottom/right via `abntex2-UFV.sty` (verified vs manual §7).
- Spacing: `\OnehalfSpacing` (manual §7).
- Page numbers: top-right (abnTeX2 default header), arabic.
- Final build: pre-body pages empty pagestyle; body page counter set by
  `\finalbuildfirstpage` in `main_final.tex` — [VERIFY] tune against the AcademicoPG
  RASCUNHO PDF after the post-defense upload.
- Chapter preface: `\begin{chapterpreface}...\end{chapterpreface}` — the italic
  time-capsule paragraph (NORTH_STAR §3).

## Layout

```
main_defense.tex / main_final.tex   entry points (set \ifdefensebuild)
0_main.tex                          the document (preamble + front matter + \include list)
abntex2-UFV.sty, abntex2-num.bst    UFV machinery (Germano tree)
chapters/1_introduction.tex ... 6_conclusion.tex + apx_{a,b,c}_*.tex
references.bib                      single global bibliography (Phase 4 merges all donors)
figures/, tables/                   chapter assets (Phase 2)
check.sh, Makefile                  lint + build
```
