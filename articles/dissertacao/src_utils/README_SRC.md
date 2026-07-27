# src/ — the dissertation working copy

Single-source LaTeX for Vitor H. O. Silva's UFV/PPGCC dissertation. Skeleton derived from the
Germano tree (`../exemples/germano/`, same advisor) per TEMPLATE.md §0; the kept-vs-stripped
ledger is the header comment of `0_main.tex`.

## Layout (restructured 2026-07-24; src_utils/ externalized 2026-07-24)

`src/` is kept as clean as possible so it can be pasted straight into Overleaf: it contains ONLY
what compiles. All support material lives in `src_utils/`, a SIBLING of `src/` (one level up),
never inside it.

```
articles/dissertacao/
  src/                        <-- paste THIS into Overleaf; compiles standalone
    main.tex                  single entry point (defense build by default; see header comment)
    0_main.tex                the document body (preamble + front matter + \include list)
    abntex2-UFV.sty           UFV machinery (Germano tree)
    abntex2-num.bst           numeric bibliography style
    references.bib            single global bibliography
    chapters/                 the six chapters + three appendices (.tex ONLY)
    figures/ tables/          chapter assets
    dissertacao.pdf           current DEFENSE build (only PDF at the root; copied from build/)
    Makefile                  build + lint targets
    build/                    ALL compile output lands here (gitignored)
  src_utils/                  <-- SIBLING of src/; NOT pasted into Overleaf
    README_SRC.md             this file
    check.sh                  lint hook (linted target is ../src)
    BIB_MERGE_REPORT.md       bibliography merge/key-mapping record
    _archive/handoffs/HANDOFF_v1.md             author handoff note
    _archive/reviews_v1/DECISOES_PENDENTES_ptBR.md  author decisions (pt-BR)
    cbic_recompute_result.md  + _archive/handoffs/cbic_recompute_handoff.json  (CBIC dataset counts)
    adaptation_ledgers/       3_cbic / 4_courb / 5_mobiwac ADAPTATION_LEDGER.md (feed Appendix B)
    _archive/reports_2026-07/FRAME_INTEGRATION_REPORT.md
    _gates/ _review_v1/ _specialists_v1/  gate + review + specialist reports
    handoff/                  working JSON
```

Nothing in `src/` `\input`s or references anything in `src_utils/` (the ledger/report references
in the chapter sources are provenance comments only), so pasting `src/` alone into Overleaf
compiles with no missing files.

## Build (two modes, one source; UFV_COMPLIANCE §1)

```
make            # or `make defense` -> build/main.pdf, copied to ./dissertacao.pdf (banca PDF)
make final      # -> build/main_final.pdf (AcademicoPG body-only upload)
make check      # lint: em-dashes, "this paper", contractions, banned words, codenames, undefined refs
make clean      # empty build/
```

The mode is one `\ifdefensebuild` switch in `main.tex` (default = defense). `make final` sets
`\FINALBUILD` on the command line before reading `main.tex`, so the body-only build needs **no
second main file**. UFV: two BUILDS are required (full-front-matter defense PDF + body-only
AcademicoPG upload); the manual governs the submission and the final PDF, not the LaTeX source, so
one main file is compliant.

Output goes to `build/` via `-output-directory=build`; input paths (`figures/`, `chapters/`)
resolve relative to the src root regardless, so no `graphicspath` change is needed. The Makefile
pre-creates `build/chapters/` because `\include` writes per-chapter `.aux` there.

## TeX tree

Requires TeX Live with `abntex2`, `newtx`, `fontaxes`, `xstring`, `enumitem`, `multirow`,
`adjustbox`, `collectbox`, `xkeyval`, `subfig`, `lastpage`, `textcase`, `xurl`, `kastrup`,
`tex-gyre`, `txfonts`. On the build machine they live in a usermode tree in the agent workspace;
export `TEXMFHOME`/`TEXMFVAR`/`TEXMFCONFIG` at it before `make` if TeX cannot find `abntex2.cls`.

## Compliance decisions baked in

- Font: `newtxtext,newtxmath` (Times; manual §8); abnTeX2 heading fonts remapped to Times bold.
- Citations: `abntex2cite [num]` — single global numeric scheme (decision #5, Viegas-style).
- Margins: 3 cm top/left, 2 cm bottom/right (`\setlrmarginsandblock`/`\setulmarginsandblock` +
  `\checkandfixthelayout[fixed]` under `\OnehalfSpacing`, verified vs manual §7).
- Spacing: `\OnehalfSpacing` (manual §7). Page numbers: top-right arabic. A4.
- Title: set to the working option; alternates in the `0_main.tex` comment (pending advisor).
- Final build: pre-body pages empty pagestyle; body counter `\finalbuildfirstpage` in `main.tex`
  ([VERIFY] tune against the AcademicoPG RASCUNHO PDF post-upload).
- Chapter prefaces: `\begin{chapterpreface}...\end{chapterpreface}` (the italic time-capsule).
