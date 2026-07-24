# src/ — the dissertation working copy

Single-source LaTeX for Vitor H. O. Silva's UFV/PPGCC dissertation. Skeleton derived from the
Germano tree (`../exemples/germano/`, same advisor) per TEMPLATE.md §0; the kept-vs-stripped
ledger is the header comment of `0_main.tex`.

## Layout (restructured 2026-07-24)

```
src/
  main.tex            single entry point (defense build by default; see the header comment)
  0_main.tex          the document body (preamble + front matter + \include list)
  abntex2-UFV.sty     UFV machinery (Germano tree)
  abntex2-num.bst     numeric bibliography style
  references.bib      single global bibliography
  chapters/           the six chapters + three appendices + adaptation ledgers
  figures/ tables/    chapter assets
  dissertacao.pdf     the current DEFENSE build (the only PDF at the root; copied from build/)
  Makefile            build + lint targets
  build/              ALL compile output lands here (gitignored)
  src_utils/          non-LaTeX: this README, check.sh, reports, review outputs, handoff JSON
```

`src_utils/` holds everything that is not part of the compiled document: `README_SRC.md`,
`check.sh` (lint), `BIB_MERGE_REPORT.md`, `HANDOFF_v1.md`, the `_gates/` and `_review_v1/` review
reports, the `handoff/` working JSON, and `cbic_recompute_result.md`. The pt_BR author-decisions
document also lives here.

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
