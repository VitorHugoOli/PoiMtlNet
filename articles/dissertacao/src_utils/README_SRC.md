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
`tex-gyre`, `txfonts`. On this machine those live in a **usermode tree** at `$HOME/Library/texmf`,
not in the system TeX Live tree, and `pdflatex` itself is at `/Library/TeX/texbin`, which is not on
a non-interactive `PATH`.

**`source src_utils/texenv.sh` before `make`.** It sets the four variables and records why each is
needed. Two distinct failures if you skip it, and the second one misleads:

| Missing | Symptom | What it actually is |
|---|---|---|
| `TEXMFHOME` | `LaTeX Error: File 'abntex2.cls' not found` | honest: the class is not on the path |
| `TEXMFVAR` | `!pdfTeX error: Font ntx-Regular-tlf-ot1r at 657 not found ==> Fatal error occurred, no output PDF file produced!` | **not** a missing font. Both the `.tfm` and the `.pfb` are present in the home tree. What is missing is the font **map**: newtx registers its 36 `ntx-*` entries in the usermode updmap output at `$TEXMFHOME/.texmf-var/fonts/map/pdftex/updmap/pdftex.map`, and the system map has none. `kpsewhich -var-value TEXMFVAR` reports an unreadable path here, so the value cannot be probed and must be set. |

`build.sh` carries the same defaults so it works when invoked directly.

## Verifying a build

Run **both** tools and read `tex_errors`:

```
source src_utils/texenv.sh
(cd src && make defense && make final)   # -halt-on-error -> the honest pass/fail
./src_utils/build.sh src both            # the report, including tex_errors
```

`build.sh` runs `-interaction=nonstopmode`, under which pdflatex **recovers** from an error and still
writes a PDF. That is how commits `6d780b58` through `a880632b` shipped "104/99 pp, 0 overfull,
0 undefined" out of a source tree carrying `! Extra }, or forgotten \endgroup` (a brace lost from the
`{\small ...}` group in `tables/frame/bib_errata.tex` during the tables reorganization). `make`
produced nothing that whole time; nobody ran it. `build.sh` now reports `tex_errors=N` and fails on
it. **A PDF existing is not evidence the source is correct.**

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
