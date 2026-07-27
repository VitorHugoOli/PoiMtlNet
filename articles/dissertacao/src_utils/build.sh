#!/bin/bash
# Build the dissertation and verify the result honestly.
#
#   usage: src_utils/build.sh <srcdir> [defense|final|both]
#   e.g.   src_utils/build.sh src both
#
# This lived in /tmp for one whole working session and was swept mid-session, which silently turned
# a "rebuild and check" step into "check a stale PDF". It belongs in the repository.
#
# VERIFICATION RULES, each learned by shipping the corresponding defect:
#   - LaTeX WRAPS warnings at 79 columns, so the log MUST be flattened before matching. A
#     line-anchored grep reported "0 undefined citations" while four rendered as (??) in both PDFs.
#   - The .blg must be read separately: a BibTeX error (e.g. a bare at-sign inside a % comment in
#     references.bib) never reaches the .log.
#   - "Text page N contains only floats" is NOT emitted for every floats-only page, so those are
#     measured from the PDF text layer instead.
#   - Report ONLY the variants this invocation actually built; a stale build/<stem>.log was
#     otherwise presented as a fresh measurement.
#   - FAIL LOUDLY if a build produced no PDF. A silent build failure is how a stale PDF gets
#     audited in place of the real one.
#
# The TeX font map must come from the user tree; without TEXMFVAR/TEXMFCONFIG set, pdflatex dies
# with "Font t1xtt ... not found".
set -u
export PATH=/Library/TeX/texbin:$PATH
export TEXMFHOME="${TEXMFHOME:-$HOME/Library/texmf}"
export TEXMFVAR="${TEXMFVAR:-/tmp/texmfvar}"
export TEXMFCONFIG="${TEXMFCONFIG:-/tmp/texmfconfig}"
mkdir -p "$TEXMFVAR" "$TEXMFCONFIG"
# Regenerate the user font map if it is missing (a swept /tmp loses it).
if ! kpsewhich pdftex.map >/dev/null 2>&1 || ! grep -q "^t1xtt" "$TEXMFVAR/fonts/map/pdftex/updmap/pdftex.map" 2>/dev/null; then
  updmap-user --quiet >/dev/null 2>&1 || true
fi

SRC="${1:?usage: build.sh <srcdir> [defense|final|both]}"
MODE="${2:-defense}"
cd "$SRC" || exit 1
mkdir -p build build/chapters

run_defense() {
  pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1
  BIBINPUTS=.: TEXMFOUTPUT=build bibtex build/main >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1
  BIBINPUTS=.: TEXMFOUTPUT=build bibtex build/main >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode -output-directory=build main.tex >/dev/null 2>&1
}
run_final() {
  J="-jobname=main_final"; F="\def\FINALBUILD{}\input{main.tex}"
  pdflatex -interaction=nonstopmode -output-directory=build $J "$F" >/dev/null 2>&1
  BIBINPUTS=.: TEXMFOUTPUT=build bibtex build/main_final >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -output-directory=build $J "$F" >/dev/null 2>&1
  BIBINPUTS=.: TEXMFOUTPUT=build bibtex build/main_final >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -output-directory=build $J "$F" >/dev/null 2>&1
  pdflatex -interaction=nonstopmode -output-directory=build $J "$F" >/dev/null 2>&1
}

BUILT=""
case "$MODE" in
  defense) run_defense; BUILT="main";;
  final)   run_final;   BUILT="main_final";;
  both)    run_defense; run_final; BUILT="main main_final";;
  *) echo "unknown mode: $MODE"; exit 2;;
esac

# NOTE: we already cd'd into "$SRC", so the verifier must resolve paths from the CURRENT
# directory, not from "$SRC" again. Passing "$SRC" here looked for src/build/ INSIDE src/ and
# printed "BUILD FAILED" for a build that had in fact succeeded. Only invocations that happened
# to pass "." were unaffected, which is why this survived a whole session.
BUILT="$BUILT" python3 - "$PWD" <<'PY'
import os, re, sys
src = sys.argv[1]
built = os.environ.get("BUILT", "").split()
label = {"main": "DEFENSE", "main_final": "FINAL"}
rc = 0
for stem in built:
    lg = os.path.join(src, "build", stem + ".log")
    pdf = os.path.join(src, "build", stem + ".pdf")
    if not os.path.exists(pdf):
        print(f"{label[stem]}: BUILD FAILED — no PDF produced")
        for pat in (r"^! .*", r"Fatal error occurred.*", r"Font \S+ .*not found"):
            if os.path.exists(lg):
                for m in re.findall(pat, open(lg, encoding="utf8", errors="replace").read(), re.M)[:3]:
                    print(f"    {m.strip()[:110]}")
        rc = 1
        continue
    raw = open(lg, encoding="utf8", errors="replace").read()
    flat = raw.replace("\n", "")                      # LaTeX wraps warnings at 79 cols
    pages = re.findall(r"Output written on \S+ \((\d+) pages", raw)
    ofh = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", raw)
    ofv = re.findall(r"Overfull \\vbox \(([\d.]+)pt too high\)", raw)
    undc = sorted(set(re.findall(r"Citation `([^']+)' on page \d+ undefined", flat)))
    undr = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", flat)))
    blg = os.path.join(src, "build", stem + ".blg")
    bibe = []
    if os.path.exists(blg):
        b = open(blg, encoding="utf8", errors="replace").read()
        bibe = [l for l in b.split("\n") if re.search(r"error|didn't find|I was expecting", l, re.I)]
    low = "unmeasured"
    try:
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf)
        low = [i + 1 for i in range(len(doc))
               if len(re.findall(r"[A-Za-z]{2,}", doc[i].get_textpage().get_text_range())) < 120] or "none"
    except Exception as exc:
        low = f"unmeasured({exc.__class__.__name__})"
    print(f"{label[stem]}: pages={pages} overfull_hbox={len(ofh)} overfull_vbox={len(ofv)} "
          f"undef_cite={len(undc)} undef_ref={len(undr)} bibtex_problems={len(bibe)} "
          f"low_text_pages(<120 words, inspect)={low}")
    for u in undc: print(f"    UNDEFINED CITE: {u}")
    for u in undr: print(f"    UNDEFINED REF: {u}")
    for e in bibe[:5]: print(f"    BIBTEX: {e[:110]}")
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\)([^\n]*)", raw):
        print(f"    hbox {m.group(1)}pt:{m.group(2)[:80]}")
    if undc or undr or bibe or ofh or ofv:
        rc = 1
sys.exit(rc)
PY
