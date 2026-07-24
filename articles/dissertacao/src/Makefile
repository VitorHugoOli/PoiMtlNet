# Dissertation build (single source main.tex, two modes; UFV_COMPLIANCE §1).
# All LaTeX output goes to build/ (gitignored); the current DEFENSE PDF is copied to
# ./dissertacao.pdf at the src root. Requires the usermode TeX tree on TEXMFHOME
# (see src_utils/README_SRC.md). Input paths (figures/, chapters/) resolve relative
# to this dir even with -output-directory, so no graphicspath change is needed.
TEXOPTS  = -interaction=nonstopmode -halt-on-error -output-directory=build
BIBINPUT = BIBINPUTS=.: TEXMFOUTPUT=build

all: defense

# DEFENSE build -> build/main.pdf, copied to ./dissertacao.pdf
defense:
	@mkdir -p build build/chapters
	pdflatex $(TEXOPTS) main.tex
	$(BIBINPUT) bibtex build/main || true
	pdflatex $(TEXOPTS) main.tex
	pdflatex $(TEXOPTS) main.tex
	@cp build/main.pdf dissertacao.pdf
	@echo "Defense PDF -> dissertacao.pdf (from build/main.pdf)"

# FINAL AcademicoPG build (body only) -> build/main_final.pdf. No second main file:
# the switch is set false on the command line before main.tex is read.
final:
	@mkdir -p build build/chapters
	pdflatex $(TEXOPTS) -jobname=main_final "\def\FINALBUILD{}\input{main.tex}"
	$(BIBINPUT) bibtex build/main_final || true
	pdflatex $(TEXOPTS) -jobname=main_final "\def\FINALBUILD{}\input{main.tex}"
	pdflatex $(TEXOPTS) -jobname=main_final "\def\FINALBUILD{}\input{main.tex}"
	@echo "Final AcademicoPG PDF -> build/main_final.pdf"

check:
	@src_utils/check.sh

clean:
	rm -rf build/*

.PHONY: all defense final check clean
