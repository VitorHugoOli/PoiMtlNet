# Dissertation build. Requires: TeX Live with the repo-local usermode tree on TEXMFHOME
# (see README_SRC.md). Two build modes per UFV_COMPLIANCE §1.
TEXOPTS = -interaction=nonstopmode

all: defense final

defense:
	pdflatex $(TEXOPTS) main_defense.tex
	bibtex main_defense || true
	pdflatex $(TEXOPTS) main_defense.tex
	pdflatex $(TEXOPTS) main_defense.tex

final:
	pdflatex $(TEXOPTS) main_final.tex
	bibtex main_final || true
	pdflatex $(TEXOPTS) main_final.tex
	pdflatex $(TEXOPTS) main_final.tex

check:
	./check.sh

clean:
	rm -f *.aux *.log *.lof *.lot *.toc *.out *.bbl *.blg *.idx chapters/*.aux

.PHONY: all defense final check clean
