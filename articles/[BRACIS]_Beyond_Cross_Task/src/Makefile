# Build samplepaper.pdf into ../out/ from this directory.
# Usage: make            # full build (pdflatex + bibtex + 2x pdflatex)
#        make quick      # single pdflatex pass (no bib refresh)
#        make clean      # wipe ../out/

OUT = ../out
TEX = samplepaper
BIBINPUTS = ../src:
BSTINPUTS = ../src:

.PHONY: all quick clean

all: $(OUT)/$(TEX).pdf

$(OUT)/$(TEX).pdf: $(TEX).tex references.bib sections/*.tex tables/*.tex
	@mkdir -p $(OUT)
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(TEX).tex
	cd $(OUT) && BIBINPUTS=$(BIBINPUTS) BSTINPUTS=$(BSTINPUTS) bibtex $(TEX)
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(TEX).tex
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(TEX).tex
	@echo "Built $(OUT)/$(TEX).pdf"

quick:
	@mkdir -p $(OUT)
	pdflatex -interaction=nonstopmode -output-directory=$(OUT) $(TEX).tex

clean:
	rm -rf $(OUT)
