# Build dos slides de defesa. O motor TEM que ser xelatex ou lualatex -- nesped.sty carrega
# fontspec, que aborta sob pdflatex (ver PLANO_FLUXO_DEFESA.md §11.2).

ENGINE = xelatex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all clean showcase check

# Build completo: 2 passes + bibtex + 2 passes, para TOC/refs/bib resolverem.
all:
	$(ENGINE) $(FLAGS) main.tex
	-bibtex main
	$(ENGINE) $(FLAGS) main.tex
	$(ENGINE) $(FLAGS) main.tex

# Só recompila (sem bibtex) -- rápido, para iterar em conteúdo sem citação nova.
fast:
	$(ENGINE) $(FLAGS) main.tex

# Recompila o showcase original do template (referência de comandos/sintaxe).
showcase:
	$(ENGINE) $(FLAGS) -jobname=template_showcase template_showcase.tex
	-bibtex template_showcase
	$(ENGINE) $(FLAGS) -jobname=template_showcase template_showcase.tex
	$(ENGINE) $(FLAGS) -jobname=template_showcase template_showcase.tex

# Só checa que compila com 0 erros, sem deixar o PDF pela metade em caso de falha.
check:
	$(ENGINE) $(FLAGS) main.tex > /tmp/nesped_check.log 2>&1; \
	if grep -q '^!' /tmp/nesped_check.log; then \
		echo "FALHOU -- ver /tmp/nesped_check.log"; grep -n '^!' /tmp/nesped_check.log; exit 1; \
	else \
		echo "OK -- $$(grep 'Output written' /tmp/nesped_check.log)"; \
	fi

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz
