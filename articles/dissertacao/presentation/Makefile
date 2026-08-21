# Build de todo o material de apresentação da defesa, a partir da raiz de presentation/.
#
# Cada projeto (nesped_slides_template/, slides/) compila para o seu próprio <projeto>/build/
# (gitignored -- aux/log/nav/etc ficam lá, nunca soltos ao lado das fontes) e o main.pdf
# resultante é copiado para a raiz do projeto (<projeto>/main.pdf), para ficar visível sem
# precisar entrar em build/.
#
# O motor TEM que ser xelatex ou lualatex -- nesped.sty carrega fontspec, que aborta sob
# pdflatex (ver PLANO_FLUXO_DEFESA.md §11.2).

ENGINE = xelatex
FLAGS  = -interaction=nonstopmode -halt-on-error

TEMPLATE_DIR = nesped_slides_template
SLIDES_DIR   = slides

.PHONY: all template slides check check-template check-slides clean clean-template clean-slides

all: template slides

# --- template (nesped_slides_template/) ---

template:
	mkdir -p $(TEMPLATE_DIR)/build
	cd $(TEMPLATE_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	-cd $(TEMPLATE_DIR) && bibtex build/main
	cd $(TEMPLATE_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	cd $(TEMPLATE_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	cp $(TEMPLATE_DIR)/build/main.pdf $(TEMPLATE_DIR)/main.pdf
	@echo "OK -- $(TEMPLATE_DIR)/main.pdf"

# só valida que compila com 0 erros, não deixa PDF pela metade em caso de falha.
check-template:
	mkdir -p $(TEMPLATE_DIR)/build
	@cd $(TEMPLATE_DIR) && $(ENGINE) -interaction=nonstopmode -output-directory=build main.tex > /tmp/nesped_check_template.log 2>&1; \
	if grep -q '^!' /tmp/nesped_check_template.log; then \
		echo "FALHOU -- ver /tmp/nesped_check_template.log"; grep -n '^!' /tmp/nesped_check_template.log; exit 1; \
	else \
		echo "OK -- $$(grep 'Output written' /tmp/nesped_check_template.log)"; \
	fi

clean-template:
	rm -rf $(TEMPLATE_DIR)/build $(TEMPLATE_DIR)/main.pdf

# --- slides (slides/) ---

slides:
	mkdir -p $(SLIDES_DIR)/build
	cd $(SLIDES_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	-cd $(SLIDES_DIR) && bibtex build/main
	cd $(SLIDES_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	cd $(SLIDES_DIR) && $(ENGINE) $(FLAGS) -output-directory=build main.tex
	cp $(SLIDES_DIR)/build/main.pdf $(SLIDES_DIR)/main.pdf
	@echo "OK -- $(SLIDES_DIR)/main.pdf"

check-slides:
	mkdir -p $(SLIDES_DIR)/build
	@cd $(SLIDES_DIR) && $(ENGINE) -interaction=nonstopmode -output-directory=build main.tex > /tmp/nesped_check_slides.log 2>&1; \
	if grep -q '^!' /tmp/nesped_check_slides.log; then \
		echo "FALHOU -- ver /tmp/nesped_check_slides.log"; grep -n '^!' /tmp/nesped_check_slides.log; exit 1; \
	else \
		echo "OK -- $$(grep 'Output written' /tmp/nesped_check_slides.log)"; \
	fi

check: check-template check-slides

clean: clean-template clean-slides
