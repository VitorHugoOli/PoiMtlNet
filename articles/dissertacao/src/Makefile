# Dissertation build. THREE targets, ONE source (UFV_COMPLIANCE §1):
#   defense   -> build/main.pdf           the reading copy, copied to ./dissertacao.pdf
#   academico -> build/main_academico.pdf the AcademicoPG deposit body (was `final`
#                                         until 2026-07-29; `make final` still works
#                                         and forwards here)
#   ppgc      -> build/main_ppgc.pdf      the defense document plus the approval sheet
# All LaTeX output goes to build/ (gitignored); the current DEFENSE PDF is copied to
# ./dissertacao.pdf at the src root. Requires the usermode TeX tree on TEXMFHOME
# (see src_utils/README_SRC.md). Input paths (figures/, chapters/) resolve relative
# to this dir even with -output-directory, so no graphicspath change is needed.
#
# `make help` lists every target. The build map, and why it is shaped this way:
#
#   PLAIN, ALWAYS AVAILABLE      defense / academico / ppgc  three passes each
#   ALL THREE AT ONCE            all3                        the same builds concurrently
#   PREAMBLE PRECOMPILED         fast / fast3                loads a format dump
#   GATES                        check (the one CI gate) + one target per checker
#
# EVERY TIMING FOR THESE TARGETS LIVES IN ONE PLACE: src_utils/_round7/20_build_speed.md,
# with the command that produced it. None are repeated here. That is deliberate -- a duration
# quoted in a Makefile comment is a measurement nobody re-runs, and this file had one written
# into it before the corresponding build had ever been executed.
#
# ---------------------------------------------------------------------------------------
# WHY EACH TARGET HAS ITS OWN AUX TREE (round 7, and it is not a preference).
# Every target used to write into build/ with the SAME build/chapters/*.aux files, so two
# targets could not be built at the same time: whichever pass was mid-write left the other
# reading a truncated aux. That is not a hypothetical -- it was measured on 2026-07-29 while
# another session was building this tree, and the victim build died with
#     Runaway argument? {\contentsline {subsection}{\numberline {4.3.1}Baseline: MTLnet ...
#     ! File ended while scanning use of \@writefile.
# which reads exactly like a LaTeX defect in chapter 4 and is nothing of the kind. Each
# target now writes its \include aux tree into build/<stem>-aux/, so the three cannot
# interfere, and `make all3` runs them concurrently.
#
# The RESULT files are copied back into build/ afterwards, so every documented path stays
# true: build/main.pdf, build/main.log and build/main.blg are what src_utils/check.sh,
# src_utils/sync_page_counts.py, src_utils/build.sh, CLAUDE.md and README_SRC.md all name.
# Only those three per target are copied. The .aux/.toc/.lof/.lot deliberately are NOT:
# a copy of an aux file sitting in a directory BibTeX also searches is how four undefined
# citations shipped in both PDFs (science/AGENT_HANDOFF.md §2.4), and the authoritative
# copies live in build/<stem>-aux/.
# ---------------------------------------------------------------------------------------
TEXFLAGS = -interaction=nonstopmode -halt-on-error
UTILS    = ../src_utils

all: defense

# ---- preflight: the TeX environment this Makefile documents as mandatory ----------------
# WHY. Every build target below needs `source ../src_utils/texenv.sh` first, this file has
# said so since it was written, and nothing checked it. Skipping it costs an hour, because the
# failure names the wrong thing: pdflatex dies with
#   !pdfTeX error: pdflatex (file t1xtt): Font t1xtt at 657 not found
# (or ntx-Regular-tlf-ot1r, depending on which face it reaches first) after three passes of
# apparently normal output. That is NOT a missing font -- the .tfm and .pfb are both in
# $TEXMFHOME -- it is the font MAP, which newtx writes into the USERMODE updmap output under
# TEXMFVAR. It cost an hour at the start of round 6 and is finding F-3 of LATEX_UPGRADE.md.
#
# WHAT IT TESTS, AND WHY NOT `test -n "$$TEXMFHOME"`. The naive guard (is the variable set?)
# would be a gate firing on the wrong condition. MEASURED on this machine, 2026-07-29:
# `kpsewhich -var-value TEXMFHOME` returns $HOME/Library/texmf whether or not the variable is
# exported, and abntex2.cls resolves either way -- so an unset TEXMFHOME is not the failure.
# What breaks is TEXMFVAR, whose DEFAULT is $HOME/Library/texlive/2026basic/texmf-var, a tree
# holding no ntx map entries at all. So the preflight tests the condition that actually
# breaks the build: pdflatex on PATH, and an ntx entry in the map under whatever TEXMFVAR
# resolves to. Measured both directions: with texenv.sh sourced the map has 168 ntx- lines;
# without it, the default tree has no pdftex.map at all.
preflight:
	@command -v pdflatex >/dev/null 2>&1 || { \
	  echo "BUILD ENVIRONMENT NOT SET UP: pdflatex is not on PATH."; \
	  echo "  run:  source ../src_utils/texenv.sh   (from articles/dissertacao: source src_utils/texenv.sh)"; \
	  exit 1; }
	@MAP="$$(kpsewhich -var-value TEXMFVAR)/fonts/map/pdftex/updmap/pdftex.map"; \
	grep -q '^ntx-' "$$MAP" 2>/dev/null || { \
	  echo "BUILD ENVIRONMENT NOT SET UP: the newtx font MAP is not on TEXMFVAR."; \
	  echo "  TEXMFVAR resolves to: $$(kpsewhich -var-value TEXMFVAR)"; \
	  echo "  expected ntx- entries in: $$MAP"; \
	  echo "  run:  source ../src_utils/texenv.sh   (from articles/dissertacao: source src_utils/texenv.sh)"; \
	  echo "  Without it this build runs three passes and then dies with"; \
	  echo "  'Font ntx-Regular-tlf-ot1r at 657 not found', which is a missing font MAP and"; \
	  echo "  not a missing font. Do not go looking for the font."; \
	  exit 1; }

# DEFENSE build -> build/main.pdf, copied to ./dissertacao.pdf
defense: preflight
	@mkdir -p build build/main-aux/chapters
	pdflatex $(TEXFLAGS) -output-directory=build/main-aux main.tex
	BIBINPUTS=.: TEXMFOUTPUT=build/main-aux bibtex build/main-aux/main; test $$? -le 1
	pdflatex $(TEXFLAGS) -output-directory=build/main-aux main.tex
	pdflatex $(TEXFLAGS) -output-directory=build/main-aux main.tex
	@cp build/main-aux/main.pdf build/main-aux/main.log build/main-aux/main.blg build/
	@cp build/main.pdf dissertacao.pdf
	@echo "Defense PDF -> dissertacao.pdf (from build/main.pdf)"

# ACADEMICO build: the AcademicoPG deposit body (no front matter) -> build/main_academico.pdf.
# Renamed from `final` on 2026-07-29 (author decision, src_utils/LATEX_UPGRADE.md §4 A-1/A-2/A-3):
# "final" said nothing next to `defense` and `ppgc`, and this build has exactly one purpose --
# the body uploaded to AcademicoPG.
#
# It also stopped being a command-line injection. It used to be
#   pdflatex ... -jobname=main_final "\def\FINALBUILD{}\input{main.tex}"
# with no entry file of its own, which made it the ONE build nobody could produce from Overleaf
# or a GUI editor: an editor can select a compile root, but cannot be told to \def a macro
# first. main_academico.tex is that root now, and this recipe is the same three-pass shape as
# `defense` and `ppgc` with no injection left anywhere.
academico: preflight
	@mkdir -p build build/main_academico-aux/chapters
	pdflatex $(TEXFLAGS) -output-directory=build/main_academico-aux main_academico.tex
	BIBINPUTS=.: TEXMFOUTPUT=build/main_academico-aux bibtex build/main_academico-aux/main_academico; test $$? -le 1
	pdflatex $(TEXFLAGS) -output-directory=build/main_academico-aux main_academico.tex
	pdflatex $(TEXFLAGS) -output-directory=build/main_academico-aux main_academico.tex
	@cp build/main_academico-aux/main_academico.pdf build/main_academico-aux/main_academico.log build/main_academico-aux/main_academico.blg build/
	@echo "AcademicoPG deposit PDF -> build/main_academico.pdf"

# The old target name, kept as an alias so a half-remembered `make final` does not silently
# do nothing. It prints what replaced it and builds the right thing rather than failing.
final:
	@echo "make final was renamed to make academico on 2026-07-29 (build/main_academico.pdf);"
	@echo "building that instead. LATEX_UPGRADE.md §4 A-1."
	@$(MAKE) --no-print-directory academico

# PPGC build: the defense document PLUS the approval sheet -> build/main_ppgc.pdf.
# main_ppgc.tex is a two-line file that sets \APPROVALSHEET and reads main.tex, so this
# is the same three-pass recipe with a different entry file and jobname.
ppgc: preflight
	@mkdir -p build build/main_ppgc-aux/chapters
	pdflatex $(TEXFLAGS) -output-directory=build/main_ppgc-aux main_ppgc.tex
	BIBINPUTS=.: TEXMFOUTPUT=build/main_ppgc-aux bibtex build/main_ppgc-aux/main_ppgc; test $$? -le 1
	pdflatex $(TEXFLAGS) -output-directory=build/main_ppgc-aux main_ppgc.tex
	pdflatex $(TEXFLAGS) -output-directory=build/main_ppgc-aux main_ppgc.tex
	@cp build/main_ppgc-aux/main_ppgc.pdf build/main_ppgc-aux/main_ppgc.log build/main_ppgc-aux/main_ppgc.blg build/
	@echo "PPGC PDF (defense + approval sheet) -> build/main_ppgc.pdf"

# ---- all three at once -----------------------------------------------------------------
# Timings for this target are in src_utils/_round7/20_build_speed.md, measured on this
# machine; do not restate them here, because a number in two places drifts in one of them.
# `-j3` is spelled inside the recipe rather than left to the caller, because a caller who
# forgets it gets the serial time and no warning. The sub-make cannot deadlock: the three
# targets share no file, which is the point of the per-target aux trees above.
all3:
	@$(MAKE) --no-print-directory -j3 defense academico ppgc

# ---- format-accelerated builds (an ACCELERATOR, never a requirement) -------------------
# The preamble (abntex2 + memoir + newtxmath + hyperref + abntex2cite) is most of the cost of
# every pdflatex pass, because the document body is only a few seconds of it. `fast`
# precompiles that preamble into a pdflatex format dump and loads the dump instead of
# re-reading it three times. The measured before/after, per target, is in
# src_utils/_round7/20_build_speed.md.
# mkformat.py rebuilds the dump automatically when the preamble's checksum moves, and
# fastbuild.sh REFUSES to run on a stale key, so a stale-format build is not reachable. The
# plain targets above never touch the format and keep working with none present -- that
# matters for Overleaf, which cannot load a local dump.
fast: fast-defense

fast-defense: preflight
	@$(UTILS)/mkformat.py --build
	@$(UTILS)/fastbuild.sh defense
	@cp build/main.pdf dissertacao.pdf
	@echo "Defense PDF -> dissertacao.pdf (from build/main.pdf, format-accelerated)"

fast-academico: preflight
	@$(UTILS)/mkformat.py --build
	@$(UTILS)/fastbuild.sh academico

fast-ppgc: preflight
	@$(UTILS)/mkformat.py --build
	@$(UTILS)/fastbuild.sh ppgc

# All three, accelerated and concurrent. The dump is built ONCE first (serialized on
# purpose: three concurrent -ini runs would race on the same mainpre.fmt).
fast3: preflight
	@$(UTILS)/mkformat.py --build
	@$(MAKE) --no-print-directory -j3 fast3-defense fast3-academico fast3-ppgc
	@cp build/main.pdf dissertacao.pdf
fast3-defense:   ; @$(UTILS)/fastbuild.sh defense
fast3-academico: ; @$(UTILS)/fastbuild.sh academico
fast3-ppgc:      ; @$(UTILS)/fastbuild.sh ppgc

format: preflight
	@$(UTILS)/mkformat.py --build
format-status:
	@$(UTILS)/mkformat.py --status

# The whole proof, from nothing: build all three the plain way, store them as the reference,
# rebuild all three with the format, and compare the text layer of every one. Takes several
# minutes on purpose -- this is the target that decides whether `fast` may be trusted, so it
# does not reuse anything it could have got wrong. Run it after any preamble change you want
# to be sure about, though the staleness guard already forces a re-dump for those.
verify-equiv:
	@$(MAKE) --no-print-directory all3
	@python3 $(UTILS)/verify_format.py --reference --note "make all3 (plain, no format)"
	@$(MAKE) --no-print-directory fast3
	@python3 $(UTILS)/verify_format.py --compare --all

# ---- gates ------------------------------------------------------------------------------
# `check` stays the ONE gate a CI-style caller runs: it is the whole suite and it exits
# nonzero on any finding. The per-checker targets below exist so nobody has to know the
# src_utils/ layout to run one gate while fixing what it found. `check-fast` is every
# checker without the build-log-dependent gates, for a quick loop while editing prose.
check:
	@$(UTILS)/check.sh

check-trapped:   ; @python3 $(UTILS)/test_trapped_prose.py && python3 $(UTILS)/check_trapped_prose.py
check-torn:      ; @python3 $(UTILS)/check_torn_sentences.py
check-doubled:   ; @python3 $(UTILS)/check_doubled_macro.py
check-texroot:   ; @python3 $(UTILS)/check_tex_root.py
check-parallelism: ; @python3 $(UTILS)/check_negative_parallelism.py
check-meta:      ; @python3 $(UTILS)/check_meta_claims.py
check-verify:    ; @python3 $(UTILS)/check_verify_list.py
check-wordcount: ; @python3 $(UTILS)/check_wordcount_claims.py
check-pages:     ; @python3 $(UTILS)/sync_page_counts.py
check-sweepguard: ; @python3 $(UTILS)/sweep_guard.py
check-format:    ; @python3 $(UTILS)/mkformat.py --selftest
check-equiv:     ; @python3 $(UTILS)/verify_format.py --compare

# Every checker that does not need a compiled build/*.log, run in one go. Reports each
# gate's exit status and fails if any failed; it does NOT stop at the first failure, so one
# run tells you everything that is broken.
check-scripts:
	@$(UTILS)/check_scripts.sh

sync-pages:
	@python3 $(UTILS)/sync_page_counts.py --write

help:
	@echo "BUILD   make defense | academico | ppgc  one target, three passes"
	@echo "        (make final is an alias of academico, renamed 2026-07-29)"
	@echo "        make all3                        all three concurrently"
	@echo "        make fast                        defense with the precompiled preamble"
	@echo "        make fast3                       all three, accelerated and concurrent"
	@echo "        (timings: src_utils/_round7/20_build_speed.md, measured on this machine)"
	@echo "        make format | format-status      build / inspect the preamble format dump"
	@echo "GATES   make check                       THE gate: the full suite, exit 0 or bust"
	@echo "        make check-scripts               every script gate, one line each, no build needed"
	@echo "        make check-trapped check-torn check-doubled check-texroot check-parallelism"
	@echo "        make check-meta check-verify check-wordcount check-pages check-sweepguard"
	@echo "        make check-format check-equiv    format split self-test / PDF equivalence"
	@echo "        make verify-equiv                the full proof: plain build vs accelerated"
	@echo "        make sync-pages                  rewrite the recorded page counts from the build"
	@echo "OTHER   make clean                       remove build/ output"

clean:
	rm -rf build/*

.PHONY: all preflight defense academico final ppgc all3 fast fast-defense fast-academico fast-ppgc \
        fast3 fast3-defense fast3-academico fast3-ppgc format format-status verify-equiv \
        check check-trapped check-torn check-doubled check-texroot check-parallelism \
        check-meta check-verify check-wordcount check-pages check-sweepguard \
        check-format check-equiv check-scripts sync-pages help clean
