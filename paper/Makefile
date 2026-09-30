SHELL := /bin/sh
BUILD := build
REVIEW_PDF := $(BUILD)/main.pdf
PUBLIC_NAME := jspace-interpscience-2026
PUBLIC_PDF := $(BUILD)/$(PUBLIC_NAME).pdf
RELEASE_PDF := ../paper.pdf

.PHONY: all public review clean

all: public

public: $(RELEASE_PDF)

review: $(REVIEW_PDF)

$(REVIEW_PDF): main.tex references.bib neurips_2026.sty figures/si_scale.png figures/paired_scatter.png figures/displacement.png ../extension/analysis/scale_gate.png
	mkdir -p $(BUILD)
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -output-directory=$(BUILD) main.tex
	bibtex $(BUILD)/main
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -output-directory=$(BUILD) main.tex
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -output-directory=$(BUILD) main.tex

$(PUBLIC_PDF): public.tex main.tex references.bib neurips_2026.sty figures/si_scale.png figures/paired_scatter.png figures/displacement.png ../extension/analysis/scale_gate.png
	mkdir -p $(BUILD)
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -jobname=$(PUBLIC_NAME) -output-directory=$(BUILD) public.tex
	bibtex $(BUILD)/$(PUBLIC_NAME)
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -jobname=$(PUBLIC_NAME) -output-directory=$(BUILD) public.tex
	pdflatex -halt-on-error -file-line-error -interaction=nonstopmode -jobname=$(PUBLIC_NAME) -output-directory=$(BUILD) public.tex

$(RELEASE_PDF): $(PUBLIC_PDF)
	cp $(PUBLIC_PDF) $(RELEASE_PDF)

clean:
	rm -f $(BUILD)/main.aux $(BUILD)/main.bbl $(BUILD)/main.blg $(BUILD)/main.log
	rm -f $(BUILD)/main.out $(BUILD)/main.pdf
	rm -f $(BUILD)/$(PUBLIC_NAME).aux $(BUILD)/$(PUBLIC_NAME).bbl $(BUILD)/$(PUBLIC_NAME).blg
	rm -f $(BUILD)/$(PUBLIC_NAME).log $(BUILD)/$(PUBLIC_NAME).out $(BUILD)/$(PUBLIC_NAME).pdf
