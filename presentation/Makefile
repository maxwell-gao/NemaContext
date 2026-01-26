# Makefile for NemaContext Presentation

# Main target
MAIN = nemacontext_progress

# SVG figures
SVGS = $(wildcard figures/*.svg)
PDFS = $(SVGS:.svg=.pdf)

# Default target
all: $(MAIN).pdf

# Compile with shell-escape for SVG support
$(MAIN).pdf: $(MAIN).tex $(PDFS)
	pdflatex --shell-escape $(MAIN).tex
	pdflatex --shell-escape $(MAIN).tex

# Convert SVG to PDF using Inkscape
figures/%.pdf: figures/%.svg
	inkscape --export-type=pdf --export-filename=$@ $<

# Alternative: compile without SVG (uses placeholder boxes)
quick:
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

# Convert all SVGs to PDF
svgs: $(PDFS)

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

# Clean all generated files
cleanall: clean
	rm -f $(MAIN).pdf figures/*.pdf

# View the PDF
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || evince $(MAIN).pdf

# Help
help:
	@echo "NemaContext Presentation Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build presentation with SVG figures (default)"
	@echo "  quick    - Build without SVG conversion (uses placeholders)"
	@echo "  svgs     - Convert all SVG figures to PDF"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files"
	@echo "  view     - Open the PDF"
	@echo ""
	@echo "Requirements:"
	@echo "  - pdflatex with beamer"
	@echo "  - inkscape (for SVG conversion)"
	@echo ""
	@echo "Notes:"
	@echo "  - Create SVG figures in figures/ folder"
	@echo "  - See figures/README.md for figure specifications"

.PHONY: all quick svgs clean cleanall view help
