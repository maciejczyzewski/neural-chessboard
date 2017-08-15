MAIN=main

pdf:
	latexmk -e '$$pdflatex=q/pdflatex -synctex=1 -shell-escape %O %S/' -pdf $(MAIN) -auxdir=output -outdir=output

all: pdf
	make -C code/programs

check:
	chktex -n1 -n2 -n8 -n11 -n12 -n13 -n14 -n26 -n34 -n33 -n36 -n40 $(MAIN)
	biber --tool -V bibliography.bib
	rm -f bibliography/biblio_bibertool.bib
	rm bibliography_bibertool.bib bibliography.bib.blg

lint:
	biber --tool bibliography.bib --output-fieldcase=lower --isbn-normalise
	mv bibliography_bibertool.bib bibliography.bib
	rm bibliography.bib.blg

arxiv: pdf
	mkdir -p arxiv
	cp main.tex figures/* output/*.pdf output/*.bbl arxiv
	rm arxiv/teax.py
	cd arxiv && latexmk -e '$$pdflatex=q/pdflatex -synctex=1 -shell-escape %O %S/' -pdf $(MAIN) -auxdir=output -outdir=output && rm -rf output
	cd arxiv && zip arxiv.zip *

clean:
	rm -rvf *.bbl *.blg *.aux *.fls *.fdb_latexmk *.log *.out *.toc $(MAIN).pdf aux output arxiv
