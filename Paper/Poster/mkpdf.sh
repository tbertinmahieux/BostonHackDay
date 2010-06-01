#/usr/bash
latex poster.tex
dvips -Ppdf poster.dvi -o poster.ps
ps2pdf poster.ps poster.pdf