@echo off
REM ============================================================
REM  Compile LaTeX manuscript (IEEE two-column format)
REM  Requires: pdflatex + bibtex (e.g., TeX Live or MiKTeX)
REM ============================================================

echo [1/4] pdflatex (first pass)...
pdflatex -interaction=nonstopmode main.tex

echo [2/4] bibtex...
bibtex main

echo [3/4] pdflatex (second pass)...
pdflatex -interaction=nonstopmode main.tex

echo [4/4] pdflatex (final pass)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo Done! Output: main.pdf
echo.
pause
