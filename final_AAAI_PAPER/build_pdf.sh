#!/usr/bin/env bash
# Compile final_AAAI_PAPER/main.tex -> main.pdf (pdflatex + bibtex, 4 passes).
# Usage (from anywhere):  bash final_AAAI_PAPER/build_pdf.sh
#
# Prepends the local MiKTeX bin to PATH if pdflatex isn't already there (the Bash
# tool's shell only sees it on PATH after Claude Code is restarted post-install).
MIKTEX="$HOME/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
if ! command -v pdflatex >/dev/null 2>&1 && [ -d "$MIKTEX" ]; then
  export PATH="$MIKTEX:$PATH"
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR" || exit 1

echo "[1/5] pdflatex ...";  pdflatex -interaction=nonstopmode main.tex >/tmp/blp1.log 2>&1
echo "[2/5] bibtex ...";    bibtex   main                              >/tmp/blp2.log 2>&1
echo "[3/5] pdflatex ...";  pdflatex -interaction=nonstopmode main.tex >/tmp/blp3.log 2>&1
echo "[4/5] pdflatex ...";  pdflatex -interaction=nonstopmode main.tex >/tmp/blp4.log 2>&1
echo "[5/5] pdflatex ...";  pdflatex -interaction=nonstopmode main.tex >/tmp/blp5.log 2>&1

if [ -f main.pdf ]; then
  grep -aE "Output written on main.pdf" /tmp/blp5.log
  over=$(grep -acE "Overfull \\\\hbox" /tmp/blp5.log)
  und=$(grep -acE "undefined" /tmp/blp5.log)
  echo "overfull-hboxes=$over  undefined-refs=$und"
  echo "OK -> $DIR/main.pdf"
else
  echo "BUILD FAILED -- see /tmp/blp1.log for the first error:"; tail -15 /tmp/blp1.log
  exit 1
fi
