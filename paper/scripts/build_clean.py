"""Build the clean manuscript: revisions accepted, marks stripped, tables too.

The revision apparatus lives in docs/main.tex and paper/tables_rev/. A reviewer
must never see it. This drives the whole conversion in one reproducible step,
because doing it by hand is how three captions reached a reviewer structurally
broken in round 1.

  docs/main.tex          -> paper/main_clean.tex
  paper/tables_rev/*.tex -> paper/tables_clean/*.tex
  \\input{tables_rev/X}   -> \\input{tables_clean/X}

Then VERIFIES the output rather than trusting it: every \\caption must still open
with a brace, no revision marker may survive, and no \\pending may remain.

Run:  python paper/scripts/build_clean.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
STRIP = os.path.join(HERE, "strip_revision_marks.py")

SRC = os.path.join(ROOT, "docs", "main.tex")
OUT = os.path.join(PAPER, "main_clean.tex")
REV = os.path.join(PAPER, "tables_rev")
CLEAN = os.path.join(PAPER, "tables_clean")


def strip(src, dst):
    r = subprocess.run([sys.executable, STRIP, src, "-o", dst],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stdout.write(r.stderr)
    return r.returncode


def check(path):
    """Structural checks on a stripped file. Returns a list of problems."""
    raw = open(path, encoding="utf-8").read()
    # Blank out comments but keep the line count, so reported lines stay usable.
    # Without this a comment mentioning "\caption:" reads as a broken caption.
    s = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in raw.split("\n"))
    bad = []
    # The round-1 defect: "\caption\textbf{...}" compiles fine and typesets the
    # caption as body text. Every \caption must be followed by a brace.
    for m in re.finditer(r"\\caption(?:of)?\s*(.)", s):
        if m.group(1) not in "{[":
            line = s[:m.start()].count("\n") + 1
            bad.append("%s:%d: \\caption not followed by '{'" % (os.path.basename(path), line))
    for marker in (r"\del{", r"\rev{", r"\color{blue}", r"\pending{"):
        n = s.count(marker)
        if n:
            bad.append("%s: %d surviving %s" % (os.path.basename(path), n, marker))
    return bad


def main():
    os.makedirs(CLEAN, exist_ok=True)
    rc = strip(SRC, OUT)

    names = sorted(f for f in os.listdir(REV) if f.endswith(".tex"))
    for f in names:
        strip(os.path.join(REV, f), os.path.join(CLEAN, f))
    print("stripped %d revision-marked tables: %s" % (len(names), ", ".join(names)))

    s = open(OUT, encoding="utf-8").read()
    s, n = re.subn(r"\\input\{tables_rev/", r"\\input{tables_clean/", s)
    open(OUT, "w", encoding="utf-8").write(s)
    print("repointed %d table inputs to tables_clean/" % n)

    problems = check(OUT)
    for f in names:
        problems += check(os.path.join(CLEAN, f))
    if problems:
        print("\nSTRUCTURAL PROBLEMS -- do not ship this build:")
        for p in problems:
            print("  " + p)
        return 1
    print("\nstructural checks passed: captions intact, no surviving markers")
    return rc


if __name__ == "__main__":
    sys.exit(main())
