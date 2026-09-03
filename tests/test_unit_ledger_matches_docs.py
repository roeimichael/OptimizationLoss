"""The docs may not claim more independent units than the LEDGER licenses.

🛑 WHY THIS EXISTS (2026-09-03). `scripts/paper_rows.MEASURED_UNITS` is the only
place a unit is a fact. Everything else -- FRAMEWORK, COVERAGE, MISSION, THEORY
-- quotes a COUNT and a sign-test p derived from it, and the two drifted apart
in the worst possible direction:

  * the ledger ships FOUR entries (A1, A2, B1, C1);
  * its own in-dict comment still references an "A1/A2/B1/B2" and calls C1
    "genuinely the fifth";
  * `docs/COVERAGE.md` claims 5/5 and p=0.031, which CLEARS p<0.05;
  * `docs/FRAMEWORK.md` 2(z26) says 4/4 and p=0.0625, which does NOT.

B2 was `("loose1", "RegNetY400MF")`, removed in commit 1a7723a0 because loose1
ran `constraint_grad_mode: clip` and is therefore a different method. That
reason is sound. What is NOT sound is that the removed unit was the DISSENTING
one and the commit is titled "the result gets BETTER" -- a sign test is valid
only under an inclusion rule fixed BEFORE the signs are seen. Adding or removing
units after reading their direction is a forking path, and this project has now
done it twice, each time moving the headline toward significance.

So: the ledger is the bound, and no document may quote a stronger one. A gate,
not a note, because the drift already produced two published numbers that
disagree about whether the headline clears 0.05.

The negative control is in the same file: a doc string claiming more units than
the ledger holds MUST fail, or this test has never been shown to work.
"""
import ast
import io
import math
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ["docs/FRAMEWORK.md", "docs/COVERAGE.md", "docs/MISSION.md",
        "docs/THEORY.md", "CLAUDE.md"]

# `n/n units` or `n of n units`, and a p-value on the same line.
CLAIM = re.compile(r"(\d+)\s*(?:/|of)\s*(\d+)\s+units", re.I)
PVAL = re.compile(r"p\s*[=<]\s*0?\.(\d+)")
# A markdown TABLE row is how COVERAGE stated 5/5 with the word `units` only in
# the column HEADER, which the line-local pattern above cannot see. Any row
# carrying `n/m` beside a bare decimal is a units-and-sign-p row.
ROW = re.compile(r"^\s*\|.*?\b(\d+)\s*/\s*(\d+)\b.*?\|\s*\**(0?\.\d+)\**\s*\|")


def ledger_size():
    """len(MEASURED_UNITS), read by AST so importing scripts/ is not required."""
    src = io.open(os.path.join(ROOT, "scripts", "paper_rows.py"),
                  encoding="utf-8").read()
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Assign) and node.targets
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MEASURED_UNITS"):
            return len(node.value.keys)
    raise AssertionError("MEASURED_UNITS not found in scripts/paper_rows.py")


def _read(rel):
    p = os.path.join(ROOT, rel)
    return io.open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def scan(text, n_units):
    """Lines claiming more units than the ledger holds, or an unreachable p.

    Two shapes are checked, and only two:
      * prose  -- `n/n units`, `n of n units`, on one line;
      * a table ROW, but ONLY inside a table whose HEADER declares a `units`
        column. COVERAGE stated 5/5 with the word `units` only in the header,
        which a line-local pattern cannot see; matching every `n/m` beside a
        decimal instead caught dose ratios like `44/44 steps`, and a gate that
        fires on healthy text is a gate somebody loosens.
    """
    floor = 0.5 ** n_units
    bad = []
    in_units_table = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("|"):
            cells = [c.strip().strip("*").lower() for c in stripped.strip("|").split("|")]
            if any(c == "units" for c in cells):
                in_units_table = True          # this is the header row
                continue
        elif in_units_table:
            in_units_table = False             # a blank/prose line ends the table

        for m in CLAIM.finditer(line):
            got, tot = int(m.group(1)), int(m.group(2))
            if tot > n_units:
                bad.append((i, "claims %d units; the ledger licenses %d"
                            % (tot, n_units), stripped[:110]))
            elif got == tot:
                pm = PVAL.search(line)
                if pm:
                    p = float("0." + pm.group(1))
                    if p < 0.5 ** got - 1e-12:
                        bad.append((i, "quotes p=%.4g below the 0.5^%d = %.4g "
                                    "floor for its own n" % (p, got, 0.5 ** got),
                                    stripped[:110]))

        if in_units_table:
            # A row that says `cells` in its own count field is a per-cell
            # tally sitting in a units-headed table. That is the exact
            # ambiguity that let 4 become 5, but the row itself is honest --
            # flag the header, not the row.
            rm = None if re.search(r"\d+\s*/\s*\d+\s*cells", line) else ROW.match(line)
            if rm:
                got, tot, p = int(rm.group(1)), int(rm.group(2)), float(rm.group(3))
                if tot > n_units:
                    bad.append((i, "table row claims %d units; the ledger "
                                "licenses %d" % (tot, n_units), stripped[:110]))
                elif got == tot and p < 0.5 ** got - 1e-12:
                    bad.append((i, "table row quotes p=%.4g below the 0.5^%d = "
                                "%.4g floor" % (p, got, 0.5 ** got), stripped[:110]))
    return bad, floor


def test_no_document_claims_more_units_than_the_ledger_licenses():
    n = ledger_size()
    assert n >= 1, "the unit ledger is empty"
    problems = []
    for rel in DOCS:
        bad, _ = scan(_read(rel), n)
        problems += ["%s:%d  %s\n      %s" % (rel, ln, why, txt)
                     for ln, why, txt in bad]
    assert not problems, (
        "scripts/paper_rows.MEASURED_UNITS holds %d entries, so the strongest "
        "one-sided sign test available is %d/%d, p=%.4g. These lines claim "
        "more:\n  %s\n\n"
        "Fix by EITHER adding the missing unit to the ledger WITH its md5 "
        "evidence, OR correcting the document. Do not 'fix' it by loosening "
        "this test: the ledger's own doctrine is that an absent entry is "
        "UNVERIFIED, not independent, and the default must not be the "
        "flattering one."
        % (n, n, n, 0.5 ** n, "\n  ".join(problems)))


def test_NEGATIVE_CONTROL_the_scanner_catches_an_inflated_claim():
    """A gate that has never failed has never been shown to work."""
    n = ledger_size()
    inflated = "tralo beats clip in %d/%d units (p=0.031) on the corpus" % (
        n + 1, n + 1)
    bad, _ = scan(inflated, n)
    assert bad, ("the scanner did not flag a claim of %d units against a "
                 "ledger of %d" % (n + 1, n))

    unreachable = "tralo beats clip in %d/%d units, p=0.001" % (n, n)
    bad2, _ = scan(unreachable, n)
    assert bad2, ("the scanner did not flag p=0.001, which is below the "
                  "0.5^%d = %.4g floor for a %d-unit sign test"
                  % (n, 0.5 ** n, n))

    ok = "tralo beats clip in %d/%d units (p=%.4g)" % (n, n, 0.5 ** n)
    bad3, _ = scan(ok, n)
    assert not bad3, ("the scanner flagged a CORRECT claim (%s), so it would "
                      "fire on honest text" % ok)

    # the TABLE path, which is how COVERAGE stated it
    tbl = ("| contrast | units | sign p | |\n|---|---|---|---|\n"
           "| `tralo` vs `clip` | **%d/%d** | **0.031** | ok |\n" % (n + 1, n + 1))
    bad4, _ = scan(tbl, n)
    assert bad4, "the scanner missed an inflated claim inside a units TABLE"

    # and it must NOT fire on a dose ratio in a table with no units column
    dose = ("| arm | dose | share | |\n|---|---|---|---|\n"
            "| tralo | 44/44 | 0.031 | landed |\n")
    bad5, _ = scan(dose, n)
    assert not bad5, ("the scanner fired on a DOSE ratio (44/44) in a table "
                      "with no units column: %s" % bad5)


@pytest.mark.parametrize("n,expected", [(4, 0.0625), (5, 0.03125), (8, 0.00390625)])
def test_the_sign_test_floor_is_what_the_docs_say_it_is(n, expected):
    """0.5^n, quoted all over the repo. Cheap, and it has been got wrong."""
    assert math.isclose(0.5 ** n, expected, rel_tol=1e-12)
