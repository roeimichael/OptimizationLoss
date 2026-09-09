"""Which documented FIGURES were produced by a scorer that has since changed?

🛑 THE DEFECT THIS EXISTS FOR (found 2026-09-10, FRAMEWORK 2(z68)). The
acceptance figure `6 of 17 = 35%` was written into `CLAUDE.md` at 09:09 on
2026-09-06. `deployed_h2h.rank_cell` -- which produces the per-arm deltas that
BOTH halves of that verdict read -- got the common-seeds fix at 22:48 the same
day. Nothing connected the two: the figure lives in a doc, the fix lives in
git, and no gate holds one against the other. The number stood for four days
as the answer to the project's own acceptance bar.

It is the same shape as the `add_seeds` extensions that were bought and never
pooled, and as `deep_scope`'s `+0.504 over 360 runs`. The signature is always
the same -- a number that was true when written, is quoted as though it were
true now, and nothing is red.

⚠️ THIS IS A REPORT, NOT A GATE, AND THE DISTINCTION IS LOAD-BEARING. A
scorer's last commit may have touched only a docstring, in which case the
figure is fine. Classifying that automatically means guessing at whether a
diff changed a number, and this project has paid for that kind of guess more
than once. So the tool prints the commit SUBJECT beside every hit and leaves
the judgement to a person. A hit means UNVERIFIED, never WRONG.

⚠️ AND IT UNDER-REPORTS BY CONSTRUCTION. A figure is only checkable if it
carries a date AND a script can be attributed to it. Measured on this repo:
51 date-stamped figures, 16 attributable within 1500 characters, 35 not.
The
35 are printed as a count so the coverage is never mistaken for completeness
-- a tool that silently examines a third of its input reads exactly like one
that found nothing wrong.
"""
import io
import os
import re
import subprocess
import sys

DOCS = ("CLAUDE.md", "docs/FRAMEWORK.md", "docs/COVERAGE.md",
        "docs/PLAYBOOK.md", "docs/MISSION.md")

# The verbs this project actually stamps a measurement with. Kept explicit
# rather than "any date", because a date in prose ("removed 2026-09-02") is
# not a figure and flagging it would bury the ones that are.
FIGURE = re.compile(
    r"(?:RUN|MEASURED|Measured|Run on|RECOUNT DONE|RECOMPUTED|Recomputed"
    r"|Verified|VERIFIED|Audited)[^.\n]{0,60}?(20\d\d-\d\d-\d\d)")

# (?<![\w/]) so `docs/paper/scripts/make_main_table.py` is NOT read as
# `scripts/make_main_table` and reported absent. That fired on the very
# first real run, which is the cheapest possible way to find it.
SCRIPT = re.compile(r"(?<![\w/])scripts[./]([a-z_][a-z_0-9]*)")

# How far back to look for the script a figure belongs to. Wide enough to
# span a CLAUDE.md command block, narrow enough that a figure in one
# FRAMEWORK entry does not attach to a script named in the previous one.
NEAR = 1500

_CACHE = {}


def last_commit(mod):
    """(date, subject) of the last commit touching `scripts/<mod>.py`."""
    if mod in _CACHE:
        return _CACHE[mod]
    path = os.path.join("scripts", mod + ".py")
    out = None
    if os.path.isfile(path):
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ad|%s", "--date=short", "--", path],
            capture_output=True, text=True)
        line = (r.stdout or "").strip()
        if line and "|" in line:
            date, subj = line.split("|", 1)
            out = (date.strip(), subj.strip())
    _CACHE[mod] = out
    return out


def figures(text, near=NEAR):
    """Every (line, figure_date, module) this doc stamps. module may be None."""
    found = []
    for m in FIGURE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        window = text[max(0, m.start() - near):m.start()]
        names = SCRIPT.findall(window)
        found.append((line, m.group(1), names[-1] if names else None))
    return found


def scan(docs=DOCS, near=NEAR):
    """(stale, fresh, unattributable, absent) over `docs`."""
    stale, fresh, unattr, absent = [], 0, 0, []
    for doc in docs:
        if not os.path.isfile(doc):
            continue
        text = io.open(doc, encoding="utf-8").read()
        for line, fig, mod in figures(text, near):
            if mod is None:
                unattr += 1
                continue
            lc = last_commit(mod)
            if lc is None:
                absent.append((doc, line, mod, fig))
                continue
            date, subj = lc
            if date > fig:
                stale.append((doc, line, mod, fig, date, subj))
            else:
                fresh += 1
    return stale, fresh, unattr, absent


def report(docs=DOCS, near=NEAR, out=sys.stdout):
    stale, fresh, unattr, absent = scan(docs, near)
    out.write("FIGURES WHOSE SCORER HAS CHANGED SINCE (UNVERIFIED, not wrong)\n")
    out.write("=" * 74 + "\n")
    if not stale:
        out.write("  none -- every attributable figure predates no change to "
                  "its scorer.\n")
    for doc, line, mod, fig, date, subj in sorted(stale, key=lambda r: r[3]):
        out.write("  %s:%d\n" % (doc, line))
        out.write("      figure %s   %-22s scorer moved %s\n"
                  % (fig, mod, date))
        out.write("      last commit: %s\n" % subj[:70])
    for doc, line, mod, fig in absent:
        out.write("  %s:%d  figure %s cites scripts/%s.py, WHICH DOES NOT "
                  "EXIST\n" % (doc, line, fig, mod))
    out.write("\n%d stale, %d fresh, %d not attributable to a script within "
              "%d chars\n" % (len(stale), fresh, unattr, near))
    if unattr:
        out.write("  ^ the %d unattributable are NOT a clean bill of health. "
                  "This tool\n    examines only figures it can tie to a "
                  "scorer.\n" % unattr)
    out.write("\nA hit is UNVERIFIED, never WRONG: read the commit subject. A "
              "docstring\ncommit moves the date and changes no number.\n")
    return 0


def self_test(out=sys.stdout):
    import shutil
    import tempfile
    checks = []

    def check(name, ok):
        checks.append((name, ok))

    # ---- the parser, on text shaped like the real docs -------------------
    txt = ("python -m scripts.tralo_wins --campaign r\n"
           "#   RUN 2026-09-06 over the whole live corpus: 6 of 17.\n")
    got = figures(txt)
    check("a stamped figure is found and attributed to the script above it",
          got == [(2, "2026-09-06", "tralo_wins")])

    check("NEGATIVE CONTROL: a script under ANOTHER dir is not read as ours",
          figures("see docs/paper/scripts/make_main_table.py\n"
                  "MEASURED 2026-08-25: the table.\n")[0][2] is None)

    check("a bare date in prose is NOT a figure",
          figures("the dataset was removed 2026-09-02 and is unrunnable\n") == [])

    # NEGATIVE CONTROL: attribution must not reach past the window.
    far = ("python -m scripts.tralo_wins\n" + ("x\n" * 900)
           + "MEASURED 2026-09-06: 6 of 17.\n")
    check("NEGATIVE CONTROL: a script too far above is NOT attributed",
          figures(far)[0][2] is None)

    # ---- the date comparison, both directions, against a real repo -------
    tmp = tempfile.mkdtemp(prefix="stalefig_")
    cwd = os.getcwd()
    try:
        os.chdir(tmp)
        os.makedirs("scripts")
        subprocess.run(["git", "init", "-q"], capture_output=True)
        subprocess.run(["git", "config", "user.email", "t@t"], capture_output=True)
        subprocess.run(["git", "config", "user.name", "t"], capture_output=True)
        io.open("scripts/mover.py", "w", encoding="utf-8").write("x = 1\n")
        io.open("scripts/still.py", "w", encoding="utf-8").write("x = 1\n")
        subprocess.run(["git", "add", "-A"], capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "scorer: fix the deltas",
                        "--date", "2026-09-08T00:00:00"],
                       capture_output=True,
                       env=dict(os.environ,
                                GIT_COMMITTER_DATE="2026-09-08T00:00:00"))
        _CACHE.clear()

        io.open("D.md", "w", encoding="utf-8").write(
            "python -m scripts.mover --x\nRUN 2026-09-01: the figure.\n")
        stale, fresh, unattr, absent = scan(["D.md"])
        check("a figure OLDER than its scorer's last commit is STALE",
              len(stale) == 1 and stale[0][2] == "mover"
              and stale[0][5] == "scorer: fix the deltas")

        # NEGATIVE CONTROL: the whole point is that this must NOT fire.
        io.open("F.md", "w", encoding="utf-8").write(
            "python -m scripts.still --x\nRUN 2026-09-09: the figure.\n")
        stale2, fresh2, _u, _a = scan(["F.md"])
        check("NEGATIVE CONTROL: a figure NEWER than its scorer is NOT stale",
              stale2 == [] and fresh2 == 1)

        # NEGATIVE CONTROL: same-day must not fire -- the tool has no clock
        # finer than a date, and flagging a same-day pair would fire on every
        # figure written the day its scorer was last touched.
        io.open("S.md", "w", encoding="utf-8").write(
            "python -m scripts.mover --x\nRUN 2026-09-08: the figure.\n")
        stale3, _f, _u, _a = scan(["S.md"])
        check("NEGATIVE CONTROL: a SAME-DAY figure is not flagged", stale3 == [])

        # An absent module is NAMED, not crashed on and not silently dropped.
        io.open("A.md", "w", encoding="utf-8").write(
            "python -m scripts.ghost --x\nRUN 2026-09-01: the figure.\n")
        _s, _f, _u, absent4 = scan(["A.md"])
        check("a figure citing an ABSENT script is named, not dropped",
              len(absent4) == 1 and absent4[0][2] == "ghost")

        # An unattributable figure is COUNTED, never silently dropped.
        io.open("U.md", "w", encoding="utf-8").write("RUN 2026-09-01: a figure.\n")
        _s, _f, unattr5, _a = scan(["U.md"])
        check("an unattributable figure is COUNTED, not dropped", unattr5 == 1)

        buf = _Buf()
        report(["U.md"], out=buf)
        check("the report SAYS the unattributable are not a clean bill",
              "NOT a clean bill of health" in buf.text)

        check("an absent doc is skipped, not a crash", scan(["nope.md"])[1] == 0)
    finally:
        os.chdir(cwd)
        _CACHE.clear()
        shutil.rmtree(tmp, ignore_errors=True)

    bad = [n for n, ok in checks if not ok]
    for n, ok in checks:
        out.write("  %s %s\n" % ("PASS" if ok else "FAIL", n))
    out.write("%s: %d/%d\n" % ("SELF-TEST PASSED" if not bad else "FAILED",
                               len(checks) - len(bad), len(checks)))
    return 1 if bad else 0


class _Buf(object):
    def __init__(self):
        self.text = ""

    def write(self, s):
        self.text += s


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(self_test())
    sys.exit(report())
