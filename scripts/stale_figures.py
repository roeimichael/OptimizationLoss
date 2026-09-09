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
78 date-stamped figures, 72 attributable, 5 not, and 1 citing a script that
no longer exists. The 5 are printed as a count so the coverage is never
mistaken for completeness -- a tool that silently examines part of its input
reads exactly like one that found nothing wrong.

⚠️ AND IT OVER-REPORTS IN THE OTHER DIRECTION. 56 of the 72 are flagged,
which is a SCREENING result, not 56 wrong numbers: the wider the attribution
window, the more figures get tied to a script that did not produce them, and
the scorers here were heavily edited through September for unrelated reasons.
Read it as a QUEUE ordered by how much a number matters -- `paper_rows` heads
it with 12 -- never as a count of defects.
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


def _sections(text):
    """(start, end) of every markdown section, so attribution cannot cross one.

    ⚠️ FENCE-AWARE, and it has to be. A naive `^#{1,2} ` split reads every
    `#   RUN 2026-09-06 ...` COMMENT LINE inside CLAUDE.md's fenced command
    blocks as a heading, chopping the file into hundreds of one-line sections
    and taking its attribution from 10 figures to 4. The whole value of the
    tool is on the other side of that bug.

    ⚠️ h1 AND h2 ONLY, AND THE DEPTH IS MEASURED RATHER THAN CHOSEN BY TASTE.
    `docs/FRAMEWORK.md` carries 1 h1, 37 h2 and 218 h3, and the h2s ARE the
    numbered entries (`## 2(z68). ...`) while the h3s sit INSIDE one. Over the
    five docs: splitting at h3 attributes 46 of 78 (it breaks an entry apart),
    at h2 it is 73 of 78, and h1-only reaches 77 by letting a figure
    claim any script in the same chapter -- higher coverage bought with
    misattribution, which is worse than a blind figure because it names the
    wrong scorer with a straight face.
    """
    starts, fenced, pos = [], False, 0
    for raw in text.split("\n"):
        if raw.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,2} \S", raw):
            starts.append(pos)
        pos += len(raw) + 1
    if not starts or starts[0] != 0:
        starts = [0] + starts
    return list(zip(starts, starts[1:] + [len(text)]))


def figures(text, near=NEAR):
    """Every (line, figure_date, module) this doc stamps. module may be None.

    Attribution is SECTION-SCOPED, and that is what makes it useful rather than
    merely present. A plain "nearest script within `near` characters, looking
    backwards" rule attributed only 10 of FRAMEWORK's 55 figures -- an entry
    states its measurement and names its tool a long way apart, and often names
    it AFTER. Bounding the search by the enclosing `##` heading and falling
    FORWARD inside that same section takes it to 53 of 55, and the five docs
    together from 21 of 78 to 73 of 78. Measured at each step, not assumed.

    The section bound is the part that keeps it honest -- without it, looking
    forward would let a figure in one entry grab a script named in the next.
    """
    found = []
    secs = _sections(text)
    for m in FIGURE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        lo, hi = next(((a, b) for a, b in secs if a <= m.start() < b),
                      (0, len(text)))
        back = SCRIPT.findall(text[max(lo, m.start() - near):m.start()])
        fwd = SCRIPT.findall(text[m.end():hi]) if not back else []
        mod = back[-1] if back else (fwd[0] if fwd else None)
        found.append((line, m.group(1), mod))
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

    # NEGATIVE CONTROL for the forward fallback: it must stop at the section
    # boundary. Without the bound, a figure at the end of one entry would
    # claim the first script named in the NEXT one -- which is exactly the
    # misattribution that makes a report worse than useless.
    across = ("## 2(z1). an entry\nRUN 2026-09-06: a figure.\n"
              "## 2(z2). the next entry\npython -m scripts.tralo_wins --x\n")
    check("NEGATIVE CONTROL: the forward fallback stops at the section bound",
          figures(across)[0][2] is None)

    # ...and the POSITIVE half, or the control above passes by doing nothing.
    within = ("## 2(z1). an entry\nRUN 2026-09-06: a figure.\n"
              "python -m scripts.tralo_wins --x\n")
    check("a script named LATER in the SAME section is attributed",
          figures(within)[0][2] == "tralo_wins")

    # A `#   RUN ...` comment inside a fence must not read as a heading.
    fenced = ("## an entry\n```bash\npython -m scripts.tralo_wins --x\n"
              "#   RUN 2026-09-06: a figure.\n```\n")
    check("a fenced `#` COMMENT is not read as a section heading",
          figures(fenced)[0][2] == "tralo_wins")

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
        # Built from parts, never as a literal `scripts/<name>.py` string:
        # `test_no_live_file_points_at_a_deleted_doc` scans every live .py for
        # path-shaped strings and demands the target exist, and these two are
        # fixtures that live only in a temp dir. Writing them literally turned
        # that gate red -- the second time in one sitting, after the same shape
        # in `doc_commands`. The gate is right; the fixture was wrong.
        for name in ("mover", "still"):
            io.open(os.path.join("scripts", name + ".py"), "w",
                    encoding="utf-8").write("x = 1\n")
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
