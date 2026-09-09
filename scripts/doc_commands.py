"""DOES EVERY COMMAND IN THE DOCS ACTUALLY RUN?

The docs are the operating manual: 117 checkable invocations across
CLAUDE.md, FRAMEWORK.md, MISSION.md, PLAYBOOK.md and COVERAGE.md. They are
copy-pasted at exactly the wrong moment -- a campaign has just landed and a
number is wanted -- and a flag that argparse rejects costs a debugging cycle
right there.

Found by hand 2026-09-07: a pre-registered command in FRAMEWORK 2(z51) read
`latch_probe --glob '<pattern>'`. `latch_probe` takes `--campaign` and `--arms`
and has never had a `--glob`. It was written, reviewed, committed and would have
failed on first use.

This is the SAME CLASS as this project's most frequent failure mode -- a flag
that does not exist, or exists and is read by nothing (five inert flags and
counting). `audit_config` gates config KEYS against their readers. Nothing gated
the flags in the DOCS against argparse.

STATIC, by AST, and it never executes the modules it checks. `python -m X --help`
would be more faithful and would also run any module that forgot argparse, on a
machine that may have a live campaign on it. Reading `add_argument` calls costs
nothing and cannot have a side effect.

WHAT IT CANNOT SEE, stated so the green is not over-read:
  * a flag built dynamically (a name in a variable, a loop over a list). The
    module is reported UNPARSED and its flags are NOT checked -- never silently
    passed.
  * whether the flag DOES anything once parsed. That is `flag_live` and an md5,
    and md5 is one-sided (2(x2)).
  * whether the VALUES are right -- a real root, a cap that exists.

Exit code is 1 on any unknown flag or missing module, so it drops into CI.
"""

import argparse
import ast
import io
import os
import re
import sys

DOCS = (
    "CLAUDE.md",
    "docs/FRAMEWORK.md",
    "docs/MISSION.md",
    "docs/PLAYBOOK.md",
    "docs/COVERAGE.md",
)

INVOKE = re.compile(r"python\s+-u?\s*-?m?\s*((?:scripts|configs)\.[A-Za-z_][A-Za-z_0-9]*)")

# 🛑 THE PATH FORM, AND IT IS WHY THE ONE ABSENT MODULE WAS NEVER CAUGHT
# (2026-09-09). `INVOKE` matches `python -m scripts.X` only. The docs also
# write `scripts/X --flag`, and `scripts/ens_panel` -- documented FOUR times in
# FRAMEWORK, describing output and a gate -- does not exist in this checkout at
# all. Zero of those four mentions matched.
# ⚠️ IT REQUIRES A FOLLOWING FLAG, DELIBERATELY. A looser rule -- "any
# `scripts/<name>` not ending in .py" -- fires on `configs/task_windows.yml`
# (a YAML file, not a module), on `configs/protocol.yml`, and on glob prose
# like `scripts/prep_*`: 16 false positives on the real docs, measured. A gate
# that cries wolf gets switched off, so this trades recall for precision and
# matches only what is unambiguously an invocation.
INVOKE_PATH = re.compile(
    r"\b((?:scripts|configs))/([A-Za-z_][A-Za-z_0-9]*)(?=\s+--[A-Za-z])")

FLAG = re.compile(r"(--[A-Za-z][A-Za-z0-9-]*)")

# ⚠️ MODULES THE DOCS NAME THAT THIS CHECKOUT DOES NOT CONTAIN.
# An entry is a DEFECT WITH A TICKET, never a permission. The gate still
# prints every one of them, loudly, on every run; what the entry buys is that
# an UNFIXABLE-today defect does not block a launch that has nothing to do with
# it. The fix is to merge the code or delete the claim, not to add a line here.
# 🛑 AND A STALE ENTRY IS ITSELF A FAILURE: if the module comes back, the gate
# says so, because an allowlist nobody prunes becomes a list of things nobody
# checks.
ABSENT_OK = {
    "scripts.ens_panel":
        "exists only on the server branch `snap/slice-provenance`, which was "
        "never merged and is not among the 40 remote-tracking branches here "
        "(last server fetch 2026-09-08). `snap2` is RUNNING from it, so its "
        "results will carry a `code_version` no other checkout can resolve. "
        "FRAMEWORK 2(z67). Blocked on host access.",
}

# argparse supplies these itself.
BUILTIN = frozenset(("--help",))


def join_continuations(lines, i):
    """The invocation on line i, with any backslash-continued lines appended.

    Markdown keeps a trailing `\\` literally, so a command wrapped for width is
    several lines in the file and one command to the reader.
    """
    out = []
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        cont = raw.rstrip().endswith("\\")
        out.append(raw.rstrip().rstrip("\\"))
        i += 1
        if not cont:
            break
    return " ".join(out)


def command_text(text):
    """Everything argparse would see: stop at a comment, a pipe or a chain."""
    for stop in ("  #", "\t#", " | ", " && ", " ; ", " > "):
        k = text.find(stop)
        if k != -1:
            text = text[:k]
    return text


def invocations(path):
    """(line_no, module, [flags]) for every documented invocation in `path`."""
    lines = io.open(path, encoding="utf-8", errors="replace").readlines()
    found = []
    for i, line in enumerate(lines):
        m = INVOKE.search(line)
        if m:
            whole = join_continuations(lines, i)
            # Re-find in the joined text so flags on continuations count.
            k = whole.find(m.group(1))
            tail = command_text(whole[k + len(m.group(1)):])
            found.append((i + 1, m.group(1),
                          sorted(set(FLAG.findall(tail)))))
            continue
        # The `scripts/X` path form. Same module, written the other way, and
        # invisible to `INVOKE` -- which is how a module absent from the whole
        # checkout stayed undetected through four documented mentions.
        mp = INVOKE_PATH.search(line)
        if mp:
            mod = "%s.%s" % (mp.group(1), mp.group(2))
            whole = join_continuations(lines, i)
            k = whole.find(mp.group(0))
            tail = command_text(whole[k + len(mp.group(0)):])
            found.append((i + 1, mod, sorted(set(FLAG.findall(tail)))))
    return found


def declared_flags(mod):
    """(flags, dynamic) declared by `add_argument` in the module's source.

    `dynamic` is True when a call's option string is not a literal, which means
    the flag set is INCOMPLETE and must not be used to reject anything.
    """
    path = os.path.join(*mod.split(".")) + ".py"
    if not os.path.exists(path):
        return None, False
    tree = ast.parse(io.open(path, encoding="utf-8", errors="replace").read())
    flags, dynamic, saw_any = set(), False, False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        saw_any = True
        positional = [a for a in node.args]
        if not positional:
            dynamic = True
            continue
        for a in positional:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                if a.value.startswith("-"):
                    flags.add(a.value)
            else:
                dynamic = True
    if not saw_any:
        return set(), True          # no argparse at all -> cannot judge
    return flags | set(BUILTIN), dynamic


def run(docs, verbose):
    cache = {}
    bad, unparsed, checked = [], set(), 0
    absent_seen = set()
    for doc in docs:
        if not os.path.exists(doc):
            print("  -- %s: absent, skipped" % doc)
            continue
        for lineno, mod, flags in invocations(doc):
            if mod not in cache:
                cache[mod] = declared_flags(mod)
            declared, dynamic = cache[mod]
            if declared is None:
                if mod in ABSENT_OK:
                    absent_seen.add(mod)
                    continue
                bad.append((doc, lineno, mod, "NO SUCH MODULE"))
                continue
            if dynamic:
                unparsed.add(mod)
                continue
            checked += 1
            unknown = [f for f in flags if f not in declared]
            if unknown:
                bad.append((doc, lineno, mod,
                            "flag(s) argparse rejects: " + " ".join(unknown)))
            elif verbose:
                print("  ok   %s:%d  %s %s" % (doc, lineno, mod, " ".join(flags)))

    print("")
    print("%d invocation(s) checked against argparse" % checked)
    if unparsed:
        print("%d module(s) build flags dynamically and were NOT checked: %s"
              % (len(unparsed), " ".join(sorted(unparsed))))

    # Announce every allowlisted absence on every run. An entry buys "does not
    # block an unrelated launch", never silence -- the whole failure mode this
    # tool exists for is a plausible-looking green.
    for mod in sorted(absent_seen):
        print("")
        print("!! DOCUMENTED BUT ABSENT: %s" % mod)
        print("   %s" % ABSENT_OK[mod])
        print("   This is a DEFECT with a ticket, not a permission. The fix is "
              "to merge the code or delete the claim.")

    # 🛑 A STALE ALLOWLIST ENTRY IS A FAILURE. If the module is back, the line
    # excusing it is now excusing nothing and hiding the next absence.
    for mod in sorted(ABSENT_OK):
        if declared_flags(mod)[0] is not None:
            bad.append(("scripts/doc_commands.py", 0, mod,
                        "STALE ABSENT_OK ENTRY -- this module EXISTS now; "
                        "remove it from ABSENT_OK"))

    if not bad:
        print("ALL DOCUMENTED COMMANDS PARSE")
        return bad
    print("")
    print("%d BROKEN COMMAND(S):" % len(bad))
    for doc, lineno, mod, why in bad:
        print("  %s:%d  %s -- %s" % (doc, lineno, mod, why))
    return bad


# --------------------------------------------------------------------------
# self-test: a gate that has never failed has never been shown to work.

def self_test():
    import shutil
    import tempfile

    ok = [True]

    def check(cond, msg):
        print("  %s %s" % ("PASS" if cond else "FAIL", msg))
        if not cond:
            ok[0] = False

    tmp = tempfile.mkdtemp(prefix="doccmd_")
    try:
        pkg = os.path.join(tmp, "scripts")
        os.makedirs(pkg)
        io.open(os.path.join(pkg, "__init__.py"), "w").close()
        io.open(os.path.join(pkg, "widget.py"), "w", encoding="utf-8").write(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--campaign')\n"
            "p.add_argument('--arms', nargs='+')\n")
        io.open(os.path.join(pkg, "dyn.py"), "w", encoding="utf-8").write(
            "import argparse\n"
            "p = argparse.ArgumentParser()\n"
            "for n in ('--a', '--b'):\n"
            "    p.add_argument(n)\n")

        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            # POSITIVE control: the real defect that motivated the tool.
            io.open("bad.md", "w", encoding="utf-8").write(
                "run `python -m scripts.widget --glob 'x/*'` to check\n")
            bad = run(["bad.md"], False)
            check(len(bad) == 1, "an undeclared flag (--glob) is CAUGHT")

            # NEGATIVE control: the corrected form must NOT fire.
            io.open("good.md", "w", encoding="utf-8").write(
                "run `python -m scripts.widget --campaign r --arms a b`\n")
            check(run(["good.md"], False) == [],
                  "  and the corrected command is NOT flagged")

            # NEGATIVE control: a comment after the command is not argparse's.
            io.open("cmt.md", "w", encoding="utf-8").write(
                "python -m scripts.widget --campaign r   # and then --glob it\n")
            check(run(["cmt.md"], False) == [],
                  "a flag inside a trailing COMMENT is not attributed to argparse")

            # NEGATIVE control: a second program after a pipe owns its own flags.
            io.open("pipe.md", "w", encoding="utf-8").write(
                "python -m scripts.widget --campaign r | grep --color x\n")
            check(run(["pipe.md"], False) == [],
                  "a flag after a PIPE belongs to the other program")

            # A backslash-continued command is ONE command.
            io.open("cont.md", "w", encoding="utf-8").write(
                "python -m scripts.widget --campaign r \\\n    --glob x\n")
            check(len(run(["cont.md"], False)) == 1,
                  "a bad flag on a CONTINUATION line is still caught")

            # A missing module is a distinct failure, not a flag complaint.
            io.open("gone.md", "w", encoding="utf-8").write(
                "python -m scripts.vanished --campaign r\n")
            check(len(run(["gone.md"], False)) == 1, "a module that does not exist is CAUGHT")

            # ---- THE PATH FORM, scripts/X --flag (2026-09-09) -----------
            # `scripts/ens_panel` is documented FOUR times in FRAMEWORK and
            # does not exist in the checkout. Zero mentions matched, because
            # every one is written `scripts/X`, not `python -m scripts.X`.
            io.open('path.md', 'w', encoding='utf-8').write(
                'Read `scripts/widget --campaign r` for the table.' + chr(10))
            check(run(['path.md'], False) == [],
                  'the PATH form scripts/X --flag is parsed at all')

            io.open('pathbad.md', 'w', encoding='utf-8').write(
                'Read `scripts/widget --nope` for the table.' + chr(10))
            check(len(run(['pathbad.md'], False)) == 1,
                  'a bad flag on the PATH form is CAUGHT')

            io.open('pathgone.md', 'w', encoding='utf-8').write(
                'Read `scripts/vanished --campaign r`.' + chr(10))
            check(len(run(['pathgone.md'], False)) == 1,
                  'an ABSENT module named in the PATH form is CAUGHT')

            # NEGATIVE CONTROLS: the three shapes that made the loose version
            # of this rule fire 16 times on the real docs.
            io.open('yaml.md', 'w', encoding='utf-8').write(
                'the windows live in `configs/task_windows.yml` and are read'
                + chr(10))
            check(run(['yaml.md'], False) == [],
                  'NEGATIVE CONTROL: a YAML path is not read as a module')

            io.open('glob.md', 'w', encoding='utf-8').write(
                'the family is `scripts/prep_*` and none takes --flags'
                + chr(10))
            check(run(['glob.md'], False) == [],
                  'NEGATIVE CONTROL: a glob is not read as a module')

            io.open('pyfile.md', 'w', encoding='utf-8').write(
                # A REAL filename, because `test_no_live_file_points_at_a_deleted_doc`
                # scans every live file for `scripts/*.py` paths and demands they
                # exist -- an invented one turns that gate red. It also makes the
                # control faithful: real docs DO reference `scripts/full_panel.py`.
                'run `scripts/full_panel.py --nope` by hand' + chr(10))
            check(run(['pyfile.md'], False) == [],
                  'NEGATIVE CONTROL: a .py FILE reference is not an invocation')

            # Dynamic construction must ABSTAIN, never pass or fail silently.
            io.open("dyn.md", "w", encoding="utf-8").write(
                "python -m scripts.dyn --anything-at-all\n")
            check(run(["dyn.md"], False) == [],
                  "a module building flags dynamically ABSTAINS (reported, not judged)")

            # An absent doc is skipped, not a crash.
            check(run(["no_such_doc.md"], False) == [], "an absent doc is skipped")
        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    print("SELF-TEST %s" % ("PASSED" if ok[0] else "FAILED"))
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", nargs="+", default=list(DOCS))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    return 1 if run(a.docs, a.verbose) else 0


if __name__ == "__main__":
    sys.exit(main())
