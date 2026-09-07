"""DOES EVERY COMMAND IN THE DOCS ACTUALLY RUN?

The docs are the operating manual: 118 `python -m scripts.<name>` invocations across
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
FLAG = re.compile(r"(--[A-Za-z][A-Za-z0-9-]*)")

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
        if not m:
            continue
        whole = join_continuations(lines, i)
        # Re-find within the joined text so flags on continuation lines count.
        k = whole.find(m.group(1))
        tail = command_text(whole[k + len(m.group(1)):])
        found.append((i + 1, m.group(1), sorted(set(FLAG.findall(tail)))))
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
    for doc in docs:
        if not os.path.exists(doc):
            print("  -- %s: absent, skipped" % doc)
            continue
        for lineno, mod, flags in invocations(doc):
            if mod not in cache:
                cache[mod] = declared_flags(mod)
            declared, dynamic = cache[mod]
            if declared is None:
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
