"""WHAT IN THIS REPO IS DEAD? Functions, classes and module constants that
nothing references.

AST, never grep. This project already learned the lesson on CONFIG KEYS: a
grep says `rho_step` is read, because a LOG LINE names it. The same trap
applies to code -- a function named in a docstring, a comment, or an f-string
is not called, and a grep cannot tell the difference.

WHAT COUNTS AS A USE, and each of these was a false positive first:

  * a plain call            `foo()`
  * a bare reference        `key=foo`, `[foo, bar]`, `functools.partial(foo)`
  * an attribute access     `mod.foo` -- covers `scripts.task_window.MIN_PRIZE`
  * an import               `from x import foo`, and `import x as foo`
  * a decorator             `@foo`
  * a bare identifier string, which is how a dispatch dict or `__all__` names
    a symbol. Kept narrow: the WHOLE string must be an identifier, so prose
    can never rescue a dead name.

Anything reachable from `main` / `self_test`, and any `test_*`, is an entry
point the interpreter or pytest calls for us, so it is never reported.

Exit code is 0 always: this is a REPORT, not a gate. A "dead" symbol here is a
candidate for deletion that a human still has to confirm, because the one thing
AST cannot see is a call assembled at runtime (`getattr(mod, name)`), and this
repo has those in its arm dispatch.

Usage:
    python -m scripts.dead_code                  # configs/ scripts/ src/
    python -m scripts.dead_code --paths configs
    python -m scripts.dead_code --self-test
"""

import argparse
import ast
import os
import sys

DEFAULT_PATHS = ("configs", "scripts", "src")
ENTRY = ("main", "self_test", "_self_test", "__init__", "__main__")


def _iter_py(paths, root="."):
    for p in paths:
        base = os.path.join(root, p)
        if os.path.isfile(base) and base.endswith(".py"):
            yield base
            continue
        for dirpath, dirnames, files in os.walk(base):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git", "node_modules")]
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(dirpath, f)


def defined(tree):
    """Top-level definitions, plus module-level UPPER_CASE constants."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out[node.name] = node.lineno
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id.isupper():
                    out[t.id] = node.lineno
    return out


def used(tree):
    """Every name this module references in a way that could be a use."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            # ONLY a Load. An assignment TARGET is a Name node too, so
            # counting it made every module constant reference ITSELF and
            # `DEAD_CONST = 1` could never be reported. Caught by this
            # module's own self-test.
            if isinstance(node.ctx, ast.Load):
                names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                names.add(a.name)
                if a.asname:
                    names.add(a.asname)
        elif isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.isidentifier():
                names.add(node.value)
    return names


def scan(paths, root="."):
    """({(file, name): line}, {name})"""
    defs, uses = {}, set()
    for path in _iter_py(paths, root):
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = os.path.relpath(path, root)
        for name, line in defined(tree).items():
            defs[(rel, name)] = line
        uses |= used(tree)
    return defs, uses


def dead(defs, uses):
    out = []
    for (path, name), line in sorted(defs.items()):
        if name in ENTRY or name.startswith("test_"):
            continue
        if name in uses:
            continue
        out.append((path, name, line))
    return out


def self_test(w=sys.stdout.write):
    """Both directions: a dead symbol is found, a live one is not."""
    ok = True

    def check(good, label):
        nonlocal ok
        w("  %-4s %s\n" % ("PASS" if good else "FAIL", label))
        ok = ok and good

    import tempfile
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "pkg"))
    open(os.path.join(d, "pkg", "a.py"), "w").write(
        "DEAD_CONST = 1\n"
        "LIVE_CONST = 2\n"
        "def dead_fn():\n    pass\n"
        "def live_fn():\n    pass\n"
        "def called_by_attr():\n    pass\n"
        "def named_in_docstring():\n    '''nothing'''\n")
    open(os.path.join(d, "pkg", "b.py"), "w").write(
        "import pkg.a as a\n"
        "from pkg.a import live_fn\n"
        "def go():\n"
        "    live_fn()\n"
        "    a.called_by_attr()\n"
        "    return a.LIVE_CONST\n"
        "# named_in_docstring is only mentioned in this comment\n"
        "X = 'prose mentioning dead_fn does not rescue it'\n")
    defs, uses = scan(["pkg"], root=d)
    names = {n for _, n, _ in dead(defs, uses)}

    check("dead_fn" in names, "an uncalled function is reported")
    check("DEAD_CONST" in names, "an unreferenced module constant is reported")
    check("live_fn" not in names, "LIVENESS: an imported+called function is NOT")
    check("called_by_attr" not in names,
          "LIVENESS: a `mod.attr()` call counts as a use")
    check("LIVE_CONST" not in names,
          "LIVENESS: a `mod.CONST` reference counts as a use")
    check("named_in_docstring" in names,
          "a name appearing ONLY in a comment/docstring is still dead")

    open(os.path.join(d, "pkg", "c.py"), "w").write(
        "from pkg.a import dead_fn\nHANDLERS = {'x': dead_fn}\n")
    defs, uses = scan(["pkg"], root=d)
    names = {n for _, n, _ in dead(defs, uses)}
    check("dead_fn" not in names,
          "LIVENESS: a bare reference in a dispatch dict rescues it")

    w("\nSELF-TEST %s\n" % ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paths", nargs="+", default=list(DEFAULT_PATHS))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    defs, uses = scan(a.paths)
    rows = dead(defs, uses)
    print("scanned %d definition(s) across %s" % (len(defs), " ".join(a.paths)))
    if not rows:
        print("nothing unreferenced.")
        return 0
    print("")
    print("%d symbol(s) with NO reference anywhere in the scanned tree." % len(rows))
    print("Confirm each by hand: a call built with getattr() is invisible here.")
    print("")
    cur = None
    for path, name, line in rows:
        if path != cur:
            print("  %s" % path)
            cur = path
        print("     %-42s line %d" % (name, line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
