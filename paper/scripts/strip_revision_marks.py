"""Produce the clean manuscript: revisions accepted, marks removed.

docs/main.tex carries the revision apparatus -- \\del{...} for superseded text in
red, \\rev{...} / {\\color{blue}...} for additions, \\pending{...} for numbers still
awaiting a rerun. That is what the professor reads. It is NOT what a reviewer or
a submission should see.

This applies the revisions and strips the apparatus:

    \\del{X}            -> (removed)
    \\rev{X}            -> X
    {\\color{blue}X}    -> X
    \\pending{X}        -> (removed, but COUNTED and reported -- a surviving
                          \\pending means a number was never recomputed)
    bare \\color{blue}  -> (removed; used inside table floats)
    \\captionsetup{labelfont={color=blue}} -> (removed)

Brace matching is explicit rather than regex-based: these macros routinely wrap
nested braces (\\mbox{$...$}, \\textbf{...}), which a regex would truncate.

Usage:
    python paper/scripts/strip_revision_marks.py docs/main.tex -o paper/main_clean.tex
"""

import argparse
import re
import sys


def find_matching(s, open_idx):
    """Index of the brace matching the '{' at open_idx, or -1."""
    assert s[open_idx] == "{"
    depth = 0
    i = open_idx
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":          # skip escaped char, e.g. \{ \} \\
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


# Dropped content leaves this marker rather than nothing, so that a line which
# held ONLY a \del{...} can be deleted outright instead of becoming a blank line.
# A blank line inside a \caption{} is a \par, which is a hard LaTeX error -- and
# it is the normal case here, since deletions are usually written on their own
# line for readability.
SENTINEL = "\x00DELETED\x00"


def strip_macro(s, macro, keep_content):
    """Remove \\macro{...}, keeping or dropping its argument. Returns (s, count)."""
    pat = "\\" + macro + "{"
    out = []
    i = 0
    n = 0
    while True:
        j = s.find(pat, i)
        if j < 0:
            out.append(s[i:])
            break
        close = find_matching(s, j + len(pat) - 1)
        if close < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        out.append(s[j + len(pat):close] if keep_content else SENTINEL)
        n += 1
        i = close + 1
    return "".join(out), n


def drop_emptied_lines(s):
    """Delete lines that exist only to hold removed content; clear the rest."""
    kept = []
    dropped = 0
    for line in s.split("\n"):
        if SENTINEL in line:
            if line.replace(SENTINEL, "").strip() == "":
                dropped += 1
                continue
            line = line.replace(SENTINEL, "")
        kept.append(line)
    return "\n".join(kept), dropped


def drop_emptied_headings(s):
    """Remove run-in headings whose entire title was deleted.

    "\\paragraph{\\del{Old heading.}}\\leavevmode" strips to "\\paragraph{}\\leavevmode",
    which typesets as a ~2-line band of empty space -- visible in the compiled PDF
    and invisible in the source. The \\leavevmode is only there to force the run-in
    heading out before \\del opens its colour group, so it goes with the heading.
    """
    return re.subn(r"\\paragraph\{\}\s*(\\leavevmode)?\s*\n?", "", s)


def strip_color_groups(s):
    """Remove the \\color{blue} switch, leaving braces alone. Returns (s, count).

    DO NOT try to unwrap "{\\color{blue}...}" as a group. That pattern is textually
    indistinguishable from "\\caption{\\color{blue}...}", where the brace belongs to
    \\caption -- matching and deleting it strips the caption's own braces and
    silently yields "\\caption\\textbf{...}", which still COMPILES (zero errors) but
    typesets the caption as body text inside the float. A blind reviewer caught
    exactly that on three captions in the first clean build.

    Deleting only the switch token is both correct and simpler: "\\caption{X}" is
    right, and a leftover "{X}" from a genuine wrapper is an inert brace group.
    """
    return re.subn(r"\\color\{blue\}\s*", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    s = open(args.src, encoding="utf-8").read()

    # Comment lines are not typeset; a \pending mentioned in a comment is not a
    # live marker, so exclude comments before counting.
    live = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in s.split("\n"))
    n_live_pending = live.count("\\pending{")

    s, n_pending = strip_macro(s, "pending", keep_content=False)
    s, n_del = strip_macro(s, "del", keep_content=False)
    s, n_rev = strip_macro(s, "rev", keep_content=True)
    s, n_col = strip_color_groups(s)

    # Leftovers that are not brace-wrapped.
    s, n_cs = re.subn(r"\\captionsetup\{labelfont=\{color=blue\}\}\s*", "", s)
    s, n_bare = re.subn(r"\\color\{blue\}\s*", "", s)

    # The apparatus itself.
    s = re.sub(r"^\\newcommand\{\\(del|rev|pending)\}.*\n", "", s, flags=re.M)
    s = re.sub(r"^\\usepackage\[normalem\]\{ulem\}.*\n", "", s, flags=re.M)

    # Comments that only explain the apparatus. They do not typeset, but a
    # reviewer read one ("\color{blue} must be repeated INSIDE \caption") as
    # evidence that a de-marking script had eaten something -- which, that round,
    # it had. Leaving them in the clean build invites the same inference.
    s = re.sub(r"^%[^\n]*\\?color\{blue\}[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r"^% *\\?(del|rev|pending)\{[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r"^% *must be repeated INSIDE[^\n]*\n", "", s, flags=re.M)
    s = re.sub(r"^% *\\normalcolor before typesetting[^\n]*\n", "", s, flags=re.M)

    s, n_lines = drop_emptied_lines(s)
    s, n_head = drop_emptied_headings(s)

    open(args.out, "w", encoding="utf-8").write(s)
    print("lines removed (held only deleted text): %d" % n_lines)
    print("emptied run-in headings removed  : %d" % n_head)

    print("deleted  \\del{...}      : %d" % n_del)
    print("accepted \\rev{...}      : %d" % n_rev)
    print("unwrapped {\\color{blue}}: %d" % n_col)
    print("bare \\color{blue}       : %d" % n_bare)
    print("captionsetup removed    : %d" % n_cs)
    print("wrote %s" % args.out)

    leftover = s.count("\\del{") + s.count("\\color{blue}") + s.count("\\rev{")
    if leftover:
        print("\nWARNING: %d revision marker(s) survived -- inspect" % leftover)
    if n_live_pending:
        print("\n!! %d live \\pending{} marker(s) were dropped. Each is a number that "
              "was never recomputed. Do NOT submit this until they are resolved in "
              "the marked manuscript." % n_live_pending)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
