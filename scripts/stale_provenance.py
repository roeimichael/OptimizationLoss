"""WHICH QUOTED FIGURES COME FROM DATA THAT NO LONGER COUNTS?

`stale_figures` asks one staleness question -- has the SCORER moved since the
figure was written. This asks the other one, and it is not the same: has the
DATA been condemned since. A figure can be perfectly reproducible by today's
code and still be a number about a campaign that is `scorable=False`, a
campaign archived for running a different recipe, or a dataset that was
removed for leaking 38.7% of its own test set.

WHY IT EXISTS. Three separate instances in one day, none of which
`stale_figures` can see, because none of them is about code:

  * 2(z71) -- the seed-budget figures (`2607 seeds at L20`, the 6-12x pairing
    penalty) come from `iwc3`, which is `scorable=False` at 68.6% dose and
    whose `keep_for` covers the fp16-dose receipt only.
  * 2(z76) -- 2(w3), the project's ONLY positive result, is `results/loose1`,
    which runs `constraint_grad_mode: clip`; 2(z26-CORRECTED) had already
    removed `loose1`/RegNetY400MF from the unit corpus BY NAME as "a different
    method", and 2(w3) never said so.
  * 2(z30)(d) -- `1.9-9.9 items` is a dermmnist figure and was bare in 17
    places across 13 files, one of them a CLI DEFAULT that gates a verdict.

THE AUTHORITIES ARE READ, NOT RESTATED. Dead and PARTIAL campaigns come from
`scripts.quarantine.REGISTRY`; the recipe census comes from `docs/COVERAGE.md`
section 0, parsed from the table itself. A tool that hardcodes either would
drift from them, which is the defect it exists to find.

!! THIS IS A QUEUE, NOT A DEFECT COUNT, and the reason is measured. 2(z68)
found that attribution by proximity is irreducibly noisy: `stale_figures`
appeared to hand `paper_rows` twelve hits and reading all twelve by hand gave
roughly 5 genuine, 5 misattributed, 2 ambiguous. The same applies here, and
more strongly -- most entries citing dermmnist are ABOUT dermmnist, where the
figure is in context and correct. A hit means "an entry quotes a number and
names condemned data without saying so"; whether that invalidates the number
is a judgement a person makes. So the output is ordered by how much the number
matters, and the tool never prints a total it wants believed.

CALIBRATED BY HAND, 2026-09-10, BECAUSE AN UNCALIBRATED REPORT IS A RUMOUR.
The top SIX of the first run were read one by one and came out **3 genuine,
2 spurious, 1 ambiguous** -- the same ratio 2(z68) measured for
`stale_figures`. Quote that ratio beside any count taken from here.

The three genuine ones were the count-function gradient entry (its 53 figures
are `iwc1`'s and sit outside `iwc1`'s `keep_for`), the count-function reversal
(on `loose1`, the `clip` recipe, unstated), and the attribution table (on a
68.6%-dose campaign). The two spurious ones name a removed dataset only for
CONTRAST -- an iwildcam entry, and a meta entry about how wrong results got
believed. The ambiguous one is a dermmnist-era entry where the figure may be
in context.

!! AND READING THEM CHANGED THE TOOL, WHICH IS THE POINT OF CALIBRATING.

  * The FIRST genuine hit turned out to be a symptom: reading it found that
    `step_direction_probe` located the cut GLOBALLY in two places, the sixth
    such site in this repo and the tool behind that entry's headline. See
    FRAMEWORK 2(z79). A provenance flag found a mechanism defect.
  * The THIRD one showed the matcher was too narrow. It required backticks,
    and the entry it flagged names `iwc3` only in passing -- its actual table
    header reads "| iwc4 final, ... |", unbackticked, so the OFF-RECIPE
    campaign the whole table comes from was invisible. Matching moved to a
    word boundary; a campaign name is distinctive enough, and a backtick is a
    formatting habit rather than a signal. That took the run from 74/32 to
    **85 cleared / 38 flagged**, so the earlier counts are superseded.

    python -m scripts.stale_provenance
    python -m scripts.stale_provenance --docs docs/FRAMEWORK.md
    python -m scripts.stale_provenance --self-test
"""
import argparse
import io
import os
import re
import sys

from scripts import quarantine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS = ("CLAUDE.md", "docs/FRAMEWORK.md", "docs/MISSION.md",
        "docs/COVERAGE.md", "docs/PLAYBOOK.md")

# `docs/paper/` is the disjoint dermmnist generation (WHICH_CORPUS.md) and
# `docs/archive/` is history by definition. Neither is an instruction.
SKIP_DOCS = ("docs/paper", "docs/archive")

REMOVED_DATASETS = ("dermmnist", "octmnist", "tissuemnist")

# A figure, in the forms this project actually writes them.
FIGURE = re.compile(r"[+-]?\d+\.\d{3,4}\b|\bp\s*=\s*0\.\d+|\b\d+\.\d\s*x\b")

# The entry says the data is condemned. Any of these anywhere in the entry
# clears it -- deliberately generous, because a false CLEAR costs a person one
# unread entry and a false HIT costs them the credibility of the whole report.
DISCLOSED = re.compile(
    r"quarantin|scorable\s*=\s*False|keep_for|UNVERIFIED|remov|leak|"
    r"archiv|different method|not the current|off.recipe|stale.recipe|"
    r"do not quote|must not be quoted|dead", re.I)

# Entry granularity. FRAMEWORK mixes h2 (newer entries) and h3/h4 (older ones),
# so splitting at h2 alone lumps dozens of entries into one and hides them.
HEAD = re.compile(r"^#{2,4}\s")


def recipe_census(path=None):
    """{campaign: (fp32, grad_mode)} parsed from COVERAGE section 0's table.

    The table is the authority on which campaign ran which TraLO
    configuration; restating it here would let the two drift apart, which is
    exactly how `loose1` stayed the headline positive after being removed from
    the corpus for its recipe.
    """
    path = path or os.path.join(REPO, "docs", "COVERAGE.md")
    try:
        txt = io.open(path, encoding="utf-8").read()
    except OSError:
        return {}
    out = {}
    for line in txt.splitlines():
        raw = line.split("|")
        cells = [c.strip().strip("*` ") for c in raw]
        if len(cells) < 7:
            continue
        # | cfg | runs | fp32 | grad_mode | campaigns | note |
        # !! READ THE CAMPAIGNS CELL RAW (2026-09-10). Stripping "*` " off the
        # whole cell eats the OUTER backticks of a list, and the name regex
        # needs one on both sides -- so the FIRST and LAST campaign of every
        # row silently vanished. That cost `iwc4` and `vitu1`, which are
        # off-recipe and carry no quarantine marker, so nothing else could
        # have caught them: the tool reported 25 condemned names while its own
        # authority listed 27.
        fp32, mode, camps = cells[3], cells[4], raw[5]
        if fp32 not in ("True", "False") or mode not in ("clip", "normalize"):
            continue
        for name in re.findall(r"`([A-Za-z0-9_]+)`", camps):
            out[name] = (fp32 == "True", mode)
    return out


def current_recipe(census):
    """The recipe the most campaigns are on -- read, not asserted."""
    if not census:
        return None
    counts = {}
    for v in census.values():
        counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get)


def condemned(census=None, registry=None):
    """{name: (reason, required_arms)} for names whose numbers need disclosure.

    `required_arms` is the whole precision story. A `scorable=False` campaign
    condemns everything in it, so its requirement is empty and any figure
    beside it fires. A PARTIAL one does NOT: `quarantine.gate()` refuses at
    ARM granularity, and `dom1` is scorable for every contrast that does not
    touch `fioretto` or `hounie`. So a PARTIAL campaign fires only when the
    entry ALSO names one of its dead arms.

    !! WITHOUT THAT THE TOOL IS UNUSABLE, AND IT IS MEASURED: on the first
    run, three PARTIAL campaigns alone produced most of the queue -- `dom1`
    appears in dozens of entries about the count function, the scope split and
    the unit ledger, none of which reads a dual. A blanket campaign-level rule
    would have told a reader to re-check them all, which is the same error as
    a blanket quarantine marker deleting three independent units to describe a
    defect in two arms.
    """
    reg = quarantine.REGISTRY if registry is None else registry
    census = recipe_census() if census is None else census
    cur = current_recipe(census)
    out = {}
    for name, row in sorted(reg.items()):
        if not row.get("scorable", False):
            out[name] = ("scorable=False", ())
        elif row.get("dead_arms"):
            arms = tuple(sorted(row["dead_arms"]))
            out[name] = ("PARTIAL, dead arms %s" % (list(arms),), arms)
    for name, rec in sorted(census.items()):
        if cur and rec != cur and name not in out:
            out[name] = ("off-recipe (fp32=%s, grad_mode=%s)" % rec, ())
    for ds in REMOVED_DATASETS:
        out[ds] = ("removed dataset", ())
    return out


def _names(name, body):
    # !! MATCH ON A WORD BOUNDARY, NOT ON BACKTICKS (2026-09-10).
    # The first version required `name`, and this repo does not backtick
    # consistently: FRAMEWORK:6878's table header reads
    # "| iwc4 final, vs `tralo_null` |", so the OFF-RECIPE campaign its whole
    # table comes from was invisible while a passing mention of `iwc3` further
    # down was what flagged the entry. A campaign name is distinctive enough
    # that a word boundary is safe; a backtick is a formatting habit.
    return re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name),
                     body) is not None


def covered_by_ancestor(ents, i, name):
    """Did an ENCLOSING section already disclose this campaign?

    A disclosure belongs at the TOP of a long entry, not copied into each of
    its subsections -- 2(w) is 300 lines over 9 headings. Without inheritance
    the tool demands the same paragraph nine times, and a person who wrote it
    once still reads eight hits, which is how a report loses its reader.

    So a STRICT ancestor (a heading of LOWER level, i.e. higher in the tree)
    clears a child -- but ONLY for the name it actually names. A parent
    disclosing `iwc3` must not clear a child quoting `dom1`, and a SIBLING
    clears nothing at all.
    """
    lvl = ents[i][3]
    for j in range(i - 1, -1, -1):
        if ents[j][3] >= lvl:
            continue
        lvl = ents[j][3]                     # walk strictly up the tree
        if DISCLOSED.search(ents[j][2]) and _names(name, ents[j][2]):
            return True
    return False


def entries(path):
    """(start_line, heading, body, level) per entry, at h2/h3/h4 granularity."""
    lines = io.open(path, encoding="utf-8").read().splitlines()
    # Fence-aware: inside a fenced block a `#` comment is not a heading, and
    # CLAUDE.md's command blocks are full of them. `stale_figures` splits at
    # h2 only; this needs h3/h4 too, so the walk is done here.
    fenced = set()
    infence = False
    for i, l in enumerate(lines):
        if l.lstrip().startswith("```"):
            infence = not infence
            fenced.add(i)
        elif infence:
            fenced.add(i)
    def lvl(l):
        return len(l) - len(l.lstrip("#"))

    # A heading that WRAPS is ONE heading. FRAMEWORK writes long entry titles
    # as several consecutive `###` lines -- 9 of its 372 heading lines are
    # continuations -- and splitting on each one shatters the entry: the first
    # fragments get an empty body and the LAST continuation inherits the whole
    # thing. That is why 2(x)'s retraction was filed under "RIVALS ARE DEAD
    # ARMS ..." and its four subsections lost their parent, so nothing could
    # inherit its disclosure.
    starts, prev = [], None
    for i, l in enumerate(lines):
        if not HEAD.match(l) or i in fenced:
            continue
        if prev == i - 1 and lvl(lines[i - 1]) == lvl(l):
            prev = i
            continue                       # adjacent, same level: a wrap
        starts.append(i)
        prev = i
    if not starts:
        starts = [0]

    def head_text(a):
        out = [lines[a]]
        j = a + 1
        while (j < len(lines) and HEAD.match(lines[j]) and j not in fenced
               and lvl(lines[j]) == lvl(lines[a])):
            out.append(lines[j].lstrip("#").strip())
            j += 1
        return " ".join(out)
    bounds = starts + [len(lines)]
    return [(a + 1, head_text(a) if HEAD.match(lines[a]) else "(preamble)",
             "\n".join(lines[a:b]),
             len(lines[a]) - len(lines[a].lstrip("#"))
             if HEAD.match(lines[a]) else 1)
            for a, b in zip(bounds, bounds[1:])]


def scan(docs=DOCS, cond=None):
    """Entries that quote a figure and name condemned data without saying so."""
    cond = condemned() if cond is None else cond
    hits, clear, inherited = [], 0, 0
    for doc in docs:
        p = doc if os.path.isabs(doc) else os.path.join(REPO, doc)
        rel = os.path.relpath(p, REPO).replace("\\", "/")
        if any(rel.startswith(s) for s in SKIP_DOCS):
            continue
        if not os.path.exists(p):
            continue
        ents = entries(p)
        for i, (ln, head, body, _lvl) in enumerate(ents):
            figs = FIGURE.findall(body)
            if not figs:
                continue
            named = []
            for n, (why, need_arms) in sorted(cond.items()):
                if not _names(n, body):
                    continue
                # PARTIAL: fire only if a DEAD ARM is named too. See
                # `condemned`.
                if need_arms and not any(("`%s`" % a) in body
                                         for a in need_arms):
                    continue
                named.append(n)
            if not named:
                continue
            if DISCLOSED.search(body):
                clear += 1
                continue
            # A parent section that already disclosed this campaign covers its
            # subsections. Per NAME, so a parent disclosing one campaign does
            # not clear a child quoting a different one.
            left = [n for n in named if not covered_by_ancestor(ents, i, n)]
            if not left:
                inherited += 1
                continue
            named = left
            hits.append({"doc": rel, "line": ln, "head": head.strip("# ").strip(),
                         "names": named, "figures": len(figs),
                         "why": [cond[n][0] for n in named]})
    # Order by how much the number matters: more figures = a reader is more
    # likely to carry one out of the entry.
    hits.sort(key=lambda h: (-h["figures"], h["doc"], h["line"]))
    return hits, clear, inherited


def report(docs=DOCS, out=sys.stdout, limit=40):
    cond = condemned()
    hits, clear, inherited = scan(docs, cond)
    w = out.write
    w("STALE PROVENANCE -- entries quoting a figure beside condemned data\n")
    w("%d condemned names in play (%d dead/partial campaigns + %d off-recipe "
      "+ %d removed datasets)\n"
      % (len(cond),
         sum(1 for v, _ in cond.values()
             if "scorable" in v or "PARTIAL" in v),
         sum(1 for v, _ in cond.values() if "off-recipe" in v),
         len(REMOVED_DATASETS)))
    w("\n")
    w("!! THIS IS A QUEUE OF ENTRIES TO READ, NOT A DEFECT COUNT. Most entries\n")
    w("   naming a removed dataset are ABOUT it, where the figure is in\n")
    w("   context and correct. Attribution by proximity is irreducibly noisy\n")
    w("   -- measured at roughly 5 genuine / 5 misattributed / 2 ambiguous out\n")
    w("   of 12 when `stale_figures` was read by hand (2(z68)). Read the entry.\n")
    w("\n")
    w("%d entries DISCLOSE the condemnation and were cleared; %d more are\n"
      "subsections of one that does -- a disclosure belongs at the TOP of a\n"
      "long entry, not copied into all nine of its headings.\n"
      % (clear, inherited))
    w("%d do not, ordered by how many figures they carry:\n\n" % len(hits))
    for h in hits[:limit]:
        w("  %-20s :%-6d %2d fig  %s\n"
          % (h["doc"].split("/")[-1], h["line"], h["figures"],
             ", ".join("%s (%s)" % (n, r)
                       for n, r in zip(h["names"], h["why"]))[:78]))
        w("      %s\n" % h["head"][:96])
    if len(hits) > limit:
        w("\n  ... and %d more; raise --limit\n" % (len(hits) - limit))
    return hits


def self_test(out=sys.stdout):
    """Gated in BOTH directions, with the false-positive controls named."""
    fails = []

    def check(ok, label):
        out.write("  %s  %s\n" % ("PASS" if ok else "FAIL", label))
        if not ok:
            fails.append(label)

    # --- the authorities are actually read -------------------------------
    census = recipe_census()
    check(bool(census), "COVERAGE's recipe table parses at all")
    check(census.get("loose1") == (True, "clip"),
          "loose1 reads as the `clip` recipe from COVERAGE, not from a "
          "hardcoded list (2(z76))")
    # The FIRST and LAST name in a row are the two a backtick-strip eats.
    check(census.get("iwc4") == (True, "clip"),
          "the FIRST campaign in a COVERAGE row is read (`iwc4`)")
    check(census.get("vitu1") == (True, "clip"),
          "the LAST campaign in a COVERAGE row is read (`vitu1`)")
    check(census.get("xfam1") == (False, "normalize"),
          "a row holding a SINGLE campaign is read (`xfam1` is both first "
          "and last)")
    check(current_recipe(census) == (True, "normalize"),
          "the CURRENT recipe is derived as fp32+normalize, not asserted")

    cond = condemned()
    check("iwc3" in cond and "scorable=False" in cond["iwc3"][0],
          "iwc3 is condemned via quarantine.REGISTRY (2(z71))")
    check("loose1" in cond and "off-recipe" in cond["loose1"][0],
          "loose1 is condemned via the recipe census (2(z76))")
    check("dermmnist" in cond, "removed datasets are condemned (2(z30)(d))")
    check("dom1" in cond and "PARTIAL" in cond["dom1"][0],
          "a PARTIAL campaign is condemned too -- scorable=True with dead "
          "arms is not clean")

    # --- the detector fires ------------------------------------------------
    bad = "## E\n\nMeasured on `iwc3`: the delta is +0.0253 items.\n"
    good = ("## E\n\nMeasured on `iwc3` -- which is quarantined, "
            "scorable=False -- the delta is +0.0253 items.\n")
    nofig = "## E\n\nMeasured on `iwc3`, and it moved.\n"
    noname = "## E\n\nThe delta is +0.0253 items.\n"

    import tempfile
    tmp = tempfile.mkdtemp()

    def one(text):
        p = os.path.join(tmp, "t.md")
        io.open(p, "w", encoding="utf-8", newline="").write(text)
        return scan([p], cond)[0]

    check(len(one(bad)) == 1, "a bare figure beside a dead campaign FIRES")
    # NEGATIVE CONTROL 1: disclosure clears it, or the tool would force a
    # caveat onto entries that already carry one.
    check(len(one(good)) == 0,
          "NEGATIVE CONTROL: an entry that DISCLOSES the quarantine is cleared")
    # NEGATIVE CONTROL 2: naming dead data without quoting a number is not a
    # finding -- half this repo's prose names iwc3.
    check(len(one(nofig)) == 0,
          "NEGATIVE CONTROL: naming dead data with NO figure does not fire")
    # NEGATIVE CONTROL 3: a figure with no condemned name is not a finding.
    check(len(one(noname)) == 0,
          "NEGATIVE CONTROL: a figure with no condemned name does not fire")

    # NEGATIVE CONTROL 4: a `#` inside a fenced block is not a heading.
    # CLAUDE.md's command blocks are full of them and a naive split shatters
    # the file into fragments, which is how `stale_figures` once dropped it
    # from 13 figures to 4.
    fenced = ("## E\n\n```bash\n# not a heading: iwc3 gave +0.0253\n"
              "python -m scripts.x\n```\n\nBody with `iwc3` and +0.1234.\n")
    p = os.path.join(tmp, "f.md")
    io.open(p, "w", encoding="utf-8", newline="").write(fenced)
    check(len(entries(p)) == 1,
          "NEGATIVE CONTROL: a `#` inside a fence is not an entry boundary")

    # --- a WRAPPED heading is one heading ----------------------------------
    # FRAMEWORK writes long titles as consecutive `###` lines. Splitting on
    # each one gave 2(x) four fragments, three empty and the fourth carrying
    # the whole retraction under the title "RIVALS ARE DEAD ARMS ...".
    wrap = ("### 2(x) RETRACTED -- THIS SECTION WAS HEADED\n"
            "### **TraLO LEADS ALL FOUR RIVAL DUALS**\n\nBody, +0.1234.\n")
    q = os.path.join(tmp, "w.md")
    io.open(q, "w", encoding="utf-8", newline="").write(wrap)
    ents = entries(q)
    check(len(ents) == 1 and "RIVAL DUALS" in ents[0][1],
          "a heading WRAPPED over two `###` lines is ONE entry, titled with "
          "both")

    # NEGATIVE CONTROL 8: adjacent but a DIFFERENT level is a real parent and
    # child, not a wrap, even with no body between them.
    nest = "### P\n#### C\n\nBody, +0.1234.\n"
    io.open(q, "w", encoding="utf-8", newline="").write(nest)
    check(len(entries(q)) == 2,
          "NEGATIVE CONTROL: `###` then `####` is a parent and a child, not "
          "a wrapped heading")

    # NEGATIVE CONTROL 9: same level but NOT adjacent is two entries. An
    # empty-bodied section is legal and must not swallow the next title.
    apart = "### A\n\n### B\n\nBody, +0.1234.\n"
    io.open(q, "w", encoding="utf-8", newline="").write(apart)
    check(len(entries(q)) == 2,
          "NEGATIVE CONTROL: two same-level headings separated by a blank "
          "line stay two entries")

    # --- a disclosure INHERITS DOWN THE TREE, and only down ----------------
    # 2(w) is 300 lines over 9 headings. Demanding the same caveat in every
    # one of them is how a report gets ignored; inheriting it from a SIBLING
    # or from an unrelated parent is how a report goes quietly wrong.
    par = ("### P\n\nQuarantined, scorable=False: `iwc3`.\n\n"
           "#### C\n\nOn `iwc3` the delta is +0.0253 items.\n")
    check(len(one(par)) == 0,
          "a parent that DISCLOSES covers its subsection")
    check(scan([os.path.join(tmp, "t.md")], cond)[2] == 1,
          "and it is counted as INHERITED, not as a first-hand disclosure")

    # NEGATIVE CONTROL 5: the parent must disclose THIS campaign. A parent
    # caveating `loose1` says nothing about a child quoting `iwc3`.
    other = ("### P\n\nQuarantined, scorable=False: `loose1`.\n\n"
             "#### C\n\nOn `iwc3` the delta is +0.0253 items.\n")
    check(len(one(other)) == 1,
          "NEGATIVE CONTROL: a parent disclosing a DIFFERENT campaign does "
          "not cover the child")

    # NEGATIVE CONTROL 6: naming is not disclosing. The parent has to say the
    # data is condemned, not merely mention it.
    bare = ("### P\n\nThe `iwc3` campaign ran on one host.\n\n"
            "#### C\n\nOn `iwc3` the delta is +0.0253 items.\n")
    check(len(one(bare)) == 1,
          "NEGATIVE CONTROL: a parent that merely NAMES the campaign does "
          "not cover the child")

    # NEGATIVE CONTROL 7: only a STRICT ancestor covers. A preceding sibling
    # at the same depth is a different entry, and a reader landing on the
    # second one never sees the first.
    sib = ("### A\n\nQuarantined, scorable=False: `iwc3`, +0.1234.\n\n"
           "### B\n\nOn `iwc3` the delta is +0.0253 items.\n")
    check(len(one(sib)) == 1,
          "NEGATIVE CONTROL: a preceding SIBLING does not cover the next "
          "section")

    # --- it says what it is ------------------------------------------------
    buf = io.StringIO()
    report(out=buf, limit=1)
    txt = buf.getvalue()
    check("QUEUE OF ENTRIES TO READ, NOT A DEFECT COUNT" in txt,
          "the report refuses to present itself as a defect count (2(z68))")
    check("DISCLOSE" in txt,
          "the report prints the CLEARED count too, so a reader can see the "
          "denominator")

    out.write("\n%s\n" % ("SELF-TEST PASSED" if not fails
                          else "SELF-TEST FAILED: %d" % len(fails)))
    # Return an EXIT CODE, not a bool. `tests/test_baseline_fidelity.py`'s
    # sweep imports `self_test` and accepts only `0` or `None`; `True` is not
    # `0`, so a bool reads as a FAILING probe that prints SELF-TEST PASSED.
    # Caught by that gate on this file's first full-suite run.
    return 0 if not fails else 1


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--docs", nargs="*", default=list(DOCS),
                   help="documents to scan (default: the five live ones)")
    a.add_argument("--limit", type=int, default=40,
                   help="entries to print (default 40)")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    report(args.docs, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
