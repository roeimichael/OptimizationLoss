"""WHICH CAMPAIGNS DOES THE DOC NAME THAT NO AUTHORITY RECORDS?

The third staleness axis, and it is not either of the other two:

  * `stale_figures`      -- has the SCORER moved since the figure was written?
  * `stale_provenance`   -- has the DATA been condemned since?
  * this one             -- does the campaign have a STATE at all?

WHY IT EXISTS. `price1` is named four times in `docs/FRAMEWORK.md`, twice in
the past tense ("pre-registered in `protocol.yml` before `price1` launched"),
once as the thing that is re-pricing 2(z29) -- the entry that currently reads
UNPRICED-NULL and holds the coin result the whole direction question rests on.
Task #78, "Launch price1", is marked COMPLETED. And `price1` appears in NO
run-state record anywhere: not in MISSION's LIVE table, not in its LANDED
table, not in `quarantine.REGISTRY`, not in COVERAGE section 0's census, not
in CLAUDE.md's archive list. So a campaign was launched, a ledger entry was
made to depend on it, and nothing says whether it ran, died, or landed.

THAT IS 2(z72)'S DEFECT IN ITS COMPLEMENTARY FORM. 2(z72) was a campaign that
had LANDED and was still announced as running -- a stale state. This is a
campaign with NO state, which is worse in one specific way: a stale state is
visible to anybody who re-reads the block, and an absent one is visible to
nobody, because there is no block to re-read. A run-state table can only be
audited for the rows it contains.

THE AUTHORITIES ARE READ, NOT RESTATED. Four of them, and each is the place a
different kind of campaign is supposed to be recorded:

  * `scripts.quarantine.REGISTRY`         -- dead and PARTIAL campaigns
  * `docs/COVERAGE.md` section 0's table  -- which recipe a campaign ran
  * `docs/MISSION.md` 0-RUNNING's tables  -- LIVE and LANDED
  * `CLAUDE.md`'s archive + holds lines   -- moved out of `results/`

A tool that hardcoded any of them would drift from the thing it is auditing.

!! IT IS A QUEUE, NOT A DEFECT COUNT, for the reason 2(z68) measured: a
campaign named only in a `gen_campaign` command has no state because it has
never run, and that is correct, not a defect. The tool cannot tell that from
an orphan without guessing, so it does not guess -- it prints the mention
lines and flags the ones carrying a PAST-EXECUTION verb, and a person reads.
The verb list is a heuristic and is named as one.

CALIBRATED BY HAND 2026-09-10 ON THE FIRST RUN, because an uncalibrated report
is a rumour. See `docs/FRAMEWORK.md` 2(z83) for the reading.

    python -m scripts.campaign_state
    python -m scripts.campaign_state --docs docs/FRAMEWORK.md
    python -m scripts.campaign_state --all
    python -m scripts.campaign_state --self-test
"""
import argparse
import io
import os
import re
import sys

from scripts import quarantine
from scripts.stale_provenance import recipe_census

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DOCS = ("CLAUDE.md", "docs/FRAMEWORK.md", "docs/MISSION.md",
        "docs/COVERAGE.md", "docs/PLAYBOOK.md")

# Two harvest channels. A campaign is named as a PATH when somebody pastes a
# command, and in BACKTICKS when somebody writes about it -- and `price1` is
# only ever the second, which is why a path-only harvest missed it entirely.
PATH = re.compile(r"results/([a-z][a-z0-9_]*)")
TICK = re.compile(r"`([^`\n]{2,40})`")

# Campaign names here are lowercase and carry a digit: dom1, iwc3, bcn1mn3,
# loosevit1, itemscale2. That shape is the filter; everything it lets through
# that is NOT a campaign is listed below WITH ITS REASON, so the list cannot
# quietly become a place a real campaign hides.
SHAPE = re.compile(r"^[a-z][a-z0-9]*[0-9][a-z0-9_]*$")

NOT_A_CAMPAIGN = {
    "dsisco01": "a HOST",
    "dsisco02": "a HOST",
    "fp32": "a precision, not a campaign",
    "float16": "a dtype",
    "bfloat16": "a dtype",
    "f1_macro": "a metric column",
    "q95": "a quantile",
    "clip03": "an arm BLOCK name (constraint_grad_clip 0.3)",
    "clip30": "an arm BLOCK name (constraint_grad_clip 3.0)",
    "reseed2": "part of the arm name tralo_reseed2",
    "steps5": "a step count",
    "seed1_bounded": "a run directory",
    "g5_hinge_oct": "a gate-bucket test name",
    # !! `seed58a` WAS ON THIS LIST AS "a run directory" AND IT IS A REAL
    # CAMPAIGN (2026-09-10): 40 completed runs in optloss-domb/results, and
    # `paper_rows.MEASURED_UNITS` maps it to unit B1. An exclusion list is a
    # place a real campaign can hide, which is why every entry carries its
    # reason -- and this one's reason was checkable and wrong. Removed.
    "ena24": "a candidate DATASET slice",
    "ganchev2009posterior": "a bibtex key",
}

# Any lowercase-hex token of 6+ chars is a git hash, not a campaign. Listed as
# a rule rather than one-by-one because new commits are quoted constantly.
HASH = re.compile(r"^[0-9a-f]{6,}$")

# PAST EXECUTION, in the forms this project actually writes it. A HEURISTIC,
# and deliberately narrow: `seeds` is excluded because a pre-registration
# names its seed count, and `completed` is excluded because 0-PERM's honest
# "has ZERO completed runs anywhere" would read as evidence of a run.
EXECUTED = re.compile(
    r"launch|landed|\bran\b|running|relaunch|discard|\bkill|"
    r"\d+\s+runs|quarantin|was staged|were staged", re.I)


def _ascii(s):
    """Echoed doc lines carry emoji; the Windows console is cp1252."""
    return s.encode("ascii", "replace").decode("ascii")


def _tokens(text):
    """Every campaign-shaped token in `text`, both channels."""
    out = set()
    for m in PATH.finditer(text):
        out.add(m.group(1))
    for m in TICK.finditer(text):
        s = m.group(1).strip()
        if SHAPE.match(s):
            out.add(s)
    return out


def harvest(docs=DOCS, root=None):
    """{name: [(doc, lineno, line), ...]} for every campaign-shaped name."""
    root = root or REPO
    out = {}
    for rel in docs:
        path = os.path.join(root, rel)
        try:
            text = io.open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for name in _tokens(line):
                if name in NOT_A_CAMPAIGN or HASH.match(name):
                    continue
                out.setdefault(name, []).append((rel, i, line.strip()))
    return out


def mission_states(path=None):
    """{name: 'MISSION 0-RUNNING'} from the run-state TABLES AND THE CENSUS BLOCK.

    Scoped to the 0-RUNNING section, so a campaign merely discussed elsewhere in
    MISSION does not read as having a state.

    🛑 IT READS THE FENCED CENSUS TOO, AND THAT WAS FOUND BY THIS TOOL'S OWN
    SELF-TEST (2026-09-10). 0-RUNNING was rebuilt from a verified 26-campaign
    census, and a census of that size is a code block rather than a table --
    at which point `fmow1` and `price2` had a checked state written down in
    plain sight and STILL read as orphans, because the parser only knew about
    `|` rows. A run-state authority that a formatting choice can switch off is
    not an authority. The liveness check went red the same minute, which is the
    whole reason the tool ships one. Same shape as 2(z85): the gate answers
    only for the form it was given.

    The census block is `NAME  n/m` pairs, so a bare campaign-shaped token
    followed by a slash-count is what counts -- the same shape the block is
    written in, so a name without a run count does not silently acquire a
    state.
    """
    path = path or os.path.join(REPO, "docs", "MISSION.md")
    try:
        lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return {}
    out, inside, fenced = {}, False, False
    for line in lines:
        if line.startswith("## "):
            inside, fenced = "0-RUNNING" in line, False
            continue
        if not inside:
            continue
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            for name in re.findall(r"\b([a-z][a-z0-9_]*[0-9][a-z0-9_]*)\s+\d+/\d+",
                                   line):
                out[name] = "MISSION 0-RUNNING census"
            continue
        if not line.lstrip().startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3:
            continue
        for name in re.findall(r"`([a-z][a-z0-9_]*)`", cells[1]):
            out[name] = "MISSION 0-RUNNING"
    return out


# The archive paragraph is prose, so it is anchored -- and the anchor is a
# GATED assumption: `test_the_claude_md_anchors_still_match` goes red if the
# wording moves, rather than the tool silently reporting 18 fresh orphans.
CLAUDE_ANCHORS = ("configs are archived", "holds `dom1`")


def claude_names(path=None):
    """{name: 'CLAUDE.md archive/holds'} from the two campaign paragraphs."""
    path = path or os.path.join(REPO, "CLAUDE.md")
    try:
        lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return {}
    out, take = {}, False
    for line in lines:
        if any(a in line for a in CLAUDE_ANCHORS):
            take = True
        elif take and (not line.strip() or line.lstrip().startswith(("* ", "#"))):
            # !! CONSUME TO THE END OF THE BULLET, NOT A FIXED WINDOW. A
            # 3-line window ended one line short of `vitdom2_cnn` and
            # `vitdom2_vit`, so two ARCHIVED campaigns read as unrecorded.
            take = False
        if not take:
            continue
        for name in re.findall(r"`([a-z][a-z0-9_-]*)`", line):
            # `iwc1-4` is a RANGE and is the only one written that way.
            m = re.match(r"^([a-z]+)(\d)-(\d)$", name)
            if m:
                for k in range(int(m.group(2)), int(m.group(3)) + 1):
                    out["%s%d" % (m.group(1), k)] = "CLAUDE.md archive/holds"
            elif SHAPE.match(name):
                out[name] = "CLAUDE.md archive/holds"
    return out


def authorities():
    """{name: [authority, ...]} -- every place a campaign state can be recorded."""
    out = {}
    for name in quarantine.REGISTRY:
        # Markers carry a suffix, e.g. uniform1_VOID_dose3.4pct_2026-08-25.
        out.setdefault(name.split("_dose")[0], []).append("quarantine.REGISTRY")
    for name in recipe_census():
        out.setdefault(name, []).append("COVERAGE census")
    for name, where in mission_states().items():
        out.setdefault(name, []).append(where)
    for name, where in claude_names().items():
        out.setdefault(name, []).append(where)
    return out


def scan(docs=DOCS, root=None):
    """(recorded, unrecorded) -- unrecorded is [(name, n_mentions, executed, hits)]."""
    seen = harvest(docs, root=root)
    known = authorities()
    recorded, unrecorded = {}, []
    for name, hits in sorted(seen.items()):
        if name in known:
            recorded[name] = known[name]
            continue
        executed = [h for h in hits if EXECUTED.search(h[2])]
        unrecorded.append((name, len(hits), executed, hits))
    unrecorded.sort(key=lambda r: (-len(r[2]), -r[1], r[0]))
    return recorded, unrecorded


def report(docs=DOCS, out=sys.stdout, limit=12, show_all=False):
    recorded, unrecorded = scan(docs)
    out.write("CAMPAIGN STATE AUDIT\n")
    out.write("  %d campaign-shaped names in the docs; %d recorded, "
              "%d with NO recorded state.\n" % (len(recorded) + len(unrecorded),
                                                len(recorded), len(unrecorded)))
    orphans = [r for r in unrecorded if r[2]]
    out.write("  %d of those carry a PAST-EXECUTION verb -- read those first.\n"
              % len(orphans))
    out.write("\n  A QUEUE, NEVER A DEFECT COUNT. A campaign named only in a\n"
              "  gen_campaign command has no state because it has not run, and\n"
              "  that is correct. The verb flag is a heuristic; read the line.\n\n")
    shown = unrecorded if show_all else unrecorded[:limit]
    for name, n, executed, hits in shown:
        tag = "ORPHAN? " if executed else "proposed?"
        out.write("%s %-14s %2d mentions, %d with an execution verb\n"
                  % (tag, name, n, len(executed)))
        for rel, i, line in (executed or hits)[:3]:
            out.write("      %s:%d  %s\n" % (rel, i, _ascii(line)[:110]))
        out.write("\n")
    if not show_all and len(unrecorded) > limit:
        out.write("  ... %d more, use --all\n" % (len(unrecorded) - limit))
    out.write("\n  WHERE A STATE BELONGS: dead or PARTIAL -> quarantine.REGISTRY;\n"
              "  in flight or landed -> MISSION 0-RUNNING; part of the corpus ->\n"
              "  COVERAGE section 0; moved off disk -> CLAUDE.md's archive list.\n")
    return unrecorded


def self_test(out=sys.stdout):
    import tempfile
    checks, fails = 0, []

    def ck(cond, label):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(label)

    # --- harvesting, both channels -------------------------------------
    ck(_tokens("run `python -m x --root results/dom1`") == {"dom1"},
       "path channel finds dom1")
    ck("price1" in _tokens("pre-registered before `price1` launched"),
       "TICK CHANNEL FINDS A NAME THAT IS NEVER A PATH -- the price1 case")
    # NEGATIVE CONTROLS: the shape filter must not swallow non-campaigns.
    ck(_tokens("on `MobileNetV2` at `L80_G95`") == set(),
       "NC: uppercase tokens are not campaigns")
    ck(_tokens("the `constraint_fp32` flag") == set(),
       "NC: a name with no digit is not a campaign")
    ck("dsisco01" not in harvest.__doc__ or True, "doc smoke")

    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "docs"))

    def w(rel, text):
        with io.open(os.path.join(root, rel), "w", encoding="utf-8") as fh:
            fh.write(text)

    w("docs/FRAMEWORK.md",
      "# f\n\nthe `zzz9` campaign, pre-registered before `zzz9` launched\n"
      "and `qqq9` is generated by `--root results/qqq9` and has not run\n"
      "and `dsisco01` is a host and `abc123def` is a hash\n")
    got = harvest(("docs/FRAMEWORK.md",), root=root)
    ck(set(got) == {"zzz9", "qqq9"},
       "NC: the exclusion list and the hash rule both fire (%s)" % sorted(got))
    # `.get`, not `[...]`: mutation M2 (killing the TICK channel) made this
    # KeyError five lines after the check that should have reported it, so a
    # real defect surfaced as a traceback instead of a named failure -- the
    # exact shape `pred_integrity` and `cell_table` exist for. (2026-09-10)
    z = got.get("zzz9", [])
    ck(len(z) == 1 and z[0][1] == 3, "line numbers are kept (%s)" % z)

    # --- the verb heuristic --------------------------------------------
    ck(bool(EXECUTED.search("before `price1` launched")), "verb: launched")
    ck(bool(EXECUTED.search("`itemscale1` landed, 192 runs")), "verb: landed")
    # NEGATIVE CONTROLS: a pre-registration must NOT read as an execution.
    ck(not EXECUTED.search("x 4 seeds, `tralo` the only arm varying"),
       "NC: a seed count in a proposal is not an execution")
    ck(not EXECUTED.search("has ZERO completed runs anywhere"),
       "NC: 'ZERO completed runs' is not evidence of a run")

    # --- the authorities are READ, not restated ------------------------
    auth = authorities()
    ck("dom1" in auth, "quarantine/census authority loads")
    ck(any("quarantine" in a for a in auth.get("iwc3", [])),
       "a quarantined campaign is RECORDED")
    # !! ASK claude_names DIRECTLY, NOT THE MERGED authorities(). `iwc4` is
    # ALSO in the COVERAGE census, so a merged check passes even with the
    # range expansion deleted -- true for the wrong reason. Mutation-tested:
    # this is the only form of the check that goes red. (2026-09-10)
    cn = claude_names()
    ck("iwc4" in cn and "iwc1" in cn,
       "the iwc1-4 RANGE in CLAUDE.md expands (%s)" % sorted(cn)[:6])
    ck("bcn1vit" in auth and "MISSION 0-RUNNING" in auth["bcn1vit"],
       "MISSION's LIVE table is an authority")
    ck("itemscale1" in auth, "MISSION's LANDED table is an authority")

    # ROT CHECK. The CLAUDE.md parse is anchored on prose, so the anchors
    # themselves are gated -- a reworded paragraph must turn this red rather
    # than silently reporting 18 fresh orphans.
    txt = io.open(os.path.join(REPO, "CLAUDE.md"), encoding="utf-8").read()
    for a in CLAUDE_ANCHORS:
        ck(a in txt, "CLAUDE.md anchor still present: %r" % a)

    # --- end to end ----------------------------------------------------
    rec, unrec = scan(("docs/FRAMEWORK.md",), root=root)
    names = [r[0] for r in unrec]
    ck(names[:1] == ["zzz9"],
       "the one with an execution verb sorts FIRST (%s)" % names)
    ck(len(unrec) == 2 and unrec[0][2] and not unrec[1][2],
       "only the executed one is flagged (%s)" % names)
    # NEGATIVE CONTROL: a campaign the authorities DO name must not appear.
    w("docs/FRAMEWORK.md", "`dom1` ran 384 runs and `zzz9` launched\n")
    rec2, unrec2 = scan(("docs/FRAMEWORK.md",), root=root)
    ck("dom1" in rec2 and [r[0] for r in unrec2] == ["zzz9"],
       "NC: a RECORDED campaign is never reported (%s)" % [r[0] for r in unrec2])

    # --- the live tree -------------------------------------------------
    # The first run found 18 unrecorded, 10 with an execution verb, headed by
    # `price1`. All 18 were given a row in MISSION 0-RUNNING's campaign-state
    # ledger on 2026-09-10, so the live tree must now be clean -- and this is
    # the check that keeps it clean rather than a note saying it once was.
    # !! IT HAS ALREADY EARNED ITS KEEP. Rebuilding 0-RUNNING from a verified
    # 26-campaign census put that census in a FENCED BLOCK, and `fmow1` and
    # `price2` -- both with a checked state written in plain sight -- went
    # straight back to reading as orphans, because `mission_states` only knew
    # about `|` table rows. This check went red the same minute. A run-state
    # authority a formatting choice can switch off is not an authority.
    rec_live, live = scan()
    with_verb = [r[0] for r in live if r[2]]
    ck(not with_verb,
       "NO campaign carries an execution verb and no state (%s)" % with_verb)
    ck("price1" in rec_live and any("MISSION" in a for a in rec_live["price1"]),
       "price1 resolves to the MISSION ledger, which is where its UNKNOWN "
       "state is recorded (%s)" % rec_live.get("price1"))

    for f in fails:
        out.write("FAIL: %s\n" % f)
    out.write("%d/%d checks passed\n" % (checks - len(fails), checks))
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--docs", nargs="*", default=list(DOCS))
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--all", action="store_true",
                    help="print every unrecorded campaign, not just --limit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    report(tuple(args.docs), limit=args.limit, show_all=args.all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
