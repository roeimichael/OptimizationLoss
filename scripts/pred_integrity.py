"""IS THIS PREDICTIONS FILE INTACT? A torn CSV still parses.

🛑 THE DEFECT THIS EXISTS FOR (2026-09-05). Two dispatchers ran against one
`results/` tree over shared NFS, and two of them wrote the same run
directory. The result was not a crash and not an empty file -- it was a
`final_predictions.csv` with **six extra rows**, one of which was the torn tail
of another line (`0.00016164035,218`, a probability and a group id with no
label in front of them).

`pandas.read_csv` accepted it without complaint. The file had the right header,
the right columns and mostly the right rows. The ONLY reason it was ever
noticed is that the stray probability landed in the `True_Label` column and
forced the dtype to float64, which made `sklearn` raise five frames deep with
`Classification metrics can't handle a mix of continuous and multiclass
targets`. Had the torn fragment carried an integer instead, the file would have
scored silently with six phantom items in it.

So the check is not "does it parse". It is:

  * **ROW COUNT.** The test set is FIXED, so every run of a campaign must emit
    exactly the same number of prediction rows. This is the decisive test and
    it needs no model, no labels and no metric -- `wc -l` finds it. Measured:
    2944 rows in 111 clean runs, 2950 and 2958 in the two torn ones.
  * **LABEL DTYPE.** `True_Label` and `Predicted_Label` are class indices. A
    float column means something that is not a label got into one.

Both are cheap enough to run on every scoring pass, which is the point: the
integrity check that only runs when somebody suspects a problem is the one
that was not running when the problem happened.
"""

import argparse
import collections
import glob
import io
import json
import os
import sys

LABEL_COLS = ("True_Label", "Predicted_Label")
PRED_FILES = ("final_predictions.csv", "final_predictions_raw.csv")


def row_count(path):
    """Physical line count, without parsing. A torn file must not get a vote
    on how many rows it has."""
    n = 0
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return n


def label_dtype_ok(path):
    """(ok, message). Every value in a label column must be an integer.

    Read as TEXT and checked lexically rather than with pandas: pandas is what
    silently accepted the torn file, and `float64` is a SYMPTOM of the tear,
    not the tear itself. A lexical check sees the fragment directly.
    """
    with io.open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\n").rstrip("\r").split(",")
        idx = [(c, header.index(c)) for c in LABEL_COLS if c in header]
        if not idx:
            return False, "no label column in the header"
        for lineno, line in enumerate(fh, start=2):
            parts = line.rstrip("\n").rstrip("\r").split(",")
            if len(parts) != len(header):
                return False, ("line %d has %d fields, header has %d -- this "
                               "is a TORN line, not a type problem"
                               % (lineno, len(parts), len(header)))
            for name, i in idx:
                v = parts[i].strip()
                if not (v.lstrip("-").isdigit()):
                    return False, ("line %d: %s = %r is not an integer class "
                                   "index" % (lineno, name, v))
    return True, "ok"


def audit(roots, out=sys.stdout, deep=True):
    """Row-count and label-dtype audit over every prediction file under `roots`.

    Returns a list of (path, reason). Empty means intact.

    The row-count comparison is made WITHIN a campaign, because that is the
    scope in which the test set is guaranteed identical. Comparing across
    campaigns would false-positive on a different slice.
    """
    problems = []
    per_campaign = collections.defaultdict(lambda: collections.defaultdict(list))
    for root in roots:
        for name in PRED_FILES:
            pat = os.path.join(root, "**", name)
            for p in glob.glob(pat, recursive=True):
                camp = _campaign_of(p, root)
                per_campaign[(camp, name)][row_count(p)].append(p)

    for (camp, name), counts in sorted(per_campaign.items()):
        if len(counts) <= 1:
            continue
        modal = max(counts, key=lambda k: len(counts[k]))
        for n, paths in sorted(counts.items()):
            if n == modal:
                continue
            for p in paths:
                problems.append((p, "%d rows against the campaign's modal %d "
                                    "(%d of %d runs) -- the test set is fixed, "
                                    "so this file is TORN or double-written"
                                 % (n, modal, len(counts[modal]),
                                    sum(len(v) for v in counts.values()))))

    if deep:
        for (camp, name), counts in sorted(per_campaign.items()):
            for n, paths in counts.items():
                for p in paths:
                    ok, why = label_dtype_ok(p)
                    if not ok:
                        problems.append((p, why))

    if problems:
        print("!! %d PREDICTION FILE(S) FAILED THE INTEGRITY CHECK" %
              len(problems), file=out)
        for p, why in problems:
            print("   %s" % p, file=out)
            print("       %s" % why, file=out)
        print("   A torn file PARSES. Scoring one puts phantom rows into a "
              "metric with no error.", file=out)
        print("", file=out)
    return problems


def _run_dir_of(path):
    """The run directory holding `path`, whether it is the dir or a file in it."""
    return path if os.path.isdir(path) else os.path.dirname(path)


def completed_only(paths, out=sys.stdout, label="run"):
    """Drop every path whose run is not `status: completed`.

    THE DEFECT THIS EXISTS FOR (2026-09-05). Resetting a run to `pending`
    does NOT remove its old `final_predictions.csv`. The file stays on disk,
    intact and parseable, describing a model that has since been discarded --
    the wrong-host recovery left four of them. `full_panel` and
    `sensitivity_screen` already refused them; `deployed_h2h`, `score_scan` and
    `paired_noise` globbed for the CSV and never looked at the status, so a
    superseded model sat in the arm-vs-arm table beside live ones.

    It is the `pred_integrity` question in another costume: the file parses, and
    parsing was never the test. Here the test is whether the run that wrote it
    still exists.

    Fails CLOSED. A run whose `config.json` is missing or unreadable is DROPPED
    and named, because that is exactly what a half-written run looks like from
    the outside. Every drop is printed -- a filter that removes data silently is
    the failure mode this module was written against.
    """
    keep, dropped, unreadable = [], collections.defaultdict(list), []
    for p in paths:
        cfg = os.path.join(_run_dir_of(p), "config.json")
        try:
            with io.open(cfg, "r", encoding="utf-8") as fh:
                status = json.load(fh).get("status")
        except Exception:
            unreadable.append(p)
            continue
        if status == "completed":
            keep.append(p)
        else:
            dropped[str(status)].append(p)

    if dropped or unreadable:
        n = sum(len(v) for v in dropped.values()) + len(unreadable)
        print("   dropping %d %s(s) that are not `status: completed` "
              "(a reset run keeps its old predictions file)"
              % (n, label), file=out)
        for status, ps in sorted(dropped.items()):
            print("     status=%-10s %d" % (status, len(ps)), file=out)
            for p in ps[:4]:
                print("        %s" % _run_dir_of(p), file=out)
        if unreadable:
            print("     config.json UNREADABLE %d -- dropped, because an "
                  "unreadable config is what a half-written run looks like"
                  % len(unreadable), file=out)
            for p in unreadable[:4]:
                print("        %s" % _run_dir_of(p), file=out)
    return keep


def _campaign_of(path, root):
    q = os.path.normpath(path).replace(os.sep, "/")
    r = os.path.normpath(root).replace(os.sep, "/").rstrip("/")
    tail = q[len(r):].lstrip("/") if q.startswith(r) else q
    return os.path.basename(r) or tail.split("/")[0]


def self_test():
    """Gate it in BOTH directions: intact files must PASS, torn ones must FAIL."""
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp(prefix="pred_integrity_")
    checks = []
    try:
        head = ("True_Label,Predicted_Label,Correct,Prob_Class_0,"
                "Prob_Class_1,Group_ID")
        good = [head] + ["%d,%d,1,0.9,0.1,%d" % (i % 2, i % 2, i % 3)
                         for i in range(20)]

        def write(camp, arm, seed, lines):
            d = os.path.join(tmp, camp, "M", "ds", "L80_G95", arm,
                             "seed_%d" % seed)
            os.makedirs(d, exist_ok=True)
            for name in PRED_FILES:
                io.open(os.path.join(d, name), "w", encoding="utf-8").write(
                    "\n".join(lines) + "\n")
            return d

        for arm in ("clip", "tralo"):
            for seed in (1, 2):
                write("camp", arm, seed, good)
        probs = audit([os.path.join(tmp, "camp")], out=io.StringIO())
        checks.append(("an INTACT campaign reports no problem", not probs))

        # NEGATIVE CONTROL 1: a torn tail, exactly the real failure. It has
        # FEWER fields, so it is caught lexically as well as by row count.
        torn = list(good) + ["0.00016164035,218"]
        write("camp", "tralo", 3, torn)
        probs = audit([os.path.join(tmp, "camp")], out=io.StringIO())
        hit = [p for p, _w in probs if "seed_3" in p]
        checks.append(("a TORN tail row is caught", bool(hit)))

        # NEGATIVE CONTROL 2: row count alone, with every line well-formed.
        # This is the case a field-count check would MISS.
        dup = list(good) + ["1,1,1,0.9,0.1,2"]
        write("camp2", "clip", 1, good)
        write("camp2", "clip", 2, good)
        write("camp2", "tralo", 1, dup)
        probs = audit([os.path.join(tmp, "camp2")], out=io.StringIO(), deep=False)
        checks.append(("a DUPLICATED but well-formed row is caught by the "
                       "row count alone", any("tralo" in p for p, _w in probs)))

        # NEGATIVE CONTROL 3: a float label with correct field count -- the
        # case that made sklearn raise five frames deep.
        flt = list(good)
        flt[1] = "0.5,1,1,0.9,0.1,0"
        write("camp3", "clip", 1, flt)
        write("camp3", "clip", 2, good)
        probs = audit([os.path.join(tmp, "camp3")], out=io.StringIO())
        checks.append(("a FLOAT in a label column is caught", bool(probs)))

        # POSITIVE CONTROL: a campaign whose runs legitimately differ in row
        # count from ANOTHER campaign must not be flagged. The test set is
        # fixed within a campaign, not across them.
        for arm in ("clip", "tralo"):
            write("other", arm, 1, good[:12])
        probs = audit([os.path.join(tmp, "camp"), os.path.join(tmp, "other")],
                      out=io.StringIO(), deep=False)
        checks.append(("a DIFFERENT campaign with its own row count is NOT "
                       "flagged", not any("other" in p for p, _w in probs)))
        # ---- completed_only: a reset run keeps its predictions file --------
        def run_with_status(camp, arm, seed, status, lines=None):
            d = write(camp, arm, seed, lines or good)
            io.open(os.path.join(d, "config.json"), "w",
                    encoding="utf-8").write('{"status": "%s"}' % status)
            return os.path.join(d, PRED_FILES[0])

        done = run_with_status("st", "clip", 1, "completed")
        pend = run_with_status("st", "tralo", 1, "pending")
        runn = run_with_status("st", "tralo", 2, "running")
        naked = write("st", "focal_clip", 1, good)      # no config.json at all
        naked = os.path.join(naked, PRED_FILES[0])

        kept = completed_only([done, pend, runn, naked], out=io.StringIO())
        checks.append(("a COMPLETED run is kept", done in kept))
        # NEGATIVE CONTROLS: each of these survives if the filter is removed.
        checks.append(("a run reset to PENDING is dropped even though its "
                       "predictions file is intact", pend not in kept))
        checks.append(("a RUNNING run is dropped", runn not in kept))
        checks.append(("a run with NO config.json is dropped -- fails CLOSED",
                       naked not in kept))
        checks.append(("nothing but the completed run survives",
                       kept == [done]))
        # The drops must be ANNOUNCED. A filter that removes data silently is
        # the exact failure this module was written against.
        buf = io.StringIO()
        completed_only([done, pend, runn, naked], out=buf, label="widget")
        said = buf.getvalue()
        checks.append(("every drop is REPORTED, with the label and the status",
                       "widget" in said and "pending" in said
                       and "running" in said and "UNREADABLE" in said))
        # POSITIVE CONTROL: an all-completed list must print NOTHING and keep
        # everything, or the filter is just refusing at random.
        quiet = io.StringIO()
        allc = completed_only([done], out=quiet)
        checks.append(("an all-completed list is kept in full and prints "
                       "nothing", allc == [done] and quiet.getvalue() == ""))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("")
    for label, ok in checks:
        print("  %-72s %s" % (label[:72], "PASS" if ok else "FAIL"))
    bad = [c for c, ok in checks if not ok]
    print("")
    print("ALL PASS" if not bad else "FAILED: %d" % len(bad))
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("roots", nargs="*", help="campaign roots to audit")
    a.add_argument("--self-test", action="store_true")
    a.add_argument("--shallow", action="store_true",
                   help="row counts only; skip the per-line lexical check")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.roots:
        a.error("give at least one campaign root (or --self-test)")
    problems = audit(args.roots, deep=not args.shallow)
    if not problems:
        print("all prediction files intact")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
