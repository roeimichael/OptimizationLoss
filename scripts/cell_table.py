"""Per-CELL table of every metric, averaged over SEED and nothing else.

`full_panel` answers "is arm A better than control C", which is the right
question for a verdict and the wrong one for a survey: it prints contrasts, so
the absolute level an arm actually reached is nowhere in its output. This emits
the levels.

THE CELL IS (campaign, dataset, model, cap, arm) AND THE ONLY AXIS COLLAPSED IS
`seed`. That is this project's most-repeated analysis error -- pooling across
cap levels, backbones or datasets has retracted a claim three times -- so the
key is asserted at runtime here, not left to a comment: `--self-test` builds a
frame whose cells differ ONLY by backbone and requires that they stay separate.

Columns per metric: mean over seeds, the within-cell sd, and n_seeds. The sd is
not decoration. The whole gap from `clip` to a PERFECT allocator on iwildcam is
1.9-9.9 items and the paired seed sd is worth ~2.7, so a mean without its sd is
unreadable -- and a cell with n_seeds < 2 has no sd at all and must be shown as
blank rather than 0.

Also carried, because they gate whether the row may be read at all:
  dose        constraint steps applied / attempted (an arm at 3.4% of its dose
              still writes `status: completed` and looks healthy from every
              other angle)
  n_md5       distinct raw-prediction md5s in the cell. 1 across seeds means
              the seeds are not varying; compare ACROSS arms to catch an inert
              flag, which is this project's most frequent failure mode.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.full_panel import panel  # noqa: E402
from scripts import quarantine  # noqa: E402

# The axis that may be collapsed. Everything else is part of the cell identity.
SEED_AXIS = "seed"
CELL_KEY = ["campaign", "dataset", "model", "cap", "arm"]

METRICS = ["AP", "AUROC", "ccF1", "ccP", "ccR", "macroF1", "macroP", "macroR",
           "uncF1", "acc", "ECE", "Brier", "NLL", "ConfGap"]


def collect(roots, dead=()):
    """One row per COMPLETED run, with the campaign name attached.

    `dead` is the arm set a PARTIAL quarantine marker disqualifies. It is
    a PARAMETER rather than a global because the gate runs in `main` and
    the enumeration runs here: reading it off the enclosing scope was a
    NameError waiting for the first partially-quarantined campaign.
    """
    rows = []
    skipped = collections.Counter()
    first_error = {}
    per_campaign = collections.Counter()
    for camp in roots:
        name = quarantine.campaign_name(camp)
        for p in quarantine.drop_dead_runs(
                glob.glob(camp + "/**/config.json", recursive=True),
                dead, label="config"):
            try:
                cfg = json.load(open(p))
            except Exception:
                continue
            if cfg.get("status") != "completed":
                skipped[cfg.get("status", "?")] += 1
                continue
            try:
                r = panel(os.path.dirname(p), cfg)
            except Exception as exc:
                # The TYPE alone is not diagnosable. `taskwin2` lost all 48 of
                # its runs to a bare `panel raised: TypeError 48` line, which
                # says nothing about which run or why, and reads like routine
                # attrition next to `skipped pending 12`.
                key = "panel raised: %s: %s" % (type(exc).__name__, exc)
                skipped[key[:120]] += 1
                first_error.setdefault(key[:120], p)
                continue
            if not r:
                skipped["unscorable"] += 1
                continue
            res = cfg.get("results") or {}
            r["campaign"] = name
            r["steps_applied"] = res.get("constraint_steps_applied")
            r["steps_attempted"] = res.get("constraint_steps_attempted")
            rows.append(r)
            per_campaign[name] += 1

    # 🛑 A CAMPAIGN THAT CONTRIBUTES NOTHING IS NOT "SOME RUNS SKIPPED". It is
    # a campaign silently absent from every table built off this CSV, and it
    # reads downstream as an absence of evidence rather than a failure to read.
    for camp in roots:
        nm = quarantine.campaign_name(camp)
        if not per_campaign.get(nm):
            print("!! %s CONTRIBUTED ZERO ROWS -- it is absent from this table "
                  "entirely, which is" % nm)
            print("   NOT the same as having nothing to say. Reasons counted "
                  "below; first example path")
            print("   for each is printed so it can be reproduced.")
    for key, path in sorted(first_error.items()):
        print("   %s" % key)
        print("     first at: %s" % path)
    return pd.DataFrame(rows), skipped


def cells(df):
    """Collapse SEED, and only seed."""
    for k in CELL_KEY:
        if k not in df.columns:
            raise SystemExit("no %r column -- refusing to aggregate without "
                             "the full cell key %s" % (k, CELL_KEY))
    out = []
    for key, g in df.groupby(CELL_KEY, dropna=False):
        row = dict(zip(CELL_KEY, key))
        n = len(g)
        row["n_seeds"] = n
        row["n_md5"] = g["raw_md5"].nunique() if "raw_md5" in g else np.nan
        app, att = g.get("steps_applied"), g.get("steps_attempted")
        if app is not None and att is not None and att.notna().any():
            row["dose"] = "%d/%d" % (np.nansum(app), np.nansum(att))
        else:
            row["dose"] = ""
        # the dF1 -> items scale, so a delta can be read in items later
        if "items_per_001" in g:
            row["items_per_001"] = float(g["items_per_001"].mean())
        for m in METRICS:
            if m not in g:
                continue
            v = g[m].astype(float)
            row[m] = float(v.mean())
            # n_seeds < 2 has NO sd. Printing 0.0 there would read as a
            # perfectly reproducible cell, which is the opposite of the truth.
            row[m + "_sd"] = float(v.std(ddof=1)) if n > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out).sort_values(CELL_KEY)


def annotate_task(c):
    """Add a `task` column: does each cell's cap pose a QUESTION at all?

    FRAMEWORK 2(z16)/2(z17)/2(z21). Which method looks best CHANGES with the
    cell selection -- `alm` is the best arm on ccF1 in `equaldose1`'s non-task
    cells and the second worst in its task cells -- so a survey that cannot
    tell the two apart invites exactly the pooling this table exists to
    prevent.

    THE THREE NEGATIVES STAY DISTINCT. `non_task` is a statement about the
    experiment; `no_window` means that (dataset, backbone) was never measured;
    `no_data` means the slice is not on this machine. Collapsing them would let
    an unmeasured backbone read as a known non-task.
    """
    try:
        from configs.gen_campaign import load_protocol
        from configs.task_cells import classify, load_windows
        P, TW = load_protocol(), load_windows()
    except Exception as e:
        c["task"] = "unavailable"
        print("  !! task-window column unavailable (%s). The rows are still "
              "correct; they just cannot say which cells pose a question."
              % type(e).__name__)
        return c
    cache = {}
    out = []
    for ds, model, cap in zip(c["dataset"], c["model"], c["cap"]):
        key = (ds, model, cap)
        if key not in cache:
            try:
                cache[key] = classify(P, TW, ds, model, cap)["status"]
            except SystemExit:
                cache[key] = "no_data"
            except Exception:
                cache[key] = "no_window"
        out.append(cache[key])
    c["task"] = out
    return c


def self_test(out=sys.stdout):
    """The gate must REFUSE to pool anything but seed, and must blank a lone sd."""
    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and cond
        print("  %-58s %s" % (name, "PASS" if cond else "FAIL"), file=out)

    # cells identical in every way EXCEPT the backbone must stay separate
    base = dict(campaign="c", dataset="iwildcam", cap="L30_G50", arm="tralo",
                raw_md5="x", items_per_001=1.0, steps_applied=29,
                steps_attempted=29)
    df = pd.DataFrame([
        dict(base, model="ViTB16", seed=1, AP=0.10),
        dict(base, model="ViTB16", seed=2, AP=0.20),
        dict(base, model="MobileNetV3", seed=1, AP=0.90),
        dict(base, model="MobileNetV3", seed=2, AP=0.80),
    ])
    c = cells(df)
    check("two backbones stay TWO rows, never pooled", len(c) == 2)
    check("neither row took the other's mean",
          set(np.round(c["AP"], 3)) == {0.15, 0.85})

    # seeds ARE collapsed
    check("the seed axis IS collapsed", (c["n_seeds"] == 2).all())

    # a single-seed cell reports NO sd
    one = pd.DataFrame([dict(base, model="ViTB16", seed=1, AP=0.5)])
    c1 = cells(one)
    check("a 1-seed cell leaves sd BLANK, not 0.0",
          bool(np.isnan(c1["AP_sd"].iloc[0])))

    # the cell key is enforced, not assumed
    try:
        cells(df.drop(columns=["model"]))
        check("aggregating without the full cell key RAISES", False)
    except SystemExit:
        check("aggregating without the full cell key RAISES", True)

    # NEGATIVE CONTROL: pooling really would have changed the answer, so the
    # test above is not vacuous
    pooled = float(df["AP"].mean())
    check("negative control: pooling WOULD have moved the number",
          abs(pooled - 0.15) > 0.1 and abs(pooled - 0.85) > 0.1)

    print("\n%s" % ("ALL PASS" if ok else "FAILURES ABOVE"), file=out)
    return 0 if ok else 1


def main(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--campaign", nargs="+")
    a.add_argument("--out")
    a.add_argument("--allow-quarantined", action="store_true",
                   help="tabulate a campaign `scripts.quarantine` marked dead")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.campaign:
        raise SystemExit("--campaign is required (or --self-test)")

    # Same refusal as `full_panel`, for the same reason: a quarantined
    # campaign produces a full, plausible table. A SURVEY is if anything more
    # dangerous than a verdict here -- the rows get read side by side with live
    # ones and the marker is nowhere on the page.
    # 🛑 NO FALLBACK. This import used to be wrapped in a bare handler that
    # replaced the gate with `lambda _r: None`, so copying this file into a
    # worktree whose `scripts/` predates `quarantine.py` -- the hand-deploy
    # CLAUDE.md explicitly sanctions mid-flight -- turned the refusal off with
    # no message. A gate that cannot fail is decoration. If this import breaks,
    # the survey must break.
    from scripts.quarantine import gate
    blocked, DEAD_ARMS = gate(args.campaign, args.allow_quarantined, "tabulate")
    if blocked:
        return 1

    df, skipped = collect(args.campaign, DEAD_ARMS)
    if df.empty:
        print("no completed, scorable runs in %s" % " ".join(args.campaign))
        for k, v in skipped.most_common():
            print("  skipped %-40s %d" % (k, v))
        return 1
    c = cells(df)
    c = annotate_task(c)
    print("%d completed runs -> %d cells (seed is the ONLY collapsed axis)"
          % (len(df), len(c)))
    counts = c.drop_duplicates(["dataset", "model", "cap"])["task"]
    tally = {k: int(v) for k, v in counts.value_counts().items()}
    # 🛑 `unavailable` IS NOT ZERO. When `configs.task_cells` cannot be
    # imported -- routine in a PINNED worktree, and true today of
    # `~/OptimizationLoss`, whose `configs/` predates the instrument -- every
    # cell lands in `unavailable` and this line printed `cap poses a question
    # in 0 of 11`. A missing instrument was being reported as eleven measured
    # non-tasks, which is the exact inversion 2(z25) is about, in the tool a
    # reader trusts to tell them whether the campaign was worth running.
    if tally.get("unavailable"):
        print("  !! cap-poses-a-question is UNKNOWN for %d of %d cell(s): the "
              "task-window" % (tally["unavailable"], int(counts.size)))
        print("     instrument is not importable in this checkout. This is "
              "NOT a count of zero,")
        print("     and it is NOT a pass. Re-run from a checkout that has "
              "`configs/task_cells.py`.")
    known = int(counts.size) - tally.get("unavailable", 0)
    if known:
        print("  cap poses a question in %d of %d MEASURABLE (dataset, model, "
              "cap) cell(s): %s"
              % (tally.get("task", 0), known,
                 {k: v for k, v in tally.items() if k != "unavailable"}))
    if tally.get("non_task"):
        print("  !! %d cell(s) pose NO question (FRAMEWORK 2(z17)). "
              "Their arms cannot be" % tally["non_task"])
        print("     distinguished by construction, and pooling them "
              "with the task cells has")
        print("     changed which method wins -- 2(z19), 2(z21). "
              "Split on the `task` column.")
    for k, v in skipped.most_common():
        print("  skipped %-40s %d" % (k, v))
    if args.out:
        c.to_csv(args.out, index=False)
        print("wrote %s" % args.out)
    else:
        print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
