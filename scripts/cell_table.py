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

# The axis that may be collapsed. Everything else is part of the cell identity.
SEED_AXIS = "seed"
CELL_KEY = ["campaign", "dataset", "model", "cap", "arm"]

METRICS = ["AP", "AUROC", "ccF1", "ccP", "ccR", "macroF1", "macroP", "macroR",
           "uncF1", "acc", "ECE", "Brier", "NLL", "ConfGap"]


def collect(roots):
    """One row per COMPLETED run, with the campaign name attached."""
    rows = []
    skipped = collections.Counter()
    for camp in roots:
        name = os.path.basename(os.path.normpath(camp))
        for p in glob.glob(camp + "/**/config.json", recursive=True):
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
                skipped["panel raised: %s" % type(exc).__name__] += 1
                continue
            if not r:
                skipped["unscorable"] += 1
                continue
            res = cfg.get("results") or {}
            r["campaign"] = name
            r["steps_applied"] = res.get("constraint_steps_applied")
            r["steps_attempted"] = res.get("constraint_steps_attempted")
            rows.append(r)
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
    try:
        from scripts.quarantine import is_quarantined
    except Exception:
        is_quarantined = lambda _r: None
    blocked = [(c, q) for c in args.campaign for q in [is_quarantined(c)] if q]
    if blocked and not args.allow_quarantined:
        for c, q in blocked:
            print("REFUSING to tabulate %s" % c)
            print("  reason   : %s" % q.get("reason"))
            print("  keep for : %s" % q.get("keep_for"))
        return 1
    for c, q in blocked:
        print("!! TABULATING A QUARANTINED CAMPAIGN: %s -- %s"
              % (c, q.get("reason")))

    df, skipped = collect(args.campaign)
    if df.empty:
        print("no completed, scorable runs in %s" % " ".join(args.campaign))
        for k, v in skipped.most_common():
            print("  skipped %-40s %d" % (k, v))
        return 1
    c = cells(df)
    print("%d completed runs -> %d cells (seed is the ONLY collapsed axis)"
          % (len(df), len(c)))
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
