"""Does the unit-norm clip actually bind, and does that differ by dataset?

The whole "lambda is divided out" story depends on the constraint gradient's
norm exceeding the clip bound. On DermMNIST it was measured at 20-25 against a
bound of 1.0, so the clip rescales by ~22x and lambda cannot affect the update.
But an OctMNIST log shows gn = 0.40 and 0.86 -- below or near the bound.

If that holds, the clip binds on one dataset and not the other, which would mean
the method is a unit-norm directional nudge on Derm and a conventional penalty
on Oct. That is a mechanism for the dataset split, and it is checkable: the
`steps` arm logs the pre-clip norm every epoch.

Reads the stdout lane logs, not the CSVs -- gn is only in stdout.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

LINE = re.compile(r"Epoch (\d+) \[Constraint\].*?lam_T=([\d.]+).*?gn=([\d.]+) clipped=([Yn])")
RUN = re.compile(r"Running (\S+/config\.json)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+",
                    default=sorted(glob.glob(os.path.expanduser("~/steps*_l*.log"))))
    args = ap.parse_args()

    rows = []
    for lg in args.logs:
        cur = None
        try:
            fh = open(lg, errors="replace")
        except IOError:
            continue
        with fh:
            for ln in fh:
                m = RUN.search(ln)
                if m:
                    p = m.group(1)
                    # .../{model}/{dataset}/{cap}/{arm}/seed_N/config.json
                    parts = p.split("/")
                    cur = {"model": parts[-6], "dataset": parts[-5],
                           "cap": parts[-4], "arm": parts[-3]} if len(parts) >= 6 else None
                    continue
                g = LINE.search(ln)
                if g and cur:
                    rows.append({**cur, "epoch": int(g.group(1)),
                                 "lam": float(g.group(2)),
                                 "gn": float(g.group(3)),
                                 "clipped": g.group(4) == "Y"})
    if not rows:
        print("no gn lines parsed -- is the logging patch live in this worktree?")
        return 1
    d = pd.DataFrame(rows)
    d = d[d.gn > 0]           # gn==0 means no constraint step was taken
    print("parsed %d epochs with a constraint step, across %d arms"
          % (len(d), d.arm.nunique()))

    print()
    print("=" * 88)
    print("PRE-CLIP GRADIENT NORM by dataset  (only epochs where a step was taken)")
    print("=" * 88)
    t = d.groupby(["dataset", "arm"]).agg(
        n=("gn", "size"), gn_median=("gn", "median"), gn_p90=("gn", lambda s: s.quantile(.9)),
        gn_max=("gn", "max"), frac_clipped=("clipped", "mean"), lam_max=("lam", "max"))
    print(t.round(3).to_string())

    print()
    print("=" * 88)
    print("THE QUESTION: at the incumbent bound of 1.0, does the clip bind?")
    print("=" * 88)
    inc = d[d.arm.isin(["steps_clip1"])]
    if inc.empty:
        print("  no steps_clip1 runs in these logs")
    else:
        for ds, g in inc.groupby("dataset"):
            over = float((g.gn > 1.0).mean())
            print("  %-12s median gn %8.3f   %5.1f%% of steps exceed 1.0   -> %s"
                  % (ds, g.gn.median(), 100 * over,
                     "clip BINDS, lambda divided out" if over > 0.5
                     else "clip rarely binds, lambda ACTS"))

    print()
    print("=" * 88)
    print("gn vs lambda -- is the norm just tracking the ratchet?")
    print("=" * 88)
    for ds, g in d.groupby("dataset"):
        if g.lam.nunique() > 2:
            r = g[["lam", "gn"]].corr(method="spearman").iloc[0, 1]
            print("  %-12s spearman(lambda, gn) = %+0.3f  (n=%d)" % (ds, r, len(g)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
