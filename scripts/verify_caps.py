"""Show the integer budgets a cap tag actually produces on the real test set.

A cap tag is a PERCENTAGE; what binds the model is the integer K it rounds to on
this dataset's test labels. Those are different questions: L30_G30 is a hard cap
on dermmnist's melanoma and a nearly free one on a class with 40 test samples,
and a percentage that rounds to K=0 disables the constraint silently.

This runs the pipeline's own `constraints.py` -- not a reimplementation -- so
what it prints is what the loss will receive.

    python -m scripts.verify_caps --datasets dermmnist octmnist tissuemnist \\
        --caps L30_G30 L30_G50 L50_G50
    python -m scripts.verify_caps --datasets dermmnist --caps L30_G50 \\
        --constrained-class 4 5      # the coupled multi-class setting
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.gen_campaign import cap_pair, load_protocol          # noqa: E402
from src.training.constraints import (compute_global_constraints,  # noqa: E402
                                      compute_local_constraints)
from src.utils.constants import UNLIMITED                          # noqa: E402


def load_test(dc):
    y = np.load(os.path.join(dc["data_dir"], "test_labels.npy")).ravel()
    meta = pd.read_csv(os.path.join(dc["data_dir"], "test_meta.csv"))
    groups = meta[dc["group_column"]].values.astype(np.int64)
    return pd.DataFrame({"label": y, dc["group_column"]: groups})


def main():
    P = load_protocol()
    a = argparse.ArgumentParser()
    a.add_argument("--datasets", nargs="+", default=sorted(P["datasets"]))
    a.add_argument("--caps", nargs="+", default=["L30_G30", "L30_G50", "L50_G50"])
    a.add_argument("--constrained-class", nargs="+", type=int, default=None)
    args = a.parse_args()
    fails, inert = [], []

    for ds in args.datasets:
        dc = dict(P["datasets"][ds])
        cls = (args.constrained_class if args.constrained_class is not None
               else dc["constrained_class"])
        cls = cls if isinstance(cls, list) else [cls]
        gcol, n_cls = dc["group_column"], dc["num_classes"]

        try:
            df = load_test(dc)
        except (OSError, KeyError) as e:
            print("%-12s SKIP -- %s" % (ds, e))
            continue

        n = len(df)
        counts = df["label"].value_counts().sort_index()
        print("=" * 78)
        print("%s   %d test items, %d classes, %d groups (%s)"
              % (ds, n, n_cls, df[gcol].nunique(), gcol))
        print("  class counts: %s"
              % "  ".join("%d:%d" % (c, counts.get(c, 0)) for c in range(n_cls)))
        bad = [c for c in cls if not 0 <= c < n_cls]
        if bad:
            fails.append("%s: class %s out of range for num_classes=%d"
                         % (ds, bad, n_cls))
            print("  FAIL: constrained_class %s out of range" % bad)
            continue
        print("  constrained: %s  (%s of test)"
              % (cls, ", ".join("%.1f%%" % (100.0 * counts.get(c, 0) / n) for c in cls)))

        for tag in args.caps:
            local_pct, global_pct = cap_pair(tag)
            try:
                gcon = compute_global_constraints(
                    df, "label", global_pct, constrained_class=cls, num_classes=n_cls)
                lcon = compute_local_constraints(
                    df, "label", local_pct, gcol,
                    constrained_class=cls, num_classes=n_cls)
            except ValueError as e:
                fails.append("%s %s: %s" % (ds, tag, e))
                print("    %-9s FAIL -- %s" % (tag, e))
                continue

            for c in cls:
                K_g = gcon[c]
                per_group = sorted(v[c] for v in lcon.values())
                # the binding budget is min(global, sum of the local caps)
                eff = min(K_g, sum(per_group))
                print("    %-9s class %d: global K=%-5d local K per group=%s "
                      "sum=%d -> effective %d (%.1f%% of the %d true)"
                      % (tag, c, K_g, per_group, sum(per_group), eff,
                         100.0 * eff / max(1, counts.get(c, 0)), counts.get(c, 0)))
                # A cap that can never be reached is an inert flag one level
                # up from a dead config key, and just as invisible.
                lsum = sum(per_group)
                if K_g > lsum:
                    print("              INERT GLOBAL: K=%d is above the local sum "
                          "%d, so it can never bind -- this tag runs the same "
                          "experiment as L%02d_G%02d."
                          % (K_g, lsum, int(local_pct * 100), int(local_pct * 100)))
                    inert.append("%s %s class %d: global (slack by %d)"
                                 % (ds, tag, c, K_g - lsum))
                elif K_g == lsum:
                    print("              REDUNDANT GLOBAL: K=%d equals the local "
                          "sum, so it binds only when every group is already at "
                          "its own cap -- it adds no constraint of its own." % K_g)
                    inert.append("%s %s class %d: global (redundant)" % (ds, tag, c))
                true_by_group = df[df["label"] == c].groupby(gcol).size()
                slack = [g for g in lcon
                         if lcon[g][c] >= int(true_by_group.get(g, 0))]
                if len(slack) == len(lcon):
                    print("              INERT LOCAL: every group cap >= that "
                          "group's true count, so no local cap can bind")
                    inert.append("%s %s class %d: local" % (ds, tag, c))
        print()

    print("=" * 78)
    if fails:
        print("CAP CHECK FAILED:")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("CAP CHECK OK -- every cap tag produces a real integer budget on every "
          "dataset.")
    if inert:
        print()
        print("%d INERT cap(s) -- these bind nothing, so the arm runs unconstrained"
              % len(inert))
        print("on that scope and the tag overstates what was tested:")
        for i in inert:
            print("  - %s" % i)
        print("A global cap only adds a constraint when it is strictly BELOW the")
        print("sum of the local caps, so G>=L never does. Sweep G<L to make the")
        print("global scope the thing under test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
