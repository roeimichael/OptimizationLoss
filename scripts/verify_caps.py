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
    a.add_argument("--strict", action="store_true",
                   help="exit 1 if any cap is inert/redundant or any group is "
                        "uninformative, so this can gate a launch. Off by "
                        "default: an inert global cap is a real fact about the "
                        "campaign, not necessarily a mistake -- but it must "
                        "never be a SILENT one.")
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
            # A skip is a FAILURE of this check, not a pass. With data/ absent
            # every dataset skipped and it still printed "CAP CHECK OK -- every
            # cap tag produces a real integer budget on every dataset" and
            # returned 0. It is the only data-touching gate in the repo and it
            # was the one that could not fail.
            print("%-12s FAIL -- could not read the slice: %s" % (ds, e))
            fails.append("%s: slice unreadable (%s). This gate proves nothing "
                         "about a dataset it never opened." % (ds, e))
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

        # A local cap is only a DIFFERENT constraint from the global one if the
        # groups differ in class composition. `synth_group` is built by
        # round-robin over array order, so every group gets the same class mix
        # and each local budget is just global/G -- the local scope then tests
        # nothing the global scope does not.
        overall = np.array([counts.get(c, 0) for c in range(n_cls)],
                           dtype=float) / n
        tvs = []
        for gid in sorted(df[gcol].unique()):
            sub = df[df[gcol] == gid]
            share = np.array([(sub["label"] == c).sum() for c in range(n_cls)],
                             dtype=float) / max(1, len(sub))
            tvs.append(0.5 * np.abs(share - overall).sum())
        worst = max(tvs) if tvs else 0.0
        print("  groups: %d, class-mix distance from the whole test set "
              "(total variation) %s"
              % (len(tvs), ["%.3f" % t for t in tvs]))
        if worst < 0.05:
            print("     UNINFORMATIVE GROUPS: every group has the same class mix "
                  "(max %.3f), so each local budget is essentially global/%d and "
                  "the local scope adds no constraint the global one does not."
                  % (worst, len(tvs)))
            inert.append("%s: groups carry no class information (max TV %.3f)"
                         % (ds, worst))

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
                # K=0 is the TIGHTEST possible cap, not slack -- `0 >= 0` would
                # otherwise report the hardest constraint in the campaign as
                # non-binding.
                zero_k = sorted(g for g in lcon if lcon[g][c] == 0)
                # A cap binds when the model's PREDICTED count exceeds it,
                # not its true count. This uses the true count, which is the
                # only thing available before a run -- so it can only ever say
                # "cannot bind even against a perfect classifier". FRAMEWORK
                # records over-prediction of 1.7-2.4x, so at L30/L50 the two
                # criteria agree; at L100+ this would call a binding cap slack.
                slack = [g for g in lcon
                         if lcon[g][c] > 0 and lcon[g][c] >= int(true_by_group.get(g, 0))]
                if zero_k:
                    print("              K=0 on group(s) %s -- no true instance "
                          "of the class there, so the budget is zero. Real and "
                          "binding; verify the loss is driving it to zero."
                          % zero_k)
                if len(slack) + len(zero_k) == len(lcon) and not zero_k:
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
        if args.strict:
            print()
            print("--strict: failing because the campaign would not exercise "
                  "what its tags claim.")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
