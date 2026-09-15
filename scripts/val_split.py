"""Can a LEGITIMATE validation split be carved out of fmow2's train countries?

THE PROBLEM IT SOLVES. `scripts/epoch_curve.py` can already show the constraint's
per-epoch contribution, but its best epoch is chosen against the TEST set, so it
is an oracle bound and not a method. Turning "stop at the best epoch" into
something reportable needs a held-out split that is not the test set -- and
`data/fmow2/oodslice/` ships only train and test.

WHY A RANDOM SPLIT OF TRAIN WOULD BE THE WRONG ONE. Train and test share ZERO
countries: 139 vs 10, disjoint by construction. The deployment shift this
dataset poses is a GROUP shift, so a validation split drawn by shuffling rows
would leave the same countries on both sides, measure a strictly easier problem,
and pick epochs for it. The split has to be group-disjoint to be a proxy for the
thing it is standing in for.

This scores candidate group-disjoint splits on how closely they reproduce the
TEST profile -- group count, item count, constrained-class balance -- while
leaving enough training data behind. It writes nothing; it only reports what is
available.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd


def profile(y, classes):
    return np.array([float((y == c).mean()) for c in classes])


def main(run_dir, n_val_groups=10, n_candidates=4000, seed=0):
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    dc = cfg["dataset_config"]
    classes = dc["constrained_class"]
    classes = classes if isinstance(classes, list) else [classes]
    gcol = dc["group_column"]
    ddir = dc["data_dir"]

    ytr = np.load(os.path.join(ddir, "train_labels.npy")).ravel()
    gtr = pd.read_csv(os.path.join(ddir, "train_meta.csv"))[gcol].to_numpy()
    yte = np.load(os.path.join(ddir, "test_labels.npy")).ravel()
    gte = pd.read_csv(os.path.join(ddir, "test_meta.csv"))[gcol].to_numpy()

    tgt_prof = profile(yte, classes)
    tgt_n = len(yte)
    tgt_sizes = np.sort(pd.Series(gte).value_counts().to_numpy())[::-1]
    print("TEST (the thing a val split must imitate):")
    print("  %d items in %d groups; group sizes %s"
          % (tgt_n, len(set(gte)), list(tgt_sizes)))
    print("  constrained-class shares %s"
          % ", ".join("cls%d:%.1f%%" % (c, 100 * p) for c, p in zip(classes, tgt_prof)))

    sizes = pd.Series(gtr).value_counts()
    # Only groups big enough to host a per-group cut are usable; a 3-item
    # country contributes nothing to a group-wise allocation decision.
    usable = sizes[sizes >= 80]
    print("")
    print("TRAIN: %d items in %d groups; %d groups have >= 80 items"
          % (len(ytr), len(sizes), len(usable)))
    if len(usable) < n_val_groups + 5:
        print("  NOT ENOUGH usable groups to carve a %d-group val split while "
              "leaving a usable train set." % n_val_groups)
        return 1

    rng = np.random.default_rng(seed)
    names = usable.index.to_numpy()
    best = None
    for _ in range(n_candidates):
        pick = rng.choice(names, size=n_val_groups, replace=False)
        m = np.isin(gtr, pick)
        n_val = int(m.sum())
        if not (0.6 * tgt_n <= n_val <= 1.6 * tgt_n):
            continue
        prof = profile(ytr[m], classes)
        # distance on class balance, plus a mild penalty on size mismatch
        d = float(np.abs(prof - tgt_prof).sum()) + abs(n_val - tgt_n) / tgt_n * 0.1
        if best is None or d < best[0]:
            best = (d, sorted(pick.tolist()), n_val, prof)

    if best is None:
        print("  no candidate split lands near the test set's item count.")
        return 1

    d, pick, n_val, prof = best
    m = np.isin(gtr, pick)
    print("")
    print("BEST group-disjoint val split found (of %d candidates):" % n_candidates)
    print("  groups: %s" % ", ".join(pick))
    print("  %d items (test has %d), leaving %d train items in %d groups"
          % (n_val, tgt_n, len(ytr) - n_val, len(set(gtr[~m]))))
    print("  constrained-class shares %s"
          % ", ".join("cls%d:%.1f%%" % (c, 100 * p) for c, p in zip(classes, prof)))
    print("  total class-share mismatch vs test: %.3f" % float(np.abs(prof - tgt_prof).sum()))
    print("")
    print("READING: a group-disjoint validation split IS constructible from the "
          "train countries, at roughly the test set's size and class balance, "
          "while still leaving %.0f%% of the training data."
          % (100.0 * (len(ytr) - n_val) / len(ytr)))
    print("  This is what would turn epoch_curve's ORACLE best-epoch into a "
          "reportable stopping rule. It costs a retrain of every arm, so it is "
          "a compute-budget decision, not a unilateral one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
