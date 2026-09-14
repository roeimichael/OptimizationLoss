"""The pre-registered interaction: does the constraint buy MORE when the boundary is alive?

`docs/MISSION.md` registers the test as

    gAP(X_tralo) - gAP(X_null)   compared against   gAP(tralo) - gAP(tralo_null)

for each intervention column X, seed-paired, per cell. Two things make that
easy to get wrong by hand, and this script exists because both already have:

1. **The control must share the column.** `gen_campaign` force-adds
   `tralo_null`, which reads as "there is a control" -- but it is the PLAIN
   null. Comparing `aug_tralo` against it confounds the constraint with the
   augmentation, and comparing against `aug_clip` confounds it with the
   schedule. Only an arm sharing the column's warm-up identity isolates the
   constraint, so this script REFUSES a column whose null is missing rather
   than silently substituting one.
2. **gAP is the allocation-free channel**, computed from the stored
   probabilities alone. It is the quantity the account is about, and it cannot
   be read off a metrics CSV -- it has to be recomputed per group.

A main effect of the intervention confirms NOTHING: the matched clipper already
has one. Only the interaction is evidence.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Column -> (trained arm, its zero-constraint control). Each pair must share a
# warm-up identity; `test_every_INTERVENTION_column_has_its_own_zero_constraint_control`
# enforces that against the protocol.
COLUMNS = (
    ("plain", "tralo", "tralo_null"),
    ("augment", "aug_tralo", "aug_tralo_null"),
    ("focal", "focal_tralo", "focal_tralo_null"),
)


def group_ap(run_dir, classes):
    """Per-group average precision on the raw probabilities, averaged.

    Allocation-free by construction: it reads the probability column and never
    the predicted label, so no allocator can influence it.
    """
    df = pd.read_csv(os.path.join(run_dir, "final_predictions_raw.csv"))
    y, groups = df.True_Label.to_numpy(), df.Group_ID.to_numpy()
    scores = []
    for c in classes:
        p = df["Prob_Class_%d" % c].to_numpy()
        for gid in np.unique(groups):
            m = groups == gid
            truth = (y[m] == c).astype(int)
            # A group with no positive (or no negative) has undefined AP.
            if 0 < truth.sum() < truth.size:
                scores.append(average_precision_score(truth, p[m]))
    return float(np.mean(scores)) if scores else float("nan")


def collect(globs):
    """(cap, seed) -> {arm: gAP} for every scorable run."""
    per = collections.defaultdict(dict)
    for g in globs:
        for d in sorted(glob.glob(g)):
            cj = os.path.join(d, "config.json")
            if not os.path.exists(os.path.join(d, "final_predictions_raw.csv")):
                continue
            cfg = json.load(open(cj))
            cls = cfg["dataset_config"]["constrained_class"]
            cls = cls if isinstance(cls, list) else [cls]
            key = (cfg["model_name"], cfg["dataset_mode"],
                   cfg["constraint_tag"], cfg["hyperparams"]["seed"])
            per[key][cfg["arm"]] = group_ap(d, cls)
    return per


def paired(per, cell, a, b):
    """Seed-paired deltas for one cell. Empty when either arm is absent."""
    return [per[k][a] - per[k][b]
            for k in sorted(per)
            if k[:3] == cell and a in per[k] and b in per[k]]


def summarise(per):
    cells = sorted({k[:3] for k in per})
    rows = []
    for cell in cells:
        effects = {}
        for label, arm, null in COLUMNS:
            present = any(arm in per[k] for k in per if k[:3] == cell)
            if not present:
                continue
            d = paired(per, cell, arm, null)
            if not d:
                rows.append((cell, label, None, None, None,
                             "REFUSED: %s has no %s in this cell, so its "
                             "constraint cannot be isolated" % (arm, null)))
                continue
            effects[label] = d
            rows.append((cell, label, len(d), float(np.mean(d)),
                         float(np.std(d, ddof=1)) if len(d) > 1 else 0.0, ""))
        base = effects.get("plain")
        for label in ("augment", "focal"):
            if base and label in effects and len(base) == len(effects[label]):
                inter = [x - y for x, y in zip(effects[label], base)]
                rows.append((cell, "%s x constraint" % label, len(inter),
                             float(np.mean(inter)),
                             float(np.std(inter, ddof=1)) if len(inter) > 1 else 0.0,
                             "INTERACTION"))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", required=True)
    args = ap.parse_args(argv)
    per = collect(args.glob)
    if not per:
        print("no scorable runs matched")
        return 1
    rows = summarise(per)
    print("gAP is allocation-free. Each row is the CONSTRAINT's effect inside its")
    print("own column: gAP(arm) - gAP(that column's zero-constraint null).")
    print("A main effect of an intervention confirms nothing -- only INTERACTION rows do.")
    print("")
    print("  %-34s %-20s %3s %10s %10s" % ("cell", "column", "n", "mean", "sd"))
    for cell, label, n, mean, sd, note in rows:
        name = "/".join(cell)
        if n is None:
            print("  %-34s %-20s  %s" % (name, label, note))
            continue
        print("  %-34s %-20s %3d %+10.5f %10.5f %s" % (
            name, label, n, mean, sd, note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
