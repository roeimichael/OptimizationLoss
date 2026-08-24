"""Split each dual family's score against `clip` into COMPUTE and CONSTRAINT.

WHY THIS EXISTS. Every trained arm gets 29 epochs the post-hoc clipper does
not. `full_panel --control clip` therefore measures (compute + constraint) as
one number, and the corpus reports exactly that number for six methods -- with
no lambda=0 twin anywhere in 7,574 rows, so none of it is attributable. This
tool needs the twin and refuses without it:

    total      = arm        - clip        what the paper reports
    compute    = arm_null   - clip        29 epochs at lambda = 0
    constraint = arm        - arm_null    the method's OWN contribution

and total = compute + constraint identically, per seed, so the split is
arithmetic rather than a model.

THE POSITIVE CONTROL IS FREE AND MANDATORY. At lambda = 0 the dual family is
irrelevant: same warm-up cache, same allocator, same seed, no constraint
gradient. So `tralo_null`, `fioretto_null` and `hounie_null` must be the SAME
RUN. Measured on xfam1 2026-08-24: byte-identical raw predictions in 12 of 12
cell-seeds. If they ever diverge, something other than lambda differs between
the families and EVERY per-family attribution below is contaminated -- so this
script checks the digests and refuses rather than reporting.

READ uncF1 BESIDE macroF1. On iwildcam 6 of the 8 classes are uncapped and
they carry macro-F1. The constraint names only the capped ones, so uncF1 is
pure collateral damage and it is where the families actually differ.

    python -m scripts.family_split --campaign results/xfam1
"""

import argparse
import collections
import json
import os

import numpy as np

from scripts.full_panel import panel

# Lower is better for these, so a raw delta flips sign before it is read.
LOWER_BETTER = {"ECE", "Brier", "NLL"}
METRICS = ["AP", "AUROC", "ECE", "Brier", "NLL",
           "ccF1", "uncF1", "macroF1"]


def load(root):
    """Every completed run in the campaign, keyed by (cell, arm, seed)."""
    out = {}
    for dirpath, _, files in os.walk(root):
        if "config.json" not in files:
            continue
        cfg = json.load(open(os.path.join(dirpath, "config.json")))
        if cfg.get("status") != "completed":
            continue
        row = panel(dirpath, cfg)
        if row is None:
            continue
        cell = (row["model"], row["cap"], row["capped"])
        out[(cell, row["arm"], row["seed"])] = row
    return out


def matched(rows, arms):
    """Cell-seeds where EVERY arm is present.

    Unmatched pooling is how this project once compared `clip` measured on 7
    cells against a treatment measured on 6 and read the cell difference as a
    method difference.
    """
    seeds = collections.defaultdict(set)
    for (cell, arm, seed) in rows:
        seeds[(cell, seed)].add(arm)
    keep = sorted(k for k, have in seeds.items() if set(arms) <= have)
    # Report the DROP. "16 matched cell-seeds" reads very differently when 18
    # existed than when 200 did, and silence made the two look identical.
    dropped = collections.Counter()
    for k, have in seeds.items():
        if set(arms) <= have:
            continue
        for a in sorted(set(arms) - have):
            dropped[a] += 1
    if dropped:
        print("  %d of %d cell-seed(s) DROPPED as incomplete. Missing arm "
              "counts: %s" % (len(seeds) - len(keep), len(seeds),
                              ", ".join("%s x%d" % (a, n)
                                        for a, n in dropped.most_common())))
    return keep


def null_identity(rows, keys, nulls):
    """The lambda=0 arms must be the same run. Returns the offending keys."""
    bad = []
    for (cell, seed) in keys:
        digests = {rows[(cell, n, seed)]["raw_md5"] for n in nulls
                   if (cell, n, seed) in rows}
        if len(digests) > 1:
            bad.append((cell, seed, sorted(digests)))
    return bad


def signed(metric, d):
    return -d if metric in LOWER_BETTER else d


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--campaign", required=True)
    a.add_argument("--control", default="clip")
    a.add_argument("--families", nargs="+",
                   default=["tralo", "fioretto", "hounie"])
    a.add_argument("--floor", default="tralo_reseed",
                   help="arm that differs from its null by RNG alone")
    args = a.parse_args()

    rows = load(args.campaign)
    nulls = [f + "_null" for f in args.families]
    need = [args.control] + list(args.families) + nulls
    keys = matched(rows, need)
    if not keys:
        raise SystemExit(
            "No cell-seed carries all of %s. This tool cannot substitute an\n"
            "unpaired read: without the lambda=0 twin the compute and the\n"
            "constraint are one number, which is the corpus's defect." % need)

    bad = null_identity(rows, keys, nulls)
    if bad:
        for cell, seed, digests in bad[:5]:
            print("  NULLS DIVERGE %s seed %s: %s" % (cell, seed, digests))
        raise SystemExit(
            "The lambda=0 arms are not the same run, so something other than\n"
            "lambda differs between the families and no attribution below\n"
            "would be valid. Refusing to print one.")

    cells = sorted({c for c, _ in keys})
    print("=" * 94)
    print("FAMILY SPLIT  --  %d matched cell-seed(s) over %d cell(s), control=%s"
          % (len(keys), len(cells), args.control))
    print("  nulls byte-identical in all %d: the compute term is ONE number, "
          "not one per family" % len(keys))
    print("=" * 94)

    scale = float(np.mean([rows[(c, args.control, s)]["items_per_001"]
                           for c, s in keys]))

    for metric in METRICS:
        print()
        print("  %s%s" % (metric, "  (lower is better; sign flipped so + = good)"
                          if metric in LOWER_BETTER else ""))
        print("     %-10s %12s %12s %12s   %s"
              % ("family", "compute", "constraint", "total",
                 "cells won (constraint)"))
        for fam in args.families:
            nul = fam + "_null"
            comp, cons, tot = [], [], []
            percell = collections.defaultdict(list)
            for c, s in keys:
                v_c = rows[(c, args.control, s)][metric]
                v_n = rows[(c, nul, s)][metric]
                v_a = rows[(c, fam, s)][metric]
                comp.append(signed(metric, v_n - v_c))
                cons.append(signed(metric, v_a - v_n))
                tot.append(signed(metric, v_a - v_c))
                percell[c].append(signed(metric, v_a - v_n))
            # A cell whose metric is NaN is UNMEASURABLE, not a loss. `nan > 0`
            # is False, so counting it in the denominator would have reported
            # "2/9" for a contrast that actually resolved 6 cells -- the
            # absent-data-reads-as-a-value class, in the win column.
            cellmeans = {c: np.mean(percell[c]) for c in percell}
            nan_cells = [c for c, v in cellmeans.items() if not np.isfinite(v)]
            won = sum(1 for v in cellmeans.values() if np.isfinite(v) and v > 0)
            res = len(cellmeans) - len(nan_cells)
            print("     %-10s %+12.4f %+12.4f %+12.4f   %d/%d%s"
                  % (fam, np.mean(comp), np.mean(cons), np.mean(tot),
                     won, res,
                     "  (%d cell(s) UNMEASURABLE, excluded)" % len(nan_cells)
                     if nan_cells else ""))
        if args.floor:
            fk = [(c, s) for c, s in keys
                  if (c, args.floor, s) in rows and (c, "tralo_null", s) in rows]
            if fk:
                d = [signed(metric, rows[(c, args.floor, s)][metric]
                            - rows[(c, "tralo_null", s)][metric]) for c, s in fk]
                print("     %-10s %12s %+12.4f %12s   noise floor: RNG stream "
                      "only, over %d" % (args.floor, "-", np.mean(d), "-", len(fk)))
        if metric == "ccF1":
            print("     ^ items = d(ccF1) * %.1f per 0.01. The whole gap from "
                  "`clip` to a" % scale)
            print("       PERFECT allocator is 1.9-9.9 items PER CLASS, so read "
                  "this line in items.")
    print()
    print("  total = compute + constraint exactly, per seed. A family that "
          "'wins' on total")
    print("  while its constraint term is negative won on the 29 epochs, "
          "which every")
    print("  trained arm gets and which the clipper does not.")


if __name__ == "__main__":
    main()
