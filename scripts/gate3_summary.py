"""Aggregate `scripts.headroom` into the four numbers gate 3 actually asks for.

FRAMEWORK gate 3: "inspect actual per-group allocation cuts, mistakes inside the
selected set, informative caps, and whether true positives outside it provide
correctable headroom. Screen every backbone separately."

RULESET section 2 adds the reason this cannot be skipped: "Training accuracy
alone cannot diagnose test-cut saturation. gate:saturation reads train accuracy.
It is a screen, NOT a diagnosis of the regime. Pair it with the test-side cut."

So the question this answers is the one train accuracy cannot: at the place the
allocator actually cuts, are the items DISTINGUISHABLE? A cut sitting at
probability 0.99999 is a cut through a saturated region -- every candidate looks
identical to the score, and no re-ranking loss can act there, however live the
training loop looks. A cut at 0.2 is contested and has something to learn.

Reads stored deployed selections only. No training, no gradients.
"""
import collections
import json
import sys

import numpy as np


def main(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        if r.get("support", 0) == 0:          # class absent from the group
            continue
        rows.append(r)
    if not rows:
        print("no scorable cells")
        return 2

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["backbone"], r["cap"])].append(r)

    print("%-14s %-9s %5s %8s %10s %9s %9s %s"
          % ("backbone", "cap", "cells", "binds", "outside_tp", "sel_err",
             "cut>0.99", "median cut"))
    for key in sorted(by):
        rs = by[key]
        binds = sum(1 for r in rs if r["emitted"] == r["K"])
        out_tp = sum(r["outside_tp"] for r in rs)
        sel_err = sum(r["selected_errors"] for r in rs)
        emitted = sum(r["emitted"] for r in rs)
        cuts = [r["cut_probability"] for r in rs if r["cut_probability"] is not None]
        sat = sum(1 for c in cuts if c > 0.99)
        print("%-14s %-9s %5d %8s %10d %9s %9s %10.3f"
              % (key[0], key[1], len(rs), "%d/%d" % (binds, len(rs)), out_tp,
                 "%d/%d" % (sel_err, emitted),
                 "%d/%d" % (sat, len(cuts)), float(np.median(cuts))))

    allcuts = [r["cut_probability"] for r in rows if r["cut_probability"] is not None]
    allcuts = np.array(allcuts)
    print("")
    print("CUT-PROBABILITY DISTRIBUTION over %d cells -- the test-side regime:" % len(allcuts))
    for lo, hi, label in [(0.0, 0.2, "contested   <0.2 "),
                          (0.2, 0.8, "middling 0.2-0.8 "),
                          (0.8, 0.99, "confident 0.8-.99"),
                          (0.99, 1.01, "SATURATED  >0.99")]:
        n = int(((allcuts >= lo) & (allcuts < hi)).sum())
        bar = "#" * int(40.0 * n / len(allcuts))
        print("  %-18s %4d  %5.1f%%  %s" % (label, n, 100.0 * n / len(allcuts), bar))
    print("")
    print("READING:")
    print("  binds      = cap is real (emitted == K). If this is not n/n the cap")
    print("               is slack and the task is not the one we think.")
    print("  outside_tp = true positives sitting OUTSIDE the cut: the correctable")
    print("               headroom. Zero would mean nothing is left to win.")
    print("  cut>0.99   = cells whose cut falls in a SATURATED region, where the")
    print("               items are numerically indistinguishable. A re-ranking")
    print("               loss cannot act on those, no matter how live training is.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
