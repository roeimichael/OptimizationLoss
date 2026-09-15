"""WHERE do the saturated cuts sit, and how much of the prize do they hold?

Gate 3 found that 35.6% of the allocator's cuts fall above probability 0.99,
where candidates are numerically indistinguishable and no re-ranking loss can
act. That is a ceiling on ANY score-improving method here, so the next question
is whether it is structural or addressable:

  - if the saturated cells are TINY (K of 2-10 slots) they hold little of the
    prize, the ceiling is cosmetic, and the contested cells are where the work
    is;
  - if they are LARGE, the ceiling is real and bounds what Stage 1 could ever
    have shown.

Splits the same `headroom` cells by where the cut falls and reports how much
capacity, headroom and error each band actually owns.

Reads stored deployed selections. No arm-vs-arm comparison: this describes the
TASK, not any method, so it cannot preview the Stage 1 gate.
"""
import collections
import json
import sys

import numpy as np


def band(p):
    if p is None:
        return None
    if p > 0.99:
        return "SATURATED >0.99"
    if p > 0.8:
        return "confident .8-.99"
    if p > 0.2:
        return "middling .2-.8"
    return "contested <0.2"


ORDER = ["contested <0.2", "middling .2-.8", "confident .8-.99", "SATURATED >0.99"]


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")
            if l.strip().startswith("{")]
    rows = [r for r in rows if r.get("support", 0) > 0
            and r.get("cut_probability") is not None]

    by = collections.defaultdict(list)
    for r in rows:
        by[band(r["cut_probability"])].append(r)

    tot_K = sum(r["K"] for r in rows)
    tot_out = sum(r["outside_tp"] for r in rows)
    tot_err = sum(r["selected_errors"] for r in rows)

    print("%-18s %6s %8s %10s %12s %10s %10s"
          % ("where the cut is", "cells", "med K", "slots", "% of slots",
             "outside_tp", "sel_err"))
    for b in ORDER:
        rs = by.get(b, [])
        if not rs:
            continue
        K = sum(r["K"] for r in rs)
        print("%-18s %6d %8.0f %10d %11.1f%% %10d %10d"
              % (b, len(rs), float(np.median([r["K"] for r in rs])), K,
                 100.0 * K / tot_K, sum(r["outside_tp"] for r in rs),
                 sum(r["selected_errors"] for r in rs)))
    print("%-18s %6d %8s %10d %11.1f%% %10d %10d"
          % ("TOTAL", len(rows), "", tot_K, 100.0, tot_out, tot_err))

    sat = by.get("SATURATED >0.99", [])
    live = [r for b in ORDER[:3] for r in by.get(b, [])]
    sat_K = sum(r["K"] for r in sat)
    print("")
    print("THE CEILING, stated as a fraction of the prize:")
    print("  saturated cells hold %.1f%% of all slots, %.1f%% of the outside_tp"
          % (100.0 * sat_K / tot_K,
             100.0 * sum(r["outside_tp"] for r in sat) / max(tot_out, 1)))
    print("  and %.1f%% of the errors inside the selection."
          % (100.0 * sum(r["selected_errors"] for r in sat) / max(tot_err, 1)))
    print("  median K: saturated %.0f vs actionable %.0f"
          % (float(np.median([r["K"] for r in sat])) if sat else float("nan"),
             float(np.median([r["K"] for r in live])) if live else float("nan")))
    print("")
    frac = 100.0 * sat_K / tot_K
    if frac < 15:
        print("READING: the saturated cells are SMALL. They are a third of the")
        print("  cells but a minor share of the slots, so the ceiling is largely")
        print("  cosmetic and the contested cells hold the prize.")
    elif frac > 40:
        print("READING: the saturated cells hold most of the capacity. The")
        print("  ceiling is REAL and bounds what any score-improving method,")
        print("  Stage 1 included, could have shown on this slice.")
    else:
        print("READING: the saturated cells hold a material but not dominant")
        print("  share of the capacity -- a partial ceiling. A method that only")
        print("  works on contested cuts is competing for the remainder.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
