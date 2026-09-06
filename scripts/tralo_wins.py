"""THE ACCEPTANCE TABLE. Does TraLO beat the clipper AND the duals often enough?

The bar, set 2026-09-06 and not to be re-litigated per-result:

    A cell counts as a TRALO WIN only if `tralo` beats the post-hoc control
    AND beats EVERY rival dual present in that same cell.
    TraLO passes if it wins at least 50% of the cells that can test it.

Two things this refuses to do, because both have produced a wrong headline here
before:

  * **Cells with no rival are NOT in the denominator.** A campaign that staged
    `tralo` alone cannot test the claim, and counting it as neither a win nor a
    loss is the only honest treatment. They are printed separately so the
    coverage hole stays visible instead of being averaged away.
  * **A WIN IS A SIGN, NOT A MEASUREMENT.** The `priced` column says whether
    the cell could support the claim at all: the spread must clear the RNG
    floor AND that floor must rest on at least `MIN_FLOOR_OBS` observations.
    On the live corpus most cells are unpriced, so a 50% win rate of unpriced
    signs is a direction to chase, never a result to report. The summary prints
    both rates and never collapses them into one.

The per-UNIT roll-up is printed beside the per-cell one because the house rule
is that sign tests run over UNITS: `dom1` and `loose1` are ONE model
byte-identically, and two cap levels in one campaign share a warm-up, so eight
cells can be four units. Cell counts answer "how consistent is the direction";
unit counts are the only ones a p-value may be computed from.

Everything is delegated to `deployed_h2h` -- its collector already carries the
quarantine gate, the completed-only filter, the verified-extension pooling and
the floor logic. Re-deriving any of that here would create a second copy free
to drift from the first.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import deployed_h2h                       # noqa: E402
from scripts import quarantine                         # noqa: E402
from scripts.floors import MIN_FLOOR_OBS               # noqa: E402
from scripts.paper_rows import MEASURED_UNITS          # noqa: E402

RIVALS = ("alm", "fioretto", "hounie")
BAR = 0.50


def _tp(rec):
    return float(rec["TP"])


def rows_for(cells, control):
    """One row per cell: the deltas vs control, the floor, and the verdict."""
    rows = []
    for key in sorted(cells):
        cell = cells[key]
        if "tralo" not in cell or control not in cell:
            continue
        order, _first = deployed_h2h.rank_cell(cell, control, _tp)
        d = dict((arm, mean) for arm, mean, _dl, _sd in order)
        if "tralo" not in d:
            continue
        floor, nfloor = deployed_h2h.rng_floor(cell, _tp)
        present = [r for r in RIVALS if r in d]
        spread = (max(d.values()) - min(d.values())) if len(d) > 1 else 0.0
        priced = (floor is not None and nfloor >= MIN_FLOOR_OBS
                  and spread > floor)
        beats_control = d["tralo"] > 0
        beats_all = all(d["tralo"] > d[r] for r in present)
        seeds = max((len(sd) for _a, _m, _dl, sd in order), default=0)
        rows.append(dict(
            campaign=key[0], model=key[1], dataset=key[2], cap=key[3],
            seeds=seeds, d=d, rivals=present, floor=floor, nfloor=nfloor,
            spread=spread, priced=priced,
            win=bool(present) and beats_control and beats_all,
            testable=bool(present),
            unit=MEASURED_UNITS.get((key[0], key[1])) or "UNVERIFIED"))
    return rows


def report(rows, out=sys.stdout, bar=BAR):
    w = out.write
    testable = [r for r in rows if r["testable"]]
    lonely = [r for r in rows if not r["testable"]]

    w("%-11s %-13s %-13s %4s %8s %8s %8s %8s %7s %6s %s\n"
      % ("campaign", "backbone", "cap", "sds", "tralo", "alm", "fioretto",
         "hounie", "floor", "priced", "verdict"))
    w("%s\n" % ("-" * 104))
    for r in sorted(rows, key=lambda r: (r["campaign"], r["model"], r["cap"])):
        def col(a):
            return ("%+8.2f" % r["d"][a]) if a in r["d"] else "       ."
        if not r["testable"]:
            verdict = "no rival -- cannot test"
        else:
            verdict = "WIN " if r["win"] else "loss"
        w("%-11s %-13s %-13s %4d %s %s %s %s %4s(%s) %6s %s\n"
          % (r["campaign"], r["model"][:13], r["cap"], r["seeds"],
             col("tralo"), col("alm"), col("fioretto"), col("hounie"),
             ("%.1f" % r["floor"]) if r["floor"] is not None else "none",
             r["nfloor"], "yes" if r["priced"] else "no", verdict))

    w("\n%s\n" % ("=" * 104))
    n = len(testable)
    wins = [r for r in testable if r["win"]]
    w("CELLS THAT CAN TEST THE CLAIM: %d  (%d more hold no rival and are "
      "excluded)\n" % (n, len(lonely)))
    if not n:
        w("VERDICT: NOT TESTABLE -- no cell holds tralo beside a rival dual.\n")
        return 1
    frac = len(wins) / float(n)
    w("  tralo beats the control AND every rival present: %d of %d = %.0f%%\n"
      % (len(wins), n, 100 * frac))

    priced = [r for r in testable if r["priced"]]
    pw = [r for r in priced if r["win"]]
    w("  ...of which PRICED (spread over a floor with >= %d observations): "
      "%d cell(s)" % (MIN_FLOOR_OBS, len(priced)))
    w(", tralo wins %d\n" % len(pw) if priced else "\n")
    if not priced:
        w("      NO cell is priced, so every WIN above is a DIRECTION, not a\n"
          "      result. Chase it; do not report it.\n")

    w("\n  per UNIT (the only axis a p-value may be computed over):\n")
    per = collections.defaultdict(list)
    for r in testable:
        per[r["unit"]].append(r)
    unit_wins = 0
    for u in sorted(per):
        g = per[u]
        k = sum(1 for r in g if r["win"])
        maj = k * 2 > len(g)
        unit_wins += 1 if maj else 0
        w("    %-11s %d of %d cells   %s\n"
          % (u, k, len(g), "TRALO" if maj else "rival"))
    w("    units where tralo takes the majority: %d of %d\n"
      % (unit_wins, len(per)))

    w("\n%s\n" % ("=" * 104))
    ok = frac >= bar
    w("VERDICT: %s -- tralo wins %.0f%% of testable cells, bar is %.0f%%\n"
      % ("PASS" if ok else "FAIL", 100 * frac, 100 * bar))
    if not ok:
        w("  The current TraLO does not clear the bar. Per the standing\n"
          "  instruction that is the trigger to change the METHOD, not to run\n"
          "  more seeds of it: more seeds sharpen an estimate, they do not\n"
          "  move a median that is already on the wrong side.\n")
    return 0 if ok else 1


def self_test(out=sys.stdout):
    """Gate the verdict in BOTH directions, and the denominator rule too."""
    checks = []

    def mk(spec, camp, cap):
        return (camp, "ViTB16", "iwildcam", cap, "2-7"), deployed_h2h._cell(spec)

    lead = {"clip": [600, 601, 599, 600], "tralo": [640, 641, 639, 640],
            "alm": [610, 611, 609, 610], "fioretto": [605, 606, 604, 605],
            "tralo_null": [600, 601, 599, 600],
            "tralo_reseed": [600, 601, 599, 600]}
    trail = dict(lead, tralo=[605, 606, 604, 605], alm=[640, 641, 639, 640])

    k1, c1 = mk(lead, "campA", "L80_G95")
    rows = rows_for({k1: c1}, "clip")
    checks.append(("a cell where tralo leads every rival is a WIN",
                   len(rows) == 1 and rows[0]["win"]))

    k2, c2 = mk(trail, "campA", "L90_G95")
    rows = rows_for({k2: c2}, "clip")
    checks.append(("NEGATIVE CONTROL: a cell where a rival leads is a LOSS",
                   len(rows) == 1 and not rows[0]["win"]))

    # Beating the control but NOT the rival must not count as a win: the bar is
    # both, and this is the case the old "tralo vs clip" framing scored green.
    mid = dict(lead, tralo=[620, 621, 619, 620], alm=[640, 641, 639, 640])
    k3, c3 = mk(mid, "campA", "L95_G80")
    rows = rows_for({k3: c3}, "clip")
    checks.append(("beating the CONTROL but not the RIVAL is NOT a win",
                   len(rows) == 1 and not rows[0]["win"]
                   and rows[0]["d"]["tralo"] > 0))

    # A cell with no rival must leave the denominator untouched.
    solo = {"clip": [600, 601, 599, 600], "tralo": [640, 641, 639, 640],
            "tralo_null": [600, 601, 599, 600],
            "tralo_reseed": [600, 601, 599, 600]}
    k4, c4 = mk(solo, "campB", "L80_G95")
    rows = rows_for({k1: c1, k4: c4}, "clip")
    testable = [r for r in rows if r["testable"]]
    checks.append(("a cell with NO rival is excluded from the denominator",
                   len(rows) == 2 and len(testable) == 1))

    import io as _io
    buf = _io.StringIO()
    rc = report(rows_for({k1: c1}, "clip"), out=buf)
    checks.append(("an all-win table reports PASS",
                   rc == 0 and "VERDICT: PASS" in buf.getvalue()))
    buf = _io.StringIO()
    rc = report(rows_for({k2: c2}, "clip"), out=buf)
    checks.append(("NEGATIVE CONTROL: an all-loss table reports FAIL",
                   rc == 1 and "VERDICT: FAIL" in buf.getvalue()))
    buf = _io.StringIO()
    rc = report(rows_for({k1: c1, k2: c2}, "clip"), out=buf)
    checks.append(("exactly 50% PASSES -- the bar is 'at least'",
                   rc == 0 and "VERDICT: PASS" in buf.getvalue()))

    print("", file=out)
    for label, good in checks:
        print("  %-70s %s" % (label[:70], "PASS" if good else "FAIL"), file=out)
    bad = [c for c, g in checks if not g]
    print("", file=out)
    print("ALL PASS" if not bad else "FAILED: %d" % len(bad), file=out)
    return 1 if bad else 0


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--campaign", nargs="+", default=[])
    a.add_argument("--control", default="clip")
    a.add_argument("--bar", type=float, default=BAR)
    a.add_argument("--allow-quarantined", action="store_true")
    a.add_argument("--self-test", action="store_true")
    args = a.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.campaign:
        a.error("give --campaign <root> ... (or --self-test)")
    blocked, dead = quarantine.gate(args.campaign, args.allow_quarantined,
                                    "score")
    if blocked:
        return 1
    cells = deployed_h2h.collect(args.campaign, dead)
    return report(rows_for(cells, args.control), bar=args.bar)


if __name__ == "__main__":
    sys.exit(main())
