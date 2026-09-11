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
  * **A CELL THAT NEVER REACHED THE TABLE IS NAMED, NOT SWALLOWED.** Three
    conditions drop a cell before it can be scored -- no `tralo`, no control,
    or `rank_cell` finding the two share no common seed -- and until
    2026-09-11 all three were a bare `continue`. The reader then sees
    "testable + no rival" and takes it for the whole input, which is the
    silent-truncation failure the house rule forbids: it reads as "covered
    everything" when it did not. Each dropped cell is now printed with its
    reason, before the verdict and before the early return.
  * **A CELL THAT POSES NO CAP QUESTION IS NOT IN THE DENOMINATOR EITHER**
    (2026-09-11). `quarantine.gate()` prints "N OF M CELLS DO NOT POSE THE CAP
    QUESTION" directly above this table, and until today `testable` was
    `bool(present)` -- rival staged, full stop -- so the cells it had just
    named went into a count headed "CELLS THAT CAN TEST THE CLAIM". The
    announcement and the arithmetic were in different modules and never spoke.
    Caught on `bcn1vit`, whose L70 and L80 are both measured `non_task`: the
    verdict read "PASS -- 67% of testable cells" over a denominator of 3 in
    which only ONE cell asked anything. `non_task` is a MEASURED absence and
    `unmeasured`/`no_window`/`no_data` are absences of MEASUREMENT; both are
    excluded, both are named, and neither is a null.

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

# 🛑 THE SECOND BAR, AND IT IS THE PROJECT'S OWN GOAL SENTENCE (2026-09-11).
# CLAUDE.md line 1: "beat a post-hoc clipping baseline". There are TWO in every
# campaign -- `gen_campaign` always adds both -- and this table scored against
# exactly one of them, `--control clip`, while `focal_clip` sat in the same cell
# at EQUAL compute (warm-up 30 / constraint 0, the same 30 optimizer epochs)
# and was never compared to anything.
#
# ⛔ THE JUSTIFICATION FOR IGNORING IT WAS A CLAIM, AND THE CLAIM IS FALSE.
# CLAUDE.md rule 2 says "`clip` is the stronger quality bar", which would make
# beating `clip` sufficient. Measured over all 33 scored cells in the eleven
# licensed-unit campaigns: `focal_clip` beats `clip` in ~20 of 33, and it beats
# `tralo` in **12 of 33**. One of those twelve is `bcn2rgn`/L100_G95, a cell
# this table counts as a TraLO WIN while `focal_clip` leads it by 11.5 items.
#
# ⚠️ THIS IS REPORTED BESIDE THE OLD FIGURE, NEVER INSTEAD OF IT. Changing an
# acceptance rule so that it moves against the method is still changing the
# rule, and 2(z108) is the entry about a number that only ever moved one way.
# Both denominators are printed; the reader picks, and the restriction travels
# with the figure exactly as 2(z66) requires.
CLIPPERS = ("focal_clip",)

# 🛑 THE CELL STATUSES THAT POSE THE CAP QUESTION, AND THE ONLY ONES THAT MAY
# ENTER THE ACCEPTANCE DENOMINATOR (2026-09-11).
#   `task`     the cap binds in EVERY seed
#   `partial`  it binds in SOME seeds, so slack seeds dilute the contrast
#              toward nothing. A positive here is CONSERVATIVE and a null is
#              weak, which is why it counts but is also sub-tallied separately.
# Everything else is excluded, and the two reasons are NOT the same thing:
#   `non_task`                         a MEASURED statement -- the cap evicts
#                                      too little, or there are no errors
#                                      inside K, or p@K is at the ceiling
#   `unmeasured` / `no_window` /       an ABSENCE of measurement
#   `no_strict_band` / `no_data`
# Neither is a null, and a contrast on either is not evidence about the
# constraint -- `quarantine.gate()` has printed exactly that sentence above
# this table since 2026-09-04.
POSES = ("task", "partial")


def statuses_for(roots):
    """`{(dataset, model, cap): status}` merged over `roots`.

    Delegates to `quarantine.cell_status`, which already loads `protocol.yml`
    and `task_windows.yml` and already makes the fail-closed distinction this
    needs. Re-deriving it here would be the `cellreport.py` defect again -- a
    second copy free to drift from the first.

    A root whose worktree PREDATES `configs/task_cells.py` returns None, not an
    empty dict, and those cells stay ABSENT from the map. Absent is read as
    "poses the question" by the caller, on purpose: excluding a cell the
    instrument could not read would shrink the acceptance denominator on a
    version-skew failure, which is the direction that manufactures a PASS.
    """
    out = {}
    for root in roots:
        got = quarantine.cell_status(root)
        if got:
            out.update(got)
    return out


def _tp(rec):
    return float(rec["TP"])


def rows_for(cells, control, dropped=None, statuses=None):
    """One row per cell: the deltas vs control, the floor, and the verdict.

    `dropped` is an optional out-list. Pass one and every cell that cannot
    reach the table is appended as `(key, reason)` instead of vanishing. It is
    optional rather than returned so no existing caller changes shape, but
    `main` always passes one -- a denominator nobody can audit is the defect.

    `statuses` is `statuses_for(roots)`. Pass it and a cell measured NOT to
    pose the cap question is kept in the table but taken OUT of the acceptance
    denominator. Omit it and every cell counts, which is the pre-2026-09-11
    behaviour and is retained only so existing callers do not change shape.
    """
    rows = []
    for key in sorted(cells):
        cell = cells[key]
        if "tralo" not in cell or control not in cell:
            if dropped is not None:
                dropped.append((key, "no tralo" if "tralo" not in cell
                                else "no %s" % control))
            continue
        order, _first = deployed_h2h.rank_cell(cell, control, _tp)
        d = dict((arm, mean) for arm, mean, _dl, _sd in order)
        if "tralo" not in d:
            # NOT the same case as "no tralo": the arm RAN, and `rank_cell`
            # dropped it for sharing no seed with the control (2(z50)). From
            # outside the two look identical, and the remedies are opposite.
            if dropped is not None:
                dropped.append((key, "tralo shares no seed with %s" % control))
            continue
        floor, nfloor, _nstream = deployed_h2h.rng_floor(cell, _tp)
        present = [r for r in RIVALS if r in d]
        # 🛑 THE MARGIN IS PAIRWISE, NEVER `max - min` (fixed 2026-09-06).
        # A RANGE over k arms grows like `sd*sqrt(2 ln k)` -- ~3.1*sd at k=10 --
        # while the RNG floor it is compared against is a TWO-arm quantity at
        # 1.13*sd, so `range >= floor` certifies pure noise as differentiated at
        # ~2.7x. Measured on the corpus: raw range/floor reads a healthy median
        # 2.51 over 50 cells and the SAME cells read 0.97 once corrected.
        #
        # This was live here and `deployed_h2h`'s de-whitelisting made it WORSE
        # by putting more arms in `d`. The claim being priced is a claim about
        # TWO margins, so price exactly those two:
        #   vs the control  -- `d["tralo"]`, already a delta against it
        #   vs the best rival present
        # and require the NARROWER of them to clear the floor, since a win needs
        # both. `abs` so a priced LOSS is reported as one, not dropped.
        margins = [abs(d["tralo"])]
        if present:
            margins.append(abs(d["tralo"] - max(d[r] for r in present)))
        spread = min(margins)
        priced = (floor is not None and nfloor >= MIN_FLOOR_OBS
                  and spread > floor)
        beats_control = d["tralo"] > 0
        beats_all = all(d["tralo"] > d[r] for r in present)
        # The second post-hoc baseline, present in the same cell at equal
        # compute. Absent from `d` means the arm was not staged, which is not
        # evidence that TraLO beat it -- so the strict verdict is only defined
        # where it actually ran.
        clippers = [c for c in CLIPPERS if c in d]
        beats_clippers = all(d["tralo"] > d[c] for c in clippers)
        # `rank_cell` hands every arm the SAME common seed list (2(z50)), so a
        # max over arms is correct here only BY INHERITANCE. Assert the contract
        # instead of trusting it: if that return shape ever changes, this must
        # fail loudly rather than revert in silence to the max-over-arms form
        # that reported "3 seeds" for a vitdual2 cell whose comparisons rested
        # on one. This is the acceptance table -- it is the last place a ragged
        # seed set should be allowed to pass unnoticed.
        seedsets = {tuple(sd) for _a, _m, _dl, sd in order}
        assert len(seedsets) <= 1, (
            "rank_cell returned ragged seed sets %s for cell %s: the arm-vs-arm "
            "margins in it are not comparable" % (sorted(seedsets), key))
        seeds = len(next(iter(seedsets))) if seedsets else 0
        # 🛑 A CELL THAT POSES NO QUESTION IS NOT A TESTABLE CELL, AND UNTIL
        # 2026-09-11 THIS TABLE COUNTED IT AS ONE. `quarantine.gate()` prints
        # "N OF M CELLS DO NOT POSE THE CAP QUESTION" immediately above this
        # output and then `testable` was `bool(present)` -- rival present, full
        # stop -- so the very cells the gate had just named went straight into
        # the denominator of a line headed "CELLS THAT CAN TEST THE CLAIM".
        # The announcement and the arithmetic lived in different modules and
        # never spoke. Found on `bcn1vit`, where 2 of 3 cells are measured
        # `non_task` and the verdict read "PASS -- 67% of testable cells".
        # `status is None` means the instrument could not classify this cell --
        # see `statuses_for` -- and is deliberately read as POSING.
        status = (statuses or {}).get((key[2], key[1], key[3]))
        poses = status is None or status in POSES
        rows.append(dict(
            campaign=key[0], model=key[1], dataset=key[2], cap=key[3],
            seeds=seeds, d=d, rivals=present, floor=floor, nfloor=nfloor,
            spread=spread, priced=priced, status=status, poses=poses,
            win=bool(present) and beats_control and beats_all,
            clippers=clippers,
            win_strict=(bool(present) and beats_control and beats_all
                        and beats_clippers),
            testable=bool(present) and poses,
            unit=MEASURED_UNITS.get((key[0], key[1])) or "UNVERIFIED"))
    return rows


def report(rows, out=sys.stdout, bar=BAR, dropped=None):
    w = out.write
    testable = [r for r in rows if r["testable"]]
    # TWO exclusion reasons, kept apart because the remedies are opposite:
    # `lonely` needs a rival STAGED, `noq` needs a different CAP. Collapsing
    # them into one "excluded" count is how the second stayed invisible.
    lonely = [r for r in rows if not r["testable"] and not r["rivals"]]
    noq = [r for r in rows if not r["testable"] and r["rivals"]]

    w("%-11s %-13s %-13s %4s %8s %8s %8s %8s %7s %6s %s\n"
      % ("campaign", "backbone", "cap", "sds", "tralo", "alm", "fioretto",
         "hounie", "floor", "priced", "verdict"))
    w("%s\n" % ("-" * 104))
    for r in sorted(rows, key=lambda r: (r["campaign"], r["model"], r["cap"])):
        def col(a):
            return ("%+8.2f" % r["d"][a]) if a in r["d"] else "       ."
        if not r["rivals"]:
            verdict = "no rival -- cannot test"
        elif not r["poses"]:
            verdict = "%s -- poses no question" % (r["status"] or "?")
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
    w("CELLS THAT CAN TEST THE CLAIM: %d\n" % n)
    w("  excluded: %d hold no rival, %d pose no cap question\n"
      % (len(lonely), len(noq)))
    for r in sorted(noq, key=lambda r: (r["campaign"], r["model"], r["cap"])):
        w("       %-11s %-13s %-13s  %s\n"
          % (r["campaign"], r["model"][:13], r["cap"], r["status"]))
    if dropped:
        w("  !! %d cell(s) never reached the table and are in NEITHER count "
          "above:\n" % len(dropped))
        for key, why in dropped:
            w("       %-11s %-13s %-13s  %s\n"
              % (key[0], str(key[1])[:13], key[3], why))
    if not n:
        # Say WHICH of the two emptied it. "No rival was staged" and "every
        # cell sits outside the measured cap window" are opposite findings:
        # the first is a coverage hole, the second is `uniform1`/`vittask1`,
        # mechanically perfect campaigns that measured the absence of a
        # question (FRAMEWORK 2(z42)).
        if noq and not lonely:
            w("VERDICT: NOT TESTABLE -- every cell holding a rival poses NO\n"
              "  cap question. The campaign is not silent about TraLO; it\n"
              "  never asked. Re-stage inside the measured window.\n")
        else:
            w("VERDICT: NOT TESTABLE -- no cell holds tralo beside a rival "
              "dual.\n")
        return 1
    frac = len(wins) / float(n)
    w("  tralo beats the control AND every rival present: %d of %d = %.0f%%\n"
      % (len(wins), n, 100 * frac))
    # The STRICT sub-tally. `partial` cells bind in some seeds only, so their
    # slack seeds take an identically zero constraint gradient and dilute the
    # contrast -- a positive there is conservative, a null is weak. Printed
    # beside the headline rather than replacing it, because which one to quote
    # depends on the claim and neither is the other's caveat.
    strict = [r for r in testable if r["status"] == "task"]
    if strict and len(strict) != n:
        sw = sum(1 for r in strict if r["win"])
        w("    ...restricted to STRICT `task` cells: %d of %d = %.0f%%\n"
          % (sw, len(strict), 100 * sw / float(len(strict))))

    # THE SECOND BAR, PRINTED BESIDE THE FIRST AND NEVER INSTEAD OF IT.
    # `focal_clip` is the other post-hoc clipping baseline, sitting in the same
    # cell at equal compute, and until 2026-09-11 nothing compared TraLO to it.
    # See CLIPPERS above for the measurement that removed the excuse.
    withclip = [r for r in testable if r["clippers"]]
    if withclip:
        sc = [r for r in withclip if r["win_strict"]]
        w("    ...AND ALSO beating `%s`, the other post-hoc baseline at "
          "equal compute: %d of %d = %.0f%%\n"
          % ("`, `".join(CLIPPERS), len(sc), len(withclip),
             100 * len(sc) / float(len(withclip))))
        lost = [r for r in withclip if r["win"] and not r["win_strict"]]
        if lost:
            w("       %d cell(s) counted as a WIN above are led by an arm "
              "taking ZERO constraint steps:\n" % len(lost))
            for r in sorted(lost, key=lambda x: (x["campaign"], x["cap"])):
                best = max(r["clippers"], key=lambda c: r["d"][c])
                w("         %-11s %-13s %-13s tralo %+.2f vs %s %+.2f\n"
                  % (r["campaign"], r["model"], r["cap"], r["d"]["tralo"],
                     best, r["d"][best]))
    if len(withclip) < len(testable):
        w("       (%d testable cell(s) staged no second clipper and are "
          "absent\n        from that denominator -- not evidence either "
          "way)\n" % (len(testable) - len(withclip)))

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
    import io as _io
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

    # THE SECOND POST-HOC BASELINE. `focal_clip` sits in every campaign at
    # equal compute and nothing compared TraLO to it until 2026-09-11. The
    # middle case is the whole point: a cell that IS a win under the old rule
    # while an arm taking ZERO constraint steps leads it (measured on
    # bcn2rgn/L100_G95, tralo +2.25 against focal_clip +13.75).
    fc_below = dict(lead, focal_clip=[615, 616, 614, 615])   # tralo still 1st
    k5, c5 = mk(fc_below, "campC", "L80_G95")
    rows = rows_for({k5: c5}, "clip")
    checks.append(("tralo ahead of the second clipper too: win AND win_strict",
                   len(rows) == 1 and rows[0]["win"] and rows[0]["win_strict"]))

    fc_above = dict(lead, focal_clip=[660, 661, 659, 660])   # focal_clip 1st
    k6, c6 = mk(fc_above, "campC", "L90_G95")
    rows = rows_for({k6: c6}, "clip")
    checks.append(("NEGATIVE CONTROL: a cell led by `focal_clip` is still a "
                   "win under the OLD rule",
                   len(rows) == 1 and rows[0]["win"]))
    checks.append(("...and is NOT a win_strict -- the bar that reads the "
                   "project's own goal sentence",
                   len(rows) == 1 and not rows[0]["win_strict"]))

    buf = _io.StringIO()
    report(rows_for({k5: c5, k6: c6}, "clip"), out=buf)
    txt = buf.getvalue()
    checks.append(("the strict-clipper tally is PRINTED beside the headline",
                   "the other post-hoc baseline" in txt and "1 of 2" in txt))
    # Assert the DEMOTED LINE itself, not just that the words occur somewhere:
    # `campC` also appears in the main table and "ZERO constraint steps" is a
    # header, so a check for either is satisfied while the naming is gone.
    demoted = [l for l in txt.splitlines()
               if "campC" in l and "tralo +" in l and "focal_clip +" in l]
    checks.append(("the demoted cell is NAMED on one line with BOTH deltas",
                   len(demoted) == 1 and "ZERO constraint steps" in txt))
    # ...and the two numbers must DIFFER with the clipper AHEAD, which is the
    # whole reason the cell was demoted. Printing tralo's delta in both slots
    # renders the line self-consistent and meaningless, and reads as a tie.
    nums = ([float(t) for t in demoted[0].replace("+", " +").split()
             if t.lstrip("+-").replace(".", "", 1).isdigit()] if demoted else [])
    checks.append(("...with the CLIPPER's delta ahead of tralo's on that line",
                   len(nums) == 2 and nums[1] > nums[0]))

    # A cell staging NO second clipper must be ABSENT from that denominator
    # rather than counted as a pass -- absence of the arm is not evidence that
    # TraLO beat it. Same one-sidedness rule as md5 (2(x2)).
    rows = rows_for({k1: c1, k5: c5}, "clip")
    checks.append(("a cell with no second clipper is absent from the strict "
                   "denominator, not counted as a pass",
                   [r["clippers"] for r in rows] == [[], ["focal_clip"]]))
    buf = _io.StringIO()
    report(rows, out=buf)
    checks.append(("...and the report SAYS how many were absent",
                   "staged no second clipper" in buf.getvalue()))

    # A cell with no rival must leave the denominator untouched.
    solo = {"clip": [600, 601, 599, 600], "tralo": [640, 641, 639, 640],
            "tralo_null": [600, 601, 599, 600],
            "tralo_reseed": [600, 601, 599, 600]}
    k4, c4 = mk(solo, "campB", "L80_G95")
    rows = rows_for({k1: c1, k4: c4}, "clip")
    testable = [r for r in rows if r["testable"]]
    checks.append(("a cell with NO rival is excluded from the denominator",
                   len(rows) == 2 and len(testable) == 1))

    # THE PAIRWISE-MARGIN GATE (2026-09-06). A far-behind arm must not be able
    # to price a cell. `fioretto` sits 80 items back, so the RANGE is wide while
    # tralo's real margin over its best rival is 2 items. A range-based
    # `spread` -- what this scorer used until today -- prices it; the pairwise
    # one must not. The check asserts BOTH that the value is the pairwise one
    # and that the two genuinely differ here, so it cannot pass vacuously.
    wide = {"clip": [600, 600, 600, 600], "tralo": [641, 640, 639, 640],
            "alm": [639, 638, 637, 638], "fioretto": [560, 561, 559, 560],
            "tralo_null": [600, 600, 600, 600],
            "tralo_reseed": [608, 592, 604, 596]}
    k5, c5 = mk(wide, "campC", "L70_G95")
    r5 = rows_for({k5: c5}, "clip")[0]
    rng = max(r5["d"].values()) - min(r5["d"].values())
    pair = min(abs(r5["d"]["tralo"]),
               abs(r5["d"]["tralo"] - max(r5["d"][x] for x in r5["rivals"])))
    checks.append(("the margin is the PAIRWISE %.1f, not the range %.1f"
                   % (pair, rng),
                   abs(r5["spread"] - pair) < 1e-9 and rng > pair + 1.0))

    # NEGATIVE CONTROL: with tralo and ONE rival and nothing trailing, the range
    # and the pairwise margin coincide, so a passing check above must not be an
    # artefact of always returning the smaller of two numbers.
    tight = {"clip": [600, 600, 600, 600], "tralo": [641, 640, 639, 640],
             "alm": [639, 638, 637, 638],
             "tralo_null": [600, 600, 600, 600],
             "tralo_reseed": [608, 592, 604, 596]}
    k6, c6 = mk(tight, "campC", "L80_G95")
    r6 = rows_for({k6: c6}, "clip")[0]
    rng6 = max(r6["d"].values()) - min(r6["d"].values())
    checks.append(("NEGATIVE CONTROL: with no trailing arm the two agree",
                   abs(r6["spread"] - rng6) < 1e-9))

    # 🛑 THE CAP-QUESTION GATE (2026-09-11). A cell holding a rival but
    # MEASURED not to pose the cap question must leave the denominator. The
    # three controls pin that it is the STATUS doing the work and not the
    # fixture: the SAME cell is testable at `task` and at `partial`, and an
    # UNCLASSIFIABLE cell stays testable -- the fail-OPEN direction, because
    # dropping a cell the instrument could not read would shrink the
    # denominator on version skew, which is the direction that manufactures a
    # PASS. `win` stays True throughout: the row is still computed, it is
    # merely not counted, and conflating those two is the original defect.
    sk = (k1[2], k1[1], k1[3])
    rnt = rows_for({k1: c1}, "clip", statuses={sk: "non_task"})
    checks.append(("a `non_task` cell is OUT of the denominator",
                   len(rnt) == 1 and not rnt[0]["testable"] and rnt[0]["win"]))
    checks.append(("NEGATIVE CONTROL: the SAME cell at `task` is IN",
                   rows_for({k1: c1}, "clip",
                            statuses={sk: "task"})[0]["testable"]))
    checks.append(("NEGATIVE CONTROL: `partial` poses the question, so it "
                   "counts toward the headline",
                   rows_for({k1: c1}, "clip",
                            statuses={sk: "partial"})[0]["testable"]))
    checks.append(("NEGATIVE CONTROL: an UNCLASSIFIABLE cell stays IN, so "
                   "version skew cannot manufacture a PASS",
                   rows_for({k1: c1}, "clip", statuses={})[0]["testable"]))
    # And the absence of `statuses` must reproduce the old shape exactly, or
    # every existing caller silently changes meaning.
    checks.append(("NEGATIVE CONTROL: with no statuses at all, nothing is "
                   "excluded", rows_for({k1: c1}, "clip")[0]["testable"]))

    import io as _io
    # The exclusion must be VISIBLE, not merely correct: a denominator that
    # quietly shrinks reads as "covered everything" exactly like the bare
    # `continue` this table was fixed for on the same day.
    buf = _io.StringIO()
    report(rows_for({k1: c1, k2: c2}, "clip",
                    statuses={sk: "non_task"}), out=buf)
    txt = buf.getvalue()
    # SCOPED to the excluded block, not the whole output. The per-row verdict
    # column also prints the status, so a bare `"non_task" in txt` is
    # satisfied by the table above and stays green even with this block
    # silenced -- measured, it was 0 of 4 mutations caught until this slice
    # was added.
    seg = txt.split("pose no cap question", 1)[-1].split("tralo beats", 1)[0]
    checks.append(("the excluded cell is NAMED with its status IN the "
                   "excluded block, not silently dropped",
                   "pose no cap question" in txt and "non_task" in seg))
    # The STRICT sub-tally appears only when the testable set MIXES `task` and
    # `partial`; with one status it would restate the headline.
    buf = _io.StringIO()
    report(rows_for({k1: c1, k2: c2}, "clip",
                    statuses={sk: "task", (k2[2], k2[1], k2[3]): "partial"}),
           out=buf)
    checks.append(("a MIXED task/partial table prints the strict sub-tally",
                   "restricted to STRICT" in buf.getvalue()))
    buf = _io.StringIO()
    report(rows_for({k1: c1, k2: c2}, "clip",
                    statuses={sk: "task", (k2[2], k2[1], k2[3]): "task"}),
           out=buf)
    checks.append(("NEGATIVE CONTROL: an all-`task` table does NOT print it",
                   "restricted to STRICT" not in buf.getvalue()))
    # And a campaign whose every rival-bearing cell poses no question must say
    # which of the two emptied the table -- `uniform1` and `vittask1` were
    # mechanically perfect and measured the absence of a question (2(z42)).
    buf = _io.StringIO()
    rc = report(rows_for({k1: c1}, "clip", statuses={sk: "non_task"}), out=buf)
    checks.append(("NOT TESTABLE names the cap window, not a missing rival",
                   rc == 1 and "never asked" in buf.getvalue()))

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

    # THE DENOMINATOR IS AUDITABLE. Until 2026-09-11 all three drop paths
    # were a bare `continue`, so a cell that could not be scored appeared in
    # NEITHER printed count and the reader took `testable + lonely` for the
    # whole input. These three gate the fix and its negative control.
    nocontrol = dict(lead)
    del nocontrol["clip"]
    k4, c4 = mk(nocontrol, "campB", "L80_G95")
    drops = []
    rows = rows_for({k4: c4}, "clip", drops)
    checks.append(("a cell with no control is DROPPED and the reason names it",
                   not rows and len(drops) == 1 and "no clip" in drops[0][1]))

    # THE SECOND DROP PATH, and it is deliberately a different reason: the arm
    # RAN, and `rank_cell` dropped it for sharing no seed with the control
    # (2(z50)). From outside the two look identical and the remedies are
    # opposite -- stage the arm, versus re-run it on the control's seeds.
    ragged = deployed_h2h._ragged({"clip": {3: 600, 4: 601},
                                   "tralo": {1: 640, 2: 641},
                                   "alm": {3: 610, 4: 611}})
    k5 = ("campC", "ViTB16", "iwildcam", "L80_G95", "2-7")
    drops2 = []
    rows = rows_for({k5: ragged}, "clip", drops2)
    checks.append(("a cell whose tralo shares NO seed with the control is "
                   "dropped, for a DIFFERENT reason",
                   not rows and len(drops2) == 1
                   and "shares no seed" in drops2[0][1]))

    ok_drops = []
    rows_for({k1: c1}, "clip", ok_drops)
    checks.append(("NEGATIVE CONTROL: a complete cell is NOT dropped",
                   ok_drops == []))

    buf = _io.StringIO()
    report(rows_for({k1: c1}, "clip"), out=buf, dropped=drops)
    checks.append(("the report NAMES the dropped cell, above the verdict",
                   "never reached the table" in buf.getvalue()
                   and "campB" in buf.getvalue()))

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
    dropped = []
    rows = rows_for(cells, args.control, dropped,
                    statuses=statuses_for(args.campaign))
    return report(rows, bar=args.bar, dropped=dropped)


if __name__ == "__main__":
    sys.exit(main())
