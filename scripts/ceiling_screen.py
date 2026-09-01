"""Is there anything to WIN on this dataset? Ask before spending a campaign.

`dataset_screen` asks whether the count constraint carries INFORMATION here.
This asks the other half, and the two are independent: even where the counts
are informative, the PRIZE can be zero.

THE ARITHMETIC, and it is exact.

When a class has `n` true test instances and the budget lets you emit `K`
predictions for it, the best achievable cc-F1 is

    ceiling = 2K / (K + n)          precision 1, recall K/n

because you cannot recall what you may not emit. A model whose precision at the
cut is `p` achieves `2pK / (K + n)`, so the ENTIRE prize, for any method, is

    prize_items = (1 - p) * K

and nothing about the loss, the dual, the allocator or the optimizer changes
that bound. It is set by `K` and by how good the ranking already is.

WHY IT BITES. `p` is not independent of `K/n`. When the budget is a small
fraction of the true positives, the top-K set is drawn from a pool several
times its own size and is filled with correct items almost by default.

🛑 AND THE NOISE MOVES THE SAME WAY, which is why a prize alone decides
nothing. A looser cap buys headroom AND cuts deeper into the contested middle,
so the seed sd rises with it. Both measured on `results/iwc3`, 36 `clip` runs,
9 cells, 4 seeds, pooled over the two capped classes:

    K/n     p@K      prize   unpaired  PAIRED sd   prize/PAIRED
    20%   0.9972      0.21      0.40       4.75      0.04x   <- L20, protocol
    30%   0.9948      0.58      0.98       6.35      0.09x   <- L30/L50, protocol
    50%   0.9881      2.20      3.47      11.12      0.20x
    70%   0.9722      7.24      7.43      17.86      0.41x
    80%   0.9544     13.50      9.66      24.95      0.54x
    90%   0.9252     24.94     13.18      27.83      0.90x

⚠️ THE `unpaired` COLUMN IS THE WRONG NOISE FOR THIS PROJECT and this file
quoted it first. Every contrast here is seed-PAIRED against the arm's own
lambda=0 twin -- and pairing does not shrink the noise, it GROWS it, because
`tralo` and `tralo_null` share one warm-up epoch and then train 29 apart. They
are two models, not two readings of one. The screen prices against `PAIRED`.

So the honest statement about iwildcam is NOT "there is no prize". It is that
**at every cap this protocol sweeps the whole gap to a perfect ranking is
smaller than the seed noise**, so a method capturing 100% of it would still not
be detectable at 4 seeds. The ratio is monotone in `K/n` and crosses 1.0 only
above ~70%, where the budget admits most of the true positives and the
constraint barely constrains. FRAMEWORK 2(v).

A dataset is worth a campaign where `(1-p@K)*K` is a comfortable multiple of
the seed sd AT THAT `K/n`. `K` needs LABELS and the CAP POLICY and nothing else
-- no images, no model, no GPU -- which is why this runs before a download;
`p@K` and the sd need a model, so the built-in curve is iwildcam's and is a
guide to WHERE to look, never a substitute for measuring them.

The budgets are not re-derived here. `compute_local_constraints` and
`compute_global_constraints` are imported from the training path, and the
binding budget is `effective_budget`'s `min(global, sum of local ceilings)`,
which is the rule that a 30x headroom bug was fixed to.

    python -m scripts.ceiling_screen data/iwildcam/oodslice --caps L20_G50 L30_G50 L50_G30 --classes 2 7
    python -m scripts.ceiling_screen --self-test
"""

import argparse
import os
import sys

SEED_NOISE_ITEMS = 2.11          # iwc3, paired within-cell sd. FRAMEWORK 2(p-post)
MEASURED_CCP = 0.9954            # iwc3, `tralo_null` against `clip`

# 🛑 A FIXED p IS WRONG, AND IT WAS WRONG HERE FIRST. `p` is precision at the
# cut, and it FALLS as the budget grows -- a bigger K reaches further down the
# ranking. Holding it at 0.9954 said "iwildcam has no prize at any cap", when
# the truth is narrower and more useful: it has no prize at the caps this
# protocol sweeps.
#
# And the noise moves too, in the same direction, which is 2(i): a looser cap
# cuts deeper into the contested middle. Quoting a prize against a fixed sd
# measured at L20-L50 overstates it by up to 6x.
#
# Both curves, measured on `results/iwc3`, 36 `clip` runs, 9 cells, 4 seeds,
# pooled over the two capped classes. `sd` is the within-cell sd of TP@K across
# seeds, in items. FRAMEWORK 2(v).
#
# `sd` is the PAIRED within-cell sd of `tralo` - `tralo_null` in TP@K items,
# which is the noise the contrast this project runs actually faces. The
# unpaired figure is 6-12x SMALLER and pricing against it overstates every
# ratio; it is kept in the header table only so the two are never confused.
#
#            K/n     p@K      sd(items, PAIRED)
IWILDCAM_CURVE = [
    (0.20, 0.9972, 4.75),
    (0.30, 0.9948, 6.35),
    (0.40, 0.9923, 8.74),
    (0.50, 0.9881, 11.12),
    (0.60, 0.9817, 14.49),
    (0.70, 0.9722, 17.86),
    (0.80, 0.9544, 24.95),
    (0.90, 0.9252, 27.83),
]


def calibrated(ratio, curve=IWILDCAM_CURVE):
    """(p, sd_items, extrapolated) at this K/n, linearly interpolated.

    ⚠️ THIS IS AN iwildcam CALIBRATION AND IT DOES NOT TRANSFER. It is here so
    the screen stops pretending p is constant, not so a candidate dataset can
    be priced without measuring its own. On a new dataset use it to see WHERE
    the ratio might become measurable, then measure p and sd there.

    🛑 THE THIRD RETURN IS THE CLAMP, AND IT USED TO BE SILENT. Outside
    K/n 0.20-0.90 this returns the nearest ENDPOINT, which is not a
    measurement of anything -- and the per-class caps now in use run to
    K/n = 1.00 (`L80-100_G95`), so the clamp fires in the live protocol, not
    in some corner. Callers must say when they are reading it.
    """
    if ratio <= curve[0][0]:
        return curve[0][1], curve[0][2], ratio < curve[0][0]
    if ratio >= curve[-1][0]:
        return curve[-1][1], curve[-1][2], ratio > curve[-1][0]
    for (r0, p0, s0), (r1, p1, s1) in zip(curve, curve[1:]):
        if r0 <= ratio <= r1:
            w = (ratio - r0) / (r1 - r0)
            return p0 + w * (p1 - p0), s0 + w * (s1 - s0), False
    return curve[-1][1], curve[-1][2], True


def budgets(meta_path, caps, classes, group_col, num_classes):
    """[(cap, class, n, k_global, k_local_sum, k_binding)], from labels alone."""
    import pandas as pd

    from configs.gen_campaign import cap_pair
    from scripts.full_panel import effective_budget
    from src.training.constraints import (compute_global_constraints,
                                          compute_local_constraints)

    te = pd.read_csv(meta_path)
    out = []
    for tag in caps:
        local_pct, global_pct = cap_pair(tag)
        G = compute_global_constraints(te, "label", global_pct, classes,
                                       num_classes=num_classes)
        L = compute_local_constraints(te, "label", local_pct, group_col,
                                      constrained_class=classes,
                                      num_classes=num_classes)
        for c in classes:
            n = int((te["label"] == c).sum())
            k_loc = sum(int(b[c]) for b in L.values())
            out.append((tag, c, n, int(G[c]), k_loc,
                        int(effective_budget(G, L, c))))
    return out


def report(rows, ccp=None, noise=None, out=sys.stdout, native=True):
    """Print the table. Returns the number of (cap, class) cells worth running.

    `ccp` and `noise` override the K/n-dependent calibration with constants --
    for the self-test, and for a dataset whose own p and sd have been measured.

    🛑 `native=False` says the built-in curve is BORROWED: this is not
    iwildcam and its own p@K and sd have never been measured. The verdict
    column then cannot kill, because the kill would be iwildcam's. It reports
    the p@K this dataset would NEED instead, which is the number to go and get
    -- FRAMEWORK 2(w2) prices fmow at `p@K <= 0.92` on exactly this arithmetic.
    """
    borrowed = (not native) and ccp is None and noise is None
    out.write("CEILING SCREEN -- how many items can ANY method win here?\n")
    out.write("  ceiling = 2K/(K+n): you cannot recall what you may not emit.\n")
    out.write("  prize   = (1-p@K)*K items. No loss, dual, allocator or\n"
              "            optimizer changes this bound.\n")
    out.write("  BOTH p@K and the seed sd move with K/n, in the SAME direction:\n"
              "  a looser cap buys headroom AND cuts deeper into the contested\n"
              "  middle. The ratio is the only thing worth reading.\n")
    if ccp is None and noise is None:
        out.write("  p@K and sd are interpolated from the iwildcam curve "
                  "(iwc3, 36 clip runs).\n"
                  "  !! THEY DO NOT TRANSFER. On a new dataset use them to see "
                  "WHERE the ratio\n"
                  "     could become measurable, then measure p and sd there.\n")
    if borrowed:
        out.write("  *** BORROWED CALIBRATION: this is not iwildcam and its own "
                  "p@K and sd\n"
                  "      have never been measured, so NO CELL BELOW CAN BE "
                  "KILLED HERE. The\n"
                  "      `needs p@K` column is what to go and measure; a cell "
                  "clears the bar\n"
                  "      if this dataset's real p@K is at or below it.\n")
    out.write("\n")
    out.write("  %-10s %6s %7s %8s %8s %8s %8s %7s %8s  %s\n"
              % ("cap", "class", "n", "K", "K/n", "p@K", "prize", "sd",
                 "prize/sd", "verdict"))
    worth = 0
    for tag, c, n, kg, kl, k in rows:
        ratio = k / float(n) if n else 0.0
        ceiling = 2.0 * k / (k + n) if (k + n) else 0.0
        cal_p, cal_sd, clamped = calibrated(ratio)
        p = cal_p if ccp is None else ccp
        sd = cal_sd if noise is None else noise
        prize = (1.0 - p) * k
        rel = (prize / sd) if sd else float("inf") if prize else 0.0
        # The p@K at which the prize would clear twice this sd.
        needed = (1.0 - 2.0 * sd / k) if k else float("nan")
        if borrowed:
            verdict = "UNPRICED HERE, needs p@K <= %.4f" % needed
        elif rel >= 2.0:
            verdict = "WORTH RUNNING"
            worth += 1
        elif rel >= 1.0:
            verdict = "marginal"
        else:
            verdict = "*** PRIZE BELOW THE NOISE"
        out.write("  %-10s %6d %7d %8d %7.1f%% %8.4f %8.2f %7.2f %7.2fx  %s\n"
                  % (tag, c, n, k, 100.0 * ratio, p, prize, sd, rel, verdict))
        out.write("             ^ ceiling %.4f" % ceiling)
        if kg != k or kl != k:
            out.write("   global K=%d, local sum K=%d, BINDING K=%d"
                      % (kg, kl, k))
        if clamped and ccp is None and noise is None:
            out.write("   !! K/n %.2f is OUTSIDE the measured 0.20-0.90, so p "
                      "and sd are the ENDPOINT, not an interpolation"
                      % ratio)
        out.write("\n")
    if borrowed:
        out.write("\n  *** NOTHING WAS DECIDED. Every verdict above is "
                  "iwildcam's, applied to a\n"
                  "      dataset that has never been measured. Measure p@K "
                  "with a finished\n"
                  "      unconstrained run, then re-run with --ccp/--noise.\n")
    elif not worth:
        out.write("\n  *** NO (cap, class) CELL HAS A PRIZE WORTH TWICE THE "
                  "SEED NOISE.\n")
        out.write("      A method would have to capture the WHOLE gap to a "
                  "perfect ranking\n"
                  "      to show here, and every trained arm still shares a "
                  "backbone with the\n"
                  "      UNCAPPED classes, where 2(s) measures only downside. "
                  "Raise K/n, or\n"
                  "      change the dataset -- and re-measure p and sd wherever "
                  "you move to.\n")
    # ASCII only, deliberately. The Windows console is cp1252 and a single
    # emoji in a print raises UnicodeEncodeError *mid-report*, so the table
    # above is printed and the process dies with exit 1 on the line after it.
    # It happened to this script on its first run.
    out.write("\n  !! p is not independent of K/n: a small budget is filled "
              "from a large\n"
              "    pool of positives and is correct almost by default. Read "
              "the two\n"
              "    columns together, and re-measure p if K/n here is far from "
              "iwildcam's\n"
              "    16-30%%.\n")
    return worth


def self_test(out=sys.stdout):
    """The gate. A screen that says yes to everything decides nothing."""
    import io as _io
    ok = True

    # iwildcam L30_G50 class 2, the measured case: K=111, n=370 -> 0.51 items.
    buf = _io.StringIO()
    worth = report([("L30_G50", 2, 370, 185, 111, 111)], out=buf)
    text = buf.getvalue()
    if worth != 0 or "PRIZE BELOW THE NOISE" not in text:
        out.write("SELF-TEST FAIL: iwildcam's measured cell must not read as "
                  "worth running:\n%s" % text)
        ok = False
    if "0.4615" not in text:
        out.write("SELF-TEST FAIL: ceiling 2*111/(111+370) = 0.4615 not "
                  "printed:\n%s" % text)
        ok = False
    if "global K=185" not in text:
        out.write("SELF-TEST FAIL: an INERT global must be named, or the 30x "
                  "bug can come back silently:\n%s" % text)
        ok = False

    # THE CALIBRATION MUST MOVE. A fixed p said iwildcam had no prize at ANY
    # cap, which is false and was the first thing this tool got wrong. At
    # K/n = 0.81 the curve gives p ~ 0.952 and the prize is ~14 items, an
    # order of magnitude above what a constant 0.9954 reports.
    buf = _io.StringIO()
    report([("L80_G80", 2, 370, 300, 300, 300)], out=buf)
    text = buf.getvalue()
    row = [l for l in text.splitlines() if "L80_G80" in l][0]
    prize_loose = float(row.split()[6])
    if not (10.0 < prize_loose < 20.0):
        out.write("SELF-TEST FAIL: at K/n=0.81 the prize should be ~14 items, "
                  "got %.2f. The p@K calibration is not being applied:\n%s"
                  % (prize_loose, text))
        ok = False

    # ... and the NOISE must move with it, or a loose cap looks free.
    if float(row.split()[7]) < 5.0:
        out.write("SELF-TEST FAIL: the seed sd at K/n=0.81 is ~9.7 items, not "
                  "%s. Quoting a loose-cap prize against a tight-cap sd "
                  "overstates it up to 6x:\n%s" % (row.split()[7], text))
        ok = False

    # The screen must be able to say YES, or it is not a screen. A worse
    # ranking makes the same budget worth chasing.
    buf = _io.StringIO()
    worth = report([("L80_G80", 2, 370, 300, 300, 300)], ccp=0.90, noise=3.0,
                   out=buf)
    if worth != 1 or "WORTH RUNNING" not in buf.getvalue():
        out.write("SELF-TEST FAIL: at p=0.90 against sd 3.0 a K=300 budget is "
                  "30 items and must read as worth running:\n%s"
                  % buf.getvalue())
        ok = False

    # A BORROWED CURVE MUST NOT KILL. iwildcam's own p@K is 0.9948-0.9972, so
    # every foreign dataset inherits `PRIZE BELOW THE NOISE` from a number
    # nobody measured on it -- and FRAMEWORK 2(w2) prices fmow at p@K <= 0.92,
    # a bar iwildcam's curve can neither pass nor test.
    buf = _io.StringIO()
    worth = report([("L30_G50", 2, 370, 185, 111, 111)], out=buf, native=False)
    text = buf.getvalue()
    if "PRIZE BELOW THE NOISE" in text or worth:
        out.write("SELF-TEST FAIL: a borrowed calibration must not return a "
                  "kill verdict:\n%s" % text)
        ok = False
    if "needs p@K <= " not in text or "NOTHING WAS DECIDED" not in text:
        out.write("SELF-TEST FAIL: the borrowed case must print the p@K to go "
                  "and measure, and say nothing was decided:\n%s" % text)
        ok = False
    # ... and the bar it prints must be the one that clears 2x the sd.
    need = float(text.split("needs p@K <= ")[1].split()[0])
    for delta, want in ((-1e-3, 1), (+1e-3, 0)):
        buf2 = _io.StringIO()
        got = report([("L30_G50", 2, 370, 185, 111, 111)], ccp=need + delta,
                     noise=6.35, out=buf2)
        if got != want:
            out.write("SELF-TEST FAIL: at p@K = %.4f (bar %.4f) the cell must "
                      "read worth=%d, got %d:\n%s"
                      % (need + delta, need, want, got, buf2.getvalue()))
            ok = False

    # THE CLAMP MUST ANNOUNCE ITSELF. The live per-class caps run to K/n = 1.00
    # and the curve stops at 0.90, so the endpoint is returned as if measured.
    if calibrated(0.95)[2] is not True or calibrated(0.50)[2] is not False:
        out.write("SELF-TEST FAIL: calibrated() must report whether it "
                  "extrapolated\n")
        ok = False
    buf = _io.StringIO()
    report([("L100_G95", 2, 370, 370, 370, 370)], out=buf)
    if "OUTSIDE the measured" not in buf.getvalue():
        out.write("SELF-TEST FAIL: a K/n outside 0.20-0.90 must say so:\n%s"
                  % buf.getvalue())
        ok = False

    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slice_dir", nargs="?",
                    help="a slice dir holding test_meta.csv")
    ap.add_argument("--caps", nargs="+", default=["L20_G50", "L30_G50"],
                    help="cap tags, e.g. L30_G50")
    ap.add_argument("--classes", nargs="+", type=int, default=[2, 7],
                    help="the capped classes")
    ap.add_argument("--group-column", default="location",
                    help="the group column in test_meta.csv")
    ap.add_argument("--num-classes", type=int, default=8)
    ap.add_argument("--ccp", type=float, default=None,
                    help="override p@K with a constant. Default: interpolate "
                         "the measured K/n curve, because p FALLS as the "
                         "budget grows and a constant said iwildcam had no "
                         "prize at any cap")
    ap.add_argument("--noise", type=float, default=None,
                    help="override the seed sd with a constant, in items. "
                         "Default: interpolate the measured curve -- the noise "
                         "grows with K too")
    ap.add_argument("--native-calibration", action="store_true",
                    help="the built-in curve WAS measured on this dataset. "
                         "Inferred for any path under iwildcam; pass it "
                         "explicitly only if you measured the curve yourself")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.slice_dir:
        ap.error("give a slice dir, or --self-test")
    meta = os.path.join(args.slice_dir, "test_meta.csv")
    if not os.path.exists(meta):
        print("no test_meta.csv under %s" % args.slice_dir)
        return 2

    native = (args.native_calibration
              or "iwildcam" in args.slice_dir.replace(chr(92), "/").lower())
    rows = budgets(meta, args.caps, args.classes, args.group_column,
                   args.num_classes)
    worth = report(rows, ccp=args.ccp, noise=args.noise, native=native)
    if not native and args.ccp is None and args.noise is None:
        return 3          # UNDECIDED: 1 would be a kill this tool cannot make
    return 0 if worth else 1


if __name__ == "__main__":
    sys.exit(main())
