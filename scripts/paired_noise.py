"""FOUR different noise numbers exist here. Quoting the wrong one is a result.

WHY THIS EXISTS. `ceiling_screen` prices a direction as `prize / sd`, and the
first two versions of that ratio were both wrong -- not in the prize, in the
`sd`. The prize is a property of K and the ranking; the noise is a property of
the CONTRAST you intend to run, and this project runs exactly one kind:
seed-paired against the arm's OWN lambda=0 twin.

Pairing normally shrinks noise, so the natural assumption is that the paired sd
is the small one and an unpaired quote is conservative. **On this design the
opposite is true**, and by a factor of 6 to 12. `tralo` and `tralo_null` share
ONE warm-up epoch and then train 29 more apart. They are two models, not two
readings of one model, so the pairing cancels almost nothing and ADDS the
variance of a second training run.

    unpaired   sd of one arm's TP@K across seeds, within cell.
               What an ABSOLUTE quality claim faces.
    reseed     sd of (reseed_arm - control). RNG stream perturbed and nothing
               else, so this is the floor under ANY paired contrast, and the
               honest bar for a new arm.
    treated    sd of (treated_arm - control). What the contrast you are
               actually running faces. Always the largest of the three here.

    (the fourth is `full_panel`'s `paired seed sd`, in `d ccF1` MACRO-averaged
     over both capped classes and converted through `(K+n)/2`. It is a
     different quantity in different units -- do not substitute it for these.)

Measured on `results/iwc3`, class 2: at K/n = 0.2 the prize is 0.42 items
against an unpaired sd of 0.80 (0.52x) but a treated sd of 7.59 (**0.05x**).
The direction that looked marginal was never close. FRAMEWORK 2(v).

    python -m scripts.paired_noise --campaign results/iwc3
    python -m scripts.paired_noise --campaign results/iwc3 --classes 2 7
    python -m scripts.paired_noise --self-test
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

COLUMNS = ["cell", "seed", "cls", "frac", "n", "k", "tp"]
DEFAULT_FRACS = [0.2, 0.3, 0.5, 0.7, 0.8, 0.9]


def load_arm(root, arm, classes, fracs):
    """TP@K for every (cell, seed, class, K/n) this arm produced.

    K comes from the LABELS and the requested fraction, not from the cap
    policy: the point here is to sweep K/n past the caps the protocol actually
    uses, so the noise curve can be read where a looser budget would put it.
    """
    rows = []
    pattern = os.path.join(root, "**", arm, "seed_*",
                           "final_predictions_raw.csv")
    for path in sorted(glob.glob(pattern, recursive=True)):
        parts = path.replace(os.sep, "/").split("/")
        cell, seed = "/".join(parts[-6:-3]), parts[-2]
        df = pd.read_csv(path)
        y = df["True_Label"].values
        for c in classes:
            col = "Prob_Class_%d" % c
            if col not in df.columns:
                continue
            p = df[col].values
            n = int((y == c).sum())
            if n == 0:
                continue
            cum = np.cumsum((y[np.argsort(-p)] == c).astype(int))
            for f in fracs:
                k = max(1, min(int(round(f * n)), len(cum)))
                rows.append((cell, seed, c, f, n, k, int(cum[k - 1])))
    return pd.DataFrame(rows, columns=COLUMNS)


def paired_sd(treated, control):
    """{(cls, frac): sd of (treated - control)}, pooled over cells.

    Pooled as the root-mean-square of the WITHIN-cell sds, never as one sd over
    the flattened set: a cell-to-cell mean shift is not noise, and pooling
    across it would inflate every figure here (house rule 4).
    """
    key = ["cell", "seed", "cls", "frac"]
    m = treated.merge(control, on=key, suffixes=("_a", "_b"))
    m["d"] = m.tp_a - m.tp_b
    out = {}
    for (c, f), grp in m.groupby(["cls", "frac"]):
        sds = grp.groupby("cell").d.std(ddof=1).dropna()
        out[(c, f)] = float(np.sqrt((sds ** 2).mean())) if len(sds) else np.nan
    return out


def unpaired_sd(df):
    """{(cls, frac): sd of TP@K across seeds}, pooled over cells the same way."""
    out = {}
    for (c, f), grp in df.groupby(["cls", "frac"]):
        sds = grp.groupby("cell").tp.std(ddof=1).dropna()
        out[(c, f)] = float(np.sqrt((sds ** 2).mean())) if len(sds) else np.nan
    return out


def prizes(bar):
    """{(cls, frac): (K, p@K, prize items)} from the quality bar's own runs.

    `prize = (1 - p@K) * K` is the whole gap to a PERFECT ranking at that
    budget: no loss, dual, allocator or optimizer can win more than the items
    the current ranking has wrong inside the top K.
    """
    out = {}
    for (c, f), grp in bar.groupby(["cls", "frac"]):
        k = float(grp.k.mean())
        p = float((grp.tp / grp.k).mean())
        out[(c, f)] = (k, p, (1.0 - p) * k)
    return out


def report(bar, floor_sd, treated_sd, classes, fracs, out=sys.stdout):
    """Print the table. Returns the number of (cls, frac) rows where the prize
    clears the TREATED noise, i.e. rows where a method could show something."""
    un = unpaired_sd(bar)
    pr = prizes(bar)
    worth = 0
    out.write("  %-4s %-6s %7s %8s %9s %9s %9s %9s %9s\n"
              % ("cls", "K/n", "K", "prize", "unpaired", "reseed", "treated",
                 "pr/reseed", "pr/treated"))
    for c in classes:
        for f in fracs:
            if (c, f) not in pr:
                continue
            k, p, prize = pr[(c, f)]
            u = un.get((c, f), float("nan"))
            a = floor_sd.get((c, f), float("nan"))
            b = treated_sd.get((c, f), float("nan"))
            ra = prize / a if a else float("nan")
            rb = prize / b if b else float("nan")
            if rb == rb and rb >= 1.0:
                worth += 1
            out.write("  %-4d %-5.0f%% %7.0f %8.2f %9.2f %9.2f %9.2f "
                      "%8.2fx %8.2fx\n"
                      % (c, 100 * f, k, prize, u, a, b, ra, rb))
        out.write("\n")
    out.write("  reseed = RNG only, the floor under ANY paired contrast.\n"
              "  treated = the contrast actually run. If `treated` exceeds\n"
              "  `unpaired`, pairing is COSTING resolution, not buying it --\n"
              "  the two arms are two models rather than two readings of one.\n")
    if not worth:
        out.write("\n  NO row has a prize at or above the treated noise: at\n"
                  "  every K/n here, a method capturing 100%% of the gap to a\n"
                  "  perfect ranking would still not be detectable.\n")
    return worth


def _synth(offsets, cells=3, seeds=4, cls=2, frac=0.2, n=100, k=20):
    """Build one arm's frame. `offsets[(cell_i, seed_i)]` gives its TP."""
    rows = []
    for ci in range(cells):
        for si in range(seeds):
            rows.append(("cell%d" % ci, "seed_%d" % si, cls, frac, n, k,
                         offsets(ci, si)))
    return pd.DataFrame(rows, columns=COLUMNS)


def self_test(out=sys.stdout):
    """The gate. A tool that can only ever report one answer is not a measurement.

    The load-bearing claim from this script is "pairing GROWS the noise here".
    That is only meaningful if the script CAN report a shrink, so the first
    case below is the liveness control: two arms differing by a constant per
    cell must come back with a paired sd of 0 against a large unpaired one.
    """
    ok = True

    # 1. LIVENESS: pairing must be able to help. Same seed-to-seed structure,
    #    offset by a constant within each cell => the difference is constant,
    #    so paired sd is exactly 0 while unpaired sd is large.
    base = _synth(lambda ci, si: 10 + 7 * si)
    same = _synth(lambda ci, si: 10 + 7 * si + 3)
    ps = paired_sd(same, base)[(2, 0.2)]
    us = unpaired_sd(base)[(2, 0.2)]
    if not (abs(ps) < 1e-9 and us > 1.0):
        out.write("SELF-TEST FAIL: pairing must be able to CANCEL shared "
                  "variation. paired=%.4f unpaired=%.4f\n" % (ps, us))
        ok = False

    # 2. The headline case: two INDEPENDENT arms. Paired sd must come back
    #    LARGER than either unpaired sd, by about sqrt(2).
    a = _synth(lambda ci, si: [10, 20, 30, 40][si])
    b = _synth(lambda ci, si: [40, 10, 30, 20][si])
    ps = paired_sd(a, b)[(2, 0.2)]
    ua = unpaired_sd(a)[(2, 0.2)]
    if not ps > ua:
        out.write("SELF-TEST FAIL: independent arms must give a paired sd "
                  "ABOVE the unpaired one. paired=%.4f unpaired=%.4f\n"
                  % (ps, ua))
        ok = False

    # 3. It must recover a KNOWN sd, not merely order two numbers.
    a = _synth(lambda ci, si: 100 + [0, 2, 4, 6][si])
    z = _synth(lambda ci, si: 100)
    want = float(np.std([0, 2, 4, 6], ddof=1))
    got = paired_sd(a, z)[(2, 0.2)]
    if abs(got - want) > 1e-9:
        out.write("SELF-TEST FAIL: known sd not recovered: got %.6f want "
                  "%.6f\n" % (got, want))
        ok = False

    # 4. Pooling must be WITHIN cell. A pure cell-to-cell shift is not noise
    #    and must not appear as any.
    shifted = _synth(lambda ci, si: 100 + 50 * ci)
    flat = _synth(lambda ci, si: 100)
    got = paired_sd(shifted, flat)[(2, 0.2)]
    if abs(got) > 1e-9:
        out.write("SELF-TEST FAIL: a cell-to-cell mean shift leaked into the "
                  "sd: %.6f, expected 0\n" % got)
        ok = False

    # 5. The verdict must be able to say YES. A screen that can only refuse
    #    decides nothing.
    bar = _synth(lambda ci, si: 5 + si)            # p@K ~ 0.32 => big prize
    worth = report(bar, {(2, 0.2): 0.5}, {(2, 0.2): 0.5}, [2], [0.2],
                   out=open(os.devnull, "w"))
    if worth != 1:
        out.write("SELF-TEST FAIL: a large prize against a small noise must "
                  "count as worth running; got %d\n" % worth)
        ok = False

    out.write("SELF-TEST %s\n" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", help="campaign root, e.g. results/iwc3")
    ap.add_argument("--bar", default="clip",
                    help="arm the prize is measured off (default clip)")
    ap.add_argument("--control", default="tralo_null",
                    help="the lambda=0 twin both contrasts pair against")
    ap.add_argument("--floor", default="tralo_reseed",
                    help="RNG-only arm giving the noise floor")
    ap.add_argument("--treated", default="tralo",
                    help="the arm whose contrast is actually run")
    ap.add_argument("--classes", type=int, nargs="+", default=[2, 7])
    ap.add_argument("--fracs", type=float, nargs="+", default=DEFAULT_FRACS,
                    help="K/n levels to sweep (default 0.2 .. 0.9)")
    ap.add_argument("--self-test", action="store_true",
                    help="check the tool against known-answer inputs")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.campaign:
        ap.error("give --campaign, or --self-test")
    if not os.path.isdir(args.campaign):
        print("no such campaign root: %s" % args.campaign)
        return 2

    frames = {}
    for name in (args.bar, args.control, args.floor, args.treated):
        frames[name] = load_arm(args.campaign, name, args.classes, args.fracs)
    per = len(args.classes) * len(args.fracs)
    print("runs: " + "  ".join(
        "%s %d" % (n, len(f) // per if per else 0) for n, f in frames.items()))

    missing = [n for n, f in frames.items() if f.empty]
    if missing:
        print("REFUSING: no runs found for %s. This tool measures the noise a\n"
              "  PAIRED contrast faces, so it cannot fall back to an unpaired\n"
              "  number -- that substitution is the defect it exists to catch."
              % ", ".join(missing))
        return 2

    print("")
    worth = report(frames[args.bar],
                   paired_sd(frames[args.floor], frames[args.control]),
                   paired_sd(frames[args.treated], frames[args.control]),
                   args.classes, args.fracs)
    return 0 if worth else 1


if __name__ == "__main__":
    sys.exit(main())
